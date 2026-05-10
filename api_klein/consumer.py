"""카프카 컨슈머 — Spring 팀이 publish 한 메시지를 끌어와 processor 에 전달.

토픽 2개를 동시 구독:
  KAFKA_TOPIC_INIT (예: fairytale_created)   → 캐릭터 정보, bible/레퍼런스 준비
  KAFKA_TOPIC_PAGE (예: fairytale_paragraph) → 페이지 단위 삽화 생성 요청

같은 group_id 로 두 토픽을 한 컨슈머가 받으므로 GPU 1개 + 단일 컨슈머 가정 그대로 유지.
메시지 타입은 토픽 이름으로 판별 (가장 명확) 후 processor 에 전달.

기본 동기 처리 (메시지 1개 끝나야 다음 메시지). 추후 throughput 부족 시 워커
수평 확장 (파티션 ≥ 워커). PAGE 토픽의 partition key 가 fairytaleId 면 같은
동화의 페이지가 같은 워커로 가서 bible 캐시 hit + 순서 유지.

실패 처리 정책:
  - 파싱 실패 (깨진 JSON): poison message — 로그 + commit (재시도해도 똑같이 깨짐).
  - BibleNotReadyError (PAGE 가 INIT 보다 먼저 도착): 짧은 백오프 + commit 보류
    → 같은 메시지 재시도. 토픽이 분리돼 있어 INIT/PAGE 순서가 보장되지 않으므로
    실제로 발생할 수 있음 — 재시도 횟수·대기 시간을 넉넉히 잡아 둠.
  - 그 외 처리 실패 (모델/S3/Redis 등): 지수 백오프 + 재시도, MAX_RETRIES 초과
    시 commit 으로 흘려보내고 에러 로그 (메시지 유실 방지를 원하면 DLQ 토픽 추가).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer, TopicPartition

from api_klein.processor import BibleNotReadyError


MAX_RETRIES = 5  # PAGE 가 INIT 보다 먼저 도착 시 INIT 처리 끝날 때까지 충분히 대기
BIBLE_WAIT_SECONDS = 5  # PAGE 가 INIT 보다 먼저 왔을 때 재시도 전 대기

# 메시지 핸들러 시그니처: (topic, payload) → Any
# topic 이름으로 INIT/PAGE 분기를 하므로 payload 만 받지 않고 topic 도 함께 전달.
HandlerType = Callable[[str, dict[str, Any]], Any | Awaitable[Any]]


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    return val.strip() if val else default


def _build_consumer() -> tuple[AIOKafkaConsumer, list[str]]:
    """환경변수에서 카프카 연결 정보를 읽어 컨슈머 생성.

    Returns:
        (consumer, [구독_토픽_리스트])
    """
    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS 환경변수가 비어있습니다.")

    topic_init = _env("KAFKA_TOPIC_INIT", "fairytale_created")
    topic_page = _env("KAFKA_TOPIC_PAGE", "fairytale_paragraph")
    topics = [topic_init, topic_page]
    group_id = _env("KAFKA_GROUP_ID", "illustration-worker")
    security_protocol = _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").upper()

    #컨슈머 그룹이 처음 메시지를 읽을 때, 
    # 가장 오래된 메시지부터 읽을지(latest) 아니면 가장 최근 메시지부터 읽을지(earliest) 설정
    kwargs: dict[str, Any] = {
        "bootstrap_servers": bootstrap,
        "group_id": group_id,
        "auto_offset_reset": "earliest", 
        "enable_auto_commit": False,  # 처리 성공 후 수동 commit
        "max_poll_records": 1,        # 한 번에 1개씩 (단일 컨슈머 + GPU)
        "security_protocol": security_protocol,
    }

    if security_protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
        kwargs["sasl_mechanism"] = _env("KAFKA_SASL_MECHANISM", "PLAIN")
        kwargs["sasl_plain_username"] = _env("KAFKA_SASL_USERNAME")
        kwargs["sasl_plain_password"] = _env("KAFKA_SASL_PASSWORD")

    return AIOKafkaConsumer(*topics, **kwargs), topics


async def run_consumer_loop(handle_message: HandlerType) -> None:
    """카프카에서 메시지를 끌어와 handle_message(topic, payload) 콜백에 전달.

    handle_message 가 동기 함수면 asyncio.to_thread 로 별도 스레드에서 실행
    (모델 추론이 GIL 을 잡고 이벤트 루프 막지 않게 — 카프카 heartbeat 유지).
    """
    consumer, topics = _build_consumer()
    await consumer.start()
    print(
        f"[consumer] 카프카 연결: {_env('KAFKA_BOOTSTRAP_SERVERS')} / "
        f"topics={topics} / group_id={_env('KAFKA_GROUP_ID')}"
    )

    # (partition, offset) → 누적 재시도 횟수.
    # 같은 offset 을 commit 없이 다시 받으면 같은 키가 나옴 → 횟수 누적.
    retry_counts: dict[tuple[int, int], int] = {}

    try:
        async for msg in consumer:
            key = (msg.partition, msg.offset)

            # 1) 파싱 — 깨진 JSON 은 poison message, 즉시 흘려보냄
            try:
                payload = json.loads(msg.value.decode("utf-8"))
            except Exception as exc:
                print(f"[consumer] 파싱 실패 partition={msg.partition} offset={msg.offset}: {exc}")
                await consumer.commit()
                retry_counts.pop(key, None)
                continue

            fid = payload.get("fairytaleId") or payload.get("fairytale_id")
            pno = payload.get("pageNo") or payload.get("page", "-")
            print(
                f"[consumer] 수신 topic={msg.topic} partition={msg.partition} offset={msg.offset} "
                f"fairytaleId={fid} pageNo={pno}"
            )
            if fid is None:
                print(f"  ↳ payload keys = {list(payload.keys())}")
                print(f"  ↳ raw payload  = {payload}")

            # 2) 처리 — 토픽 이름을 함께 전달해 INIT/PAGE 분기에 사용
            try:
                if asyncio.iscoroutinefunction(handle_message):
                    await handle_message(msg.topic, payload)
                else:
                    await asyncio.to_thread(handle_message, msg.topic, payload)

                await consumer.commit()
                retry_counts.pop(key, None)
                print(f"[consumer] 처리 완료, offset commit")

            except BibleNotReadyError as exc:
                # PAGE 가 INIT 보다 먼저 왔음 — commit 안 하고 잠시 대기 후 재시도.
                # partition key 가 제대로 박혀 있으면 거의 발생 안 함.
                count = retry_counts.get(key, 0) + 1
                retry_counts[key] = count
                print(
                    f"[consumer] bible 미준비 ({exc}) — {BIBLE_WAIT_SECONDS}s 대기 후 재시도 "
                    f"({count}/{MAX_RETRIES})"
                )
                if count >= MAX_RETRIES:
                    # INIT 가 영원히 안 올 가능성 — 메시지 유실 방지하려면 DLQ 추가
                    print(f"[consumer] 재시도 한계 도달 — 메시지 흘려보냄 (offset={msg.offset})")
                    await consumer.commit()
                    retry_counts.pop(key, None)
                else:
                    await asyncio.sleep(BIBLE_WAIT_SECONDS)
                    # commit 안 했어도 다음 poll 은 다음 offset 으로 진행되므로
                    # 같은 메시지를 다시 처리하려면 명시적으로 seek 해야 함.
                    consumer.seek(TopicPartition(msg.topic, msg.partition), msg.offset)

            except Exception as exc:
                count = retry_counts.get(key, 0) + 1
                retry_counts[key] = count
                print(
                    f"[consumer] 처리 실패 ({count}/{MAX_RETRIES}) "
                    f"partition={msg.partition} offset={msg.offset}: {exc!r}"
                )
                if count >= MAX_RETRIES:
                    print(f"[consumer] 재시도 한계 도달 — 메시지 흘려보냄 (offset={msg.offset})")
                    await consumer.commit()
                    retry_counts.pop(key, None)
                else:
                    backoff = 2 ** count
                    print(f"[consumer] {backoff}s 백오프 후 재시도")
                    await asyncio.sleep(backoff)
                    consumer.seek(TopicPartition(msg.topic, msg.partition), msg.offset)
    finally:
        await consumer.stop()
        print("[consumer] 카프카 연결 종료")
