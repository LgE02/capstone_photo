"""카프카 결과 publisher — 페이지 삽화 생성 완료 결과를 Spring 으로 전송.

발송 토픽 (KAFKA_TOPIC_RESULT, 기본 'fairytale_image') 페이로드:
    {
      "fairytaleId": 17,
      "pageNo": 3,
      "imageurl": "s3://bucket/.../page_03.png"
    }

key 는 fairytaleId 로 박아 같은 동화의 페이지가 같은 파티션으로 가게 함
(Spring 측 컨슈머가 순서 보장 필요하면 활용 가능).

이 모듈은 동기 인터페이스 (publish_page_complete) 를 노출 — processor 가
asyncio.to_thread 안에서 호출하므로 내부적으로 별도 이벤트 루프에서 producer
를 돌려 완료까지 대기시킨다.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from aiokafka import AIOKafkaProducer


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    return val.strip() if val else default


class KafkaResultPublisher:
    """카프카 결과 publisher — 단일 토픽으로 결과 메시지 송신.

    동기 인터페이스를 제공하지만 내부는 aiokafka(async). processor 가 동기 함수라
    별도 백그라운드 이벤트 루프 스레드에서 producer 를 돌리고, 동기 호출 측은
    Future.result() 로 완료를 기다린다.
    """

    def __init__(
        self,
        bootstrap: str | None = None,
        topic: str | None = None,
        security_protocol: str | None = None,
    ):
        self.bootstrap = bootstrap or _env("KAFKA_BOOTSTRAP_SERVERS")
        self.topic = topic or _env("KAFKA_TOPIC_RESULT", "fairytale_image")
        self.dlq_topic = _env("KAFKA_TOPIC_DLQ", "fairytale_dlq")
        self.security_protocol = (security_protocol or _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")).upper()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._producer: AIOKafkaProducer | None = None
        self._started = False
        self._lock = threading.Lock()

    # ── 라이프사이클 ───────────────────────────────────────────────────────

    def start(self) -> None:
        """백그라운드 이벤트 루프 + AIOKafkaProducer 기동. 한 번만 호출."""
        with self._lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_loop, name="kafka-publisher-loop", daemon=True
            )
            self._loop_thread.start()

            # producer 생성·start 를 백그라운드 루프에서 실행 후 대기
            fut = asyncio.run_coroutine_threadsafe(self._async_start(), self._loop)
            fut.result(timeout=30)
            self._started = True
            print(f"[publisher] 카프카 producer 시작: topic={self.topic}")

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_start(self) -> None:
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self.bootstrap,
            "security_protocol": self.security_protocol,
            "acks": "all",            # 모든 ISR ack — 최대 내구성
            "enable_idempotence": True,
        }
        if self.security_protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
            kwargs["sasl_mechanism"] = _env("KAFKA_SASL_MECHANISM", "PLAIN")
            kwargs["sasl_plain_username"] = _env("KAFKA_SASL_USERNAME")
            kwargs["sasl_plain_password"] = _env("KAFKA_SASL_PASSWORD")

        self._producer = AIOKafkaProducer(**kwargs)
        await self._producer.start()

    def stop(self) -> None:
        """워커 종료 시 호출 — producer flush + 루프 종료."""
        with self._lock:
            if not self._started or self._loop is None:
                return
            try:
                fut = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
                fut.result(timeout=10)
            except Exception as exc:
                print(f"[publisher] 종료 중 예외 (무시): {exc}")
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            self._started = False
            print("[publisher] 카프카 producer 종료")

    async def _async_stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    # ── 메인 publish ───────────────────────────────────────────────────────

    def publish_page_complete(
        self,
        fairytale_id: int,
        page_no: int,
        image_url: str,
    ) -> None:
        """페이지 삽화 생성 완료 → fairytale_image 토픽에 결과 메시지 송신.

        동기 인터페이스. 내부적으로 백그라운드 루프의 producer.send_and_wait 를 호출.
        """
        if not self._started or self._loop is None or self._producer is None:
            raise RuntimeError("publisher.start() 가 먼저 호출돼야 함")

        payload = {
            "fairytaleId": fairytale_id,
            "pageNo": page_no,
            "imageurl": image_url,    # ★ Spring 측 합의 — 전부 소문자
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        key = str(fairytale_id).encode("utf-8")

        fut = asyncio.run_coroutine_threadsafe(
            self._producer.send_and_wait(self.topic, body, key=key),
            self._loop,
        )
        fut.result(timeout=15)
        print(f"[publisher] → {self.topic} fairytaleId={fairytale_id} pageNo={page_no}")

    # ── DLQ publish ────────────────────────────────────────────────────────

    def publish_dlq(
        self,
        *,
        original_topic: str,
        original_partition: int,
        original_offset: int,
        original_key: bytes | None,
        original_payload_bytes: bytes | None,
        failure_reason: str,
        last_error: str,
        retry_count: int,
    ) -> None:
        """실패 메시지를 DLQ 토픽으로 격리.

        DLQ 전송 자체가 실패해도 메인 컨슈머 루프를 죽이지 않도록 예외 흡수.
        원본 payload 는 best-effort 로 JSON 파싱, 실패 시 base64 로 보존.
        """
        if not self._started or self._loop is None or self._producer is None:
            print("[publisher] DLQ publish skip — publisher 가 시작되지 않음")
            return

        if original_payload_bytes is None:
            original_payload: Any = None
        else:
            try:
                original_payload = json.loads(original_payload_bytes.decode("utf-8"))
            except Exception:
                original_payload = {
                    "_encoding": "base64",
                    "_value": base64.b64encode(original_payload_bytes).decode("ascii"),
                }

        dlq_payload = {
            "originalTopic": original_topic,
            "originalPartition": original_partition,
            "originalOffset": original_offset,
            "originalKey": original_key.decode("utf-8", errors="replace") if original_key else None,
            "originalPayload": original_payload,
            "failureReason": failure_reason,
            "lastError": last_error,
            "retryCount": retry_count,
            "failedAt": datetime.now(timezone.utc).isoformat(),
        }
        body = json.dumps(dlq_payload, ensure_ascii=False).encode("utf-8")
        # key 는 원본 fairytaleId 가 있으면 그대로 — 같은 동화 실패가 같은 partition 으로 모이게.
        dlq_key = original_key if original_key else None

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._producer.send_and_wait(self.dlq_topic, body, key=dlq_key),
                self._loop,
            )
            fut.result(timeout=15)
            print(
                f"[publisher] → DLQ {self.dlq_topic} reason={failure_reason} "
                f"orig_topic={original_topic} offset={original_offset}"
            )
        except Exception as exc:
            # DLQ 실패가 컨슈머를 죽이면 안 됨 — 로그만 남기고 진행.
            print(f"[publisher] ⚠ DLQ publish 실패 (무시): {exc!r}")

    # ── 헬스 체크 ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """producer 가 메타데이터를 받아올 수 있으면 healthy."""
        if not self._started or self._loop is None or self._producer is None:
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._producer.client.fetch_all_metadata(), self._loop
            )
            fut.result(timeout=5)
            return True
        except Exception:
            return False
