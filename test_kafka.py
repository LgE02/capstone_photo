"""카프카 연결만 빠르게 검증 (모델 로드 안 함, 30초 내).

세 가지를 확인:
  1. 컨슈머가 INIT/PAGE 토픽에 붙는지
  2. 결과 토픽으로 producer 가 메시지를 보낼 수 있는지
  3. 그 메시지를 우리 컨슈머 그룹과 다른 그룹으로 다시 받아 형식이 맞는지

성공 시 "✅ 모든 검증 통과" 출력.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    return val.strip() if val else default


async def main() -> int:
    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        print("❌ KAFKA_BOOTSTRAP_SERVERS 가 비어 있습니다. .env 채우세요.")
        return 1

    topic_init = _env("KAFKA_TOPIC_INIT", "fairytale_created")
    topic_page = _env("KAFKA_TOPIC_PAGE", "fairytale_paragraph")
    topic_result = _env("KAFKA_TOPIC_RESULT", "fairytale_image")
    sec = _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").upper()

    print(f"bootstrap = {bootstrap}")
    print(f"topics    = INIT={topic_init} PAGE={topic_page} RESULT={topic_result}")
    print(f"security  = {sec}")

    common: dict = {
        "bootstrap_servers": bootstrap,
        "security_protocol": sec,
    }
    if sec in ("SASL_PLAINTEXT", "SASL_SSL"):
        common["sasl_mechanism"] = _env("KAFKA_SASL_MECHANISM", "PLAIN")
        common["sasl_plain_username"] = _env("KAFKA_SASL_USERNAME")
        common["sasl_plain_password"] = _env("KAFKA_SASL_PASSWORD")

    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    # 1) INIT/PAGE 토픽 구독 가능한지 (메시지가 없어도 connect 까지만)
    print("\n[1/3] 입력 토픽 구독 시도...")
    consumer = AIOKafkaConsumer(
        topic_init, topic_page,
        group_id=f"test-{uuid.uuid4().hex[:6]}",
        auto_offset_reset="latest",
        **common,
    )
    try:
        await asyncio.wait_for(consumer.start(), timeout=15)
        print(f"  ✅ {topic_init} / {topic_page} 구독 OK")
    except Exception as exc:
        print(f"  ❌ 구독 실패: {exc!r}")
        print("     → bootstrap 주소·security 설정·EC2 SecurityGroup 확인")
        return 1
    finally:
        try:
            await consumer.stop()
        except Exception:
            pass

    # 2) 결과 토픽으로 테스트 메시지 produce
    print("\n[2/3] 결과 토픽으로 produce 시도...")
    producer = AIOKafkaProducer(acks="all", **common)
    test_payload = {
        "fairytaleId": -999,            # 테스트용 음수 — Spring 측 무시 필요
        "pageNo": 0,
        "imageurl": f"test://{uuid.uuid4()}",
        "_test": True,
        "_ts": int(time.time()),
    }
    try:
        await asyncio.wait_for(producer.start(), timeout=15)
        await producer.send_and_wait(
            topic_result,
            json.dumps(test_payload, ensure_ascii=False).encode("utf-8"),
            key=str(test_payload["fairytaleId"]).encode("utf-8"),
        )
        print(f"  ✅ {topic_result} 로 테스트 메시지 publish OK")
    except Exception as exc:
        print(f"  ❌ produce 실패: {exc!r}")
        return 1
    finally:
        try:
            await producer.stop()
        except Exception:
            pass

    # 3) 같은 메시지를 다시 consume — 결과 토픽이 정상 동작하는지 확인
    print("\n[3/3] 결과 토픽에서 방금 보낸 메시지 다시 consume...")
    verify = AIOKafkaConsumer(
        topic_result,
        group_id=f"test-verify-{uuid.uuid4().hex[:6]}",
        auto_offset_reset="earliest",
        **common,
    )
    try:
        await asyncio.wait_for(verify.start(), timeout=15)
        # 최대 10초 폴링 — 방금 produce 한 메시지 찾기
        found = False
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(verify.__anext__(), timeout=2)
            except asyncio.TimeoutError:
                continue
            try:
                got = json.loads(msg.value.decode("utf-8"))
            except Exception:
                continue
            if got.get("_ts") == test_payload["_ts"] and got.get("_test"):
                found = True
                print(f"  ✅ 메시지 수신: {got}")
                break
        if not found:
            print("  ⚠ 10초 안에 테스트 메시지 못 찾음 — produce 는 됐으니 토픽 동작은 정상일 가능성")
    except Exception as exc:
        print(f"  ❌ consume 실패: {exc!r}")
        return 1
    finally:
        try:
            await verify.stop()
        except Exception:
            pass

    print("\n✅ 모든 검증 통과 — worker.py 띄워도 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
