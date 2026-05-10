"""테스트용 INIT + PAGE 메시지를 직접 publish.

사용법:
    python test_send.py            # 기본값으로 INIT + PAGE 1개 보냄
    python test_send.py --id 9999  # fairytaleId 지정

워커를 띄워둔 상태에서 실행하면 즉시 처리되는 걸 콘솔에서 볼 수 있음.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    return val.strip() if val else default


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=9999, help="fairytaleId (기본 9999)")
    ap.add_argument("--page", type=int, default=1, help="pageNo (기본 1)")
    ap.add_argument("--init-only", action="store_true", help="INIT 만 보내고 PAGE 는 안 보냄")
    ap.add_argument("--page-only", action="store_true", help="PAGE 만 보냄 (INIT 이미 보냈을 때)")
    args = ap.parse_args()

    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        print("❌ KAFKA_BOOTSTRAP_SERVERS 비어 있음")
        return 1

    topic_init = _env("KAFKA_TOPIC_INIT", "fairytale_created")
    topic_page = _env("KAFKA_TOPIC_PAGE", "fairytale_paragraph")
    sec = _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").upper()

    common: dict = {
        "bootstrap_servers": bootstrap,
        "security_protocol": sec,
        "acks": "all",
    }
    if sec in ("SASL_PLAINTEXT", "SASL_SSL"):
        common["sasl_mechanism"] = _env("KAFKA_SASL_MECHANISM", "PLAIN")
        common["sasl_plain_username"] = _env("KAFKA_SASL_USERNAME")
        common["sasl_plain_password"] = _env("KAFKA_SASL_PASSWORD")

    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(**common)
    await producer.start()

    fid = args.id

    try:
        # ── INIT 보내기 ───────────────────────────────────────
        if not args.page_only:
            init_payload = {
                "fairytaleId": fid,
                "setting": "KOREAN_TRADITIONAL",
                "character_type": "ANIMAL",
                "characters": {
                    "HERO": "강아지",
                    "VILLAIN": "독수리",
                    "DISPATCHER": "자라",
                },
            }
            await producer.send_and_wait(
                topic_init,
                json.dumps(init_payload, ensure_ascii=False).encode("utf-8"),
                key=str(fid).encode("utf-8"),
            )
            print(f"✅ INIT publish → {topic_init}")
            print(f"   {init_payload}")

        # ── PAGE 보내기 ───────────────────────────────────────
        if not args.init_only:
            page_payload = {
                "fairytaleId": fid,
                "pageNo": args.page,
                "sentences": (
                    "옛날 옛적에 자라가 살았어요.\n"
                    "어느 날 자라는 강아지를 만났어요.\n"
                    "둘은 함께 모험을 떠났어요."
                ),
            }
            await producer.send_and_wait(
                topic_page,
                json.dumps(page_payload, ensure_ascii=False).encode("utf-8"),
                key=str(fid).encode("utf-8"),
            )
            print(f"✅ PAGE publish → {topic_page}")
            print(f"   {page_payload}")

        print("\n워커 콘솔에서 처리되는 거 확인해 보세요.")
        return 0
    finally:
        await producer.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
