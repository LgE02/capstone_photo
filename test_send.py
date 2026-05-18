"""Publish INIT and PAGE Kafka messages for local illustration-worker testing.

Examples:
    python test_send.py
    python test_send.py --id 999
    python test_send.py --character HERO=토끼 --character VILLAIN=거북이 --character DONOR=용왕님
    python test_send.py --sentences "토끼와 거북이는 용궁으로 가서 용왕님을 만났어요."
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


def _parse_character_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "character must use ROLE=NAME format, for example HERO=토끼"
        )

    role, name = raw.split("=", 1)
    role = role.strip().upper()
    name = name.strip()

    if not role or not name:
        raise argparse.ArgumentTypeError(
            "character must use ROLE=NAME format, for example HERO=토끼"
        )

    return role, name


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=9999, help="fairytaleId (default: 9999)")
    ap.add_argument("--page", type=int, default=1, help="pageNo (default: 1)")
    ap.add_argument("--init-only", action="store_true", help="publish only INIT")
    ap.add_argument("--page-only", action="store_true", help="publish only PAGE")
    ap.add_argument(
        "--setting",
        default="KOREAN_TRADITIONAL",
        help="INIT setting value (default: KOREAN_TRADITIONAL)",
    )
    ap.add_argument(
        "--character-type",
        default="ANIMAL",
        help="INIT character_type value (default: ANIMAL)",
    )
    ap.add_argument(
        "--character",
        action="append",
        type=_parse_character_arg,
        default=[],
        help="character mapping in ROLE=NAME format; can be repeated",
    )
    ap.add_argument(
        "--sentences",
        default="토끼와 거북이는 용궁으로 가서 용왕님을 만났어요.",
        help="PAGE sentences value; include \\n for multi-line input",
    )
    args = ap.parse_args()

    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        print("KAFKA_BOOTSTRAP_SERVERS is empty")
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
    characters = dict(args.character) if args.character else {
        "HERO": "토끼",
        "VILLAIN": "거북이",
        "DONOR": "용왕님",
    }

    try:
        if not args.page_only:
            init_payload = {
                "fairytaleId": fid,
                "setting": args.setting,
                "character_type": args.character_type.upper(),
                "characters": characters,
            }
            await producer.send_and_wait(
                topic_init,
                json.dumps(init_payload, ensure_ascii=False).encode("utf-8"),
                key=str(fid).encode("utf-8"),
            )
            print(f"INIT publish -> {topic_init}")
            print(init_payload)

        if not args.init_only:
            page_payload = {
                "fairytaleId": fid,
                "pageNo": args.page,
                "sentences": args.sentences,
            }
            await producer.send_and_wait(
                topic_page,
                json.dumps(page_payload, ensure_ascii=False).encode("utf-8"),
                key=str(fid).encode("utf-8"),
            )
            print(f"PAGE publish -> {topic_page}")
            print(page_payload)

        print("\nCheck the worker console for generation progress.")
        return 0
    finally:
        await producer.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
