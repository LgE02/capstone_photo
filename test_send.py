"""로컬 illustration-worker 테스트용 Kafka publisher.

두 가지 모드를 지원:

  1) 단일 페이지(또는 INIT만 / PAGE만) — CLI 인자로 1건 publish
       python test_send.py
       python test_send.py --id 999
       python test_send.py --character HERO=토끼 --character VILLAIN=거북이 --character DONOR=용왕님
       python test_send.py --sentences "토끼와 거북이는 용궁으로 가서 용왕님을 만났어요."
       python test_send.py --init-only --id 999
       python test_send.py --page-only --id 999 --page 2 --sentences "..."

  2) 동화 1편 시퀀스 — JSON 파일로 INIT + 다중 PAGE 를 한 번에 publish
       python test_send.py --id 1500 --story story.json
       python test_send.py --id 1500 --story story.json --init-wait 60 --page-delay 1.0
       python test_send.py --id 1500 --story story.json --skip-init   # bible 이미 있을 때

     story.json 형식:
       {
         "setting": "FOREST_NATURE",
         "char_species": "ANIMAL",
         "characters": {"HERO": "다람쥐", "VILLAIN": "늑대", "HELPER": "고양이"},
         "pages": [
           "page 1 sentences ...",
           "page 2 sentences ...",
           ...
         ]
       }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

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


def _load_story(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("setting", "char_species", "characters", "pages"):
        if key not in data:
            raise ValueError(f"story JSON 에 '{key}' 가 없음: {path}")
    if not isinstance(data["pages"], list) or not data["pages"]:
        raise ValueError(f"story JSON 의 'pages' 는 비어있지 않은 리스트여야 함: {path}")
    return data


def _build_producer_kwargs() -> dict[str, Any]:
    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS 환경변수가 비어있음")

    sec = _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").upper()
    kwargs: dict[str, Any] = {
        "bootstrap_servers": bootstrap,
        "security_protocol": sec,
        "acks": "all",
    }
    if sec in ("SASL_PLAINTEXT", "SASL_SSL"):
        kwargs["sasl_mechanism"] = _env("KAFKA_SASL_MECHANISM", "PLAIN")
        kwargs["sasl_plain_username"] = _env("KAFKA_SASL_USERNAME")
        kwargs["sasl_plain_password"] = _env("KAFKA_SASL_PASSWORD")
    return kwargs


async def _send_init(producer, topic: str, fid: int, setting: str, char_species: str, characters: dict[str, str]) -> None:
    payload = {
        "fairytaleId": fid,
        "setting": setting,
        "char_species": char_species.upper(),
        "characters": characters,
    }
    await producer.send_and_wait(
        topic,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        key=str(fid).encode("utf-8"),
    )
    print(f"[INIT] → {topic}  fairytaleId={fid}  characters={characters}")


async def _send_page(producer, topic: str, fid: int, page_no: int, sentences: str) -> None:
    payload = {
        "fairytaleId": fid,
        "pageNo": page_no,
        "sentences": sentences,
    }
    await producer.send_and_wait(
        topic,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        key=str(fid).encode("utf-8"),
    )
    preview = sentences.split("\n")[0][:40] + ("..." if len(sentences) > 40 else "")
    print(f"[PAGE {page_no}] → {topic}  '{preview}'")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=9999, help="fairytaleId (default: 9999)")
    ap.add_argument("--page", type=int, default=1, help="단일 PAGE 모드의 pageNo (default: 1)")
    ap.add_argument("--init-only", action="store_true", help="INIT 만 publish")
    ap.add_argument("--page-only", action="store_true", help="PAGE 만 publish (--story 와 함께 쓰면 INIT 생략 + 시퀀스 publish)")
    ap.add_argument("--skip-init", action="store_true", help="--page-only 의 별칭 — bible 이미 있을 때")
    ap.add_argument(
        "--setting",
        default="KOREAN_TRADITIONAL",
        help="INIT setting (단일 모드 default: KOREAN_TRADITIONAL, --story 사용 시 JSON 값 우선)",
    )
    ap.add_argument(
        "--char-species",
        default="ANIMAL",
        help="INIT char_species (단일 모드 default: ANIMAL, --story 사용 시 JSON 값 우선)",
    )
    ap.add_argument(
        "--character",
        action="append",
        type=_parse_character_arg,
        default=[],
        help="단일 모드용 character ROLE=NAME (반복 가능)",
    )
    ap.add_argument(
        "--sentences",
        default="토끼와 거북이는 용궁으로 가서 용왕님을 만났어요.",
        help="단일 PAGE 모드의 sentences",
    )
    ap.add_argument(
        "--story",
        help="동화 1편 JSON 경로 — setting/char_species/characters/pages 포함",
    )
    ap.add_argument(
        "--init-wait",
        type=float,
        default=60.0,
        help="--story 모드에서 INIT publish 후 PAGE 시작 전 대기(초) — 레퍼런스 생성 시간 (default: 60)",
    )
    ap.add_argument(
        "--page-delay",
        type=float,
        default=1.0,
        help="--story 모드에서 PAGE 간 publish 간격(초) (default: 1.0)",
    )
    args = ap.parse_args()

    skip_init = args.page_only or args.skip_init

    try:
        producer_kwargs = _build_producer_kwargs()
    except RuntimeError as exc:
        print(f"[fail] {exc}")
        return 1

    topic_init = _env("KAFKA_TOPIC_INIT", "fairytale_created")
    topic_page = _env("KAFKA_TOPIC_PAGE", "fairytale_paragraph")

    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(**producer_kwargs)
    await producer.start()

    fid = args.id

    try:
        # ── 시퀀스 모드 ────────────────────────────────────────────────────
        if args.story:
            story = _load_story(args.story)
            if not skip_init:
                await _send_init(
                    producer,
                    topic_init,
                    fid,
                    story["setting"],
                    story["char_species"],
                    dict(story["characters"]),
                )
                print(f"[wait] INIT 처리(레퍼런스 생성) 대기 — {args.init_wait:.0f}s")
                await asyncio.sleep(args.init_wait)

            pages: list[str] = story["pages"]
            for i, sentences in enumerate(pages, start=1):
                await _send_page(producer, topic_page, fid, i, sentences)
                if i < len(pages):
                    await asyncio.sleep(args.page_delay)

            print()
            print(f"publish 완료. 워커 콘솔에서 페이지 {len(pages)}장 생성 모니터링.")
            print(f"결과 S3 경로: fairytales/{fid}/pages/page_01.png ~ page_{len(pages):02d}.png")
            return 0

        # ── 단일 모드 ──────────────────────────────────────────────────────
        characters = dict(args.character) if args.character else {
            "HERO": "토끼",
            "VILLAIN": "거북이",
            "DONOR": "용왕님",
        }

        if not args.page_only:
            await _send_init(producer, topic_init, fid, args.setting, args.char_species, characters)

        if not args.init_only:
            await _send_page(producer, topic_page, fid, args.page, args.sentences)

        print("\n워커 콘솔에서 생성 진행 상황 확인.")
        return 0
    finally:
        await producer.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
