"""삽화 생성 워커 진입점.

실행:
    python worker.py

순서:
  1. .env 로드 (OPENAI/HF/AWS/Kafka 환경변수)
  2. FLUX.2-klein-4B 모델 로드 (싱글톤)
  3. S3 / Kafka producer (결과 publish) 초기화
  4. 카프카 컨슈머 시작 (fairytale_created + fairytale_paragraph)
     → 메시지마다 FairytaleProcessor.handle_message 호출
     → PAGE 처리 후 fairytale_image 토픽으로 결과 publish

종료(SIGINT 등) 시 컨슈머·producer·모델 정리.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()


def _check_env() -> None:
    """필수 환경변수 누락 시 친절한 에러."""
    missing: list[str] = []
    required = [
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET",
        "KAFKA_BOOTSTRAP_SERVERS",
    ]
    for key in required:
        if not (os.environ.get(key) or "").strip():
            missing.append(key)

    if missing:
        print("[worker] 누락된 환경변수:", ", ".join(missing))
        print("        .env 파일에 채워 넣으세요. (.env.example 참고)")
        sys.exit(1)


async def _main() -> None:
    _check_env()

    # 1) FLUX 모델 로드 — import는 안에서 (환경변수 로드 후)
    from api_klein.consumer import run_consumer_loop
    from api_klein.pipeline.model_manager import KleinModelManager
    from api_klein.processor import FairytaleProcessor
    from api_klein.publisher import KafkaResultPublisher
    from api_klein.storage import S3Storage

    print("[worker] FLUX.2-klein-4B 로드 시작 (30~60초 소요)")
    mgr = KleinModelManager.get()
    mgr.load()
    print("[worker] 모델 로드 완료")

    # 2) 어댑터 초기화
    storage = S3Storage()
    publisher = KafkaResultPublisher()
    publisher.start()  # 백그라운드 producer 루프 기동

    if not publisher.ping():
        print("[worker] ⚠ Kafka producer 메타데이터 조회 실패 — 결과 publish 가 안 될 수 있음")

    processor = FairytaleProcessor(storage=storage, publisher=publisher)

    # 3) Graceful shutdown
    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        print("\n[worker] 종료 신호 수신 — 정리 중...")
        stop_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _on_signal)
    # Windows는 KeyboardInterrupt를 별도 처리

    consumer_task = asyncio.create_task(run_consumer_loop(processor.handle_message, publisher))
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        done, pending = await asyncio.wait(
            {consumer_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Windows: signal handler 가 안 붙어서 KeyboardInterrupt 가 여기로 옴.
        print("\n[worker] KeyboardInterrupt — 컨슈머 정리 중...")
        stop_event.set()
        consumer_task.cancel()
        try:
            await asyncio.wait_for(consumer_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        publisher.stop()
        mgr.unload()
        print("[worker] 종료 완료")
        return

    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    for task in done:
        # 컨슈머가 예외로 끝났으면 위로
        if task is consumer_task and not task.cancelled() and task.exception():
            publisher.stop()
            mgr.unload()
            raise task.exception()  # type: ignore[misc]

    publisher.stop()
    mgr.unload()
    print("[worker] 종료 완료")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        # asyncio.run 바깥에서 잡힌 경우 — 위 _main 에서 이미 정리했을 가능성 높음
        print("\n[worker] 종료")
