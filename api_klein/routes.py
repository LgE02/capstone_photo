"""FastAPI 엔드포인트 — Klein API 전용."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api_klein.schemas import GenerateRequest
from api_klein.tasks import KleinJobExecutor, JobStore

router = APIRouter()

_store: JobStore | None = None
_executor: KleinJobExecutor | None = None


def init_routes(store: JobStore, executor: KleinJobExecutor) -> None:
    global _store, _executor
    _store = store
    _executor = executor


@router.post("/generate")
async def generate(request: GenerateRequest):
    """동화 삽화 생성 (SSE 스트리밍).
    캐릭터 레퍼런스 생성 후 페이지별 삽화를 실시간으로 스트리밍.
    """
    if _store is None or _executor is None:
        raise HTTPException(500, "서버 초기화 안 됨")

    job = _store.create(
        story_text=request.story_text,
        protagonist_type=request.protagonist_type.value,
        theme=request.theme.value,
        seed=request.seed,
    )

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event_type: str, data: dict) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, (event_type, data))

    submitted = _executor.submit(job.job_id, on_event=on_event)
    if not submitted:
        raise HTTPException(503, "서버가 처리 중입니다. 잠시 후 다시 시도해주세요.")

    async def event_generator():
        try:
            while True:
                event_type, data = await event_queue.get()
                yield {
                    "event": event_type,
                    "data": json.dumps(data, ensure_ascii=False),
                }
                if event_type in ("complete", "error"):
                    break
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Job 상태 조회."""
    if _store is None:
        raise HTTPException(500, "서버 초기화 안 됨")

    job = _store.get(job_id)
    if not job:
        raise HTTPException(404, "Job을 찾을 수 없습니다")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_pages": job.total_pages,
        "completed_pages": job.completed_pages,
        "character_reference_url": job.character_reference_url,
        "pages": [
            {
                "page_index": p.page_index,
                "source_text": p.source_text,
                "image_url": p.image_url,
                "elapsed": p.elapsed,
            }
            for p in job.page_results
        ],
        "error": job.error,
        "total_elapsed": job.total_elapsed,
    }
