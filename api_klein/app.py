"""FLUX.2-klein-4B 기반 동화 삽화 생성 API 서버.

실행:
    uvicorn api_klein.app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_klein.routes import init_routes, router
from api_klein.tasks import KleinJobExecutor, JobStore, LocalStorage

OUTPUTS_DIR = Path("outputs/klein_jobs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api_klein.pipeline.model_manager import KleinModelManager

    print("[Klein API] FLUX.2-klein-4B 로드 중...")
    mgr = KleinModelManager.get()
    mgr.load()
    print("[Klein API] 모델 로드 완료")

    storage = LocalStorage(base_dir=str(OUTPUTS_DIR), url_prefix="/images")
    store = JobStore(max_jobs=20)
    executor = KleinJobExecutor(store=store, storage=storage, max_queue=3)
    executor.start()

    init_routes(store, executor)
    print("[Klein API] 서버 준비 완료")

    yield

    executor.shutdown()
    mgr.unload()
    print("[Klein API] 서버 종료")


app = FastAPI(
    title="동화 삽화 생성 API (Klein)",
    description="FLUX.2-klein-4B 기반 캐릭터 일관성 동화 삽화 생성",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(OUTPUTS_DIR)), name="images")
app.include_router(router)
