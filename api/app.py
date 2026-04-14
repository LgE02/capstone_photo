"""FastAPI 앱 — 동화 삽화 생성 API.

실행:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import init_routes, router
from api.tasks import JobExecutor, JobStore, LocalStorage

# 이미지 저장 경로
OUTPUTS_DIR = Path("outputs/jobs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 언로드."""
    from api.pipeline.model_manager import ModelManager

    # ── 시작: FLUX 모델 로드 ─────────────────────────────────────────
    print("[API] FLUX Schnell (NF4) 로드 중...")
    mgr = ModelManager.get()
    mgr.load()
    print("[API] 모델 로드 완료")

    # ── Job 시스템 초기화 ────────────────────────────────────────────
    storage = LocalStorage(base_dir=str(OUTPUTS_DIR), url_prefix="/images")
    store = JobStore(max_jobs=20)
    executor = JobExecutor(store=store, storage=storage, max_queue=3)
    executor.start()

    init_routes(store, executor)
    print("[API] 서버 준비 완료")

    yield

    # ── 종료: 정리 ───────────────────────────────────────────────────
    executor.shutdown()
    mgr.unload()
    print("[API] 서버 종료")


app = FastAPI(
    title="동화 삽화 생성 API",
    description="동화 텍스트를 받아 FLUX.1 Schnell 기반 삽화를 생성합니다.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 생성된 이미지 정적 서빙
app.mount("/images", StaticFiles(directory=str(OUTPUTS_DIR)), name="images")

app.include_router(router)
