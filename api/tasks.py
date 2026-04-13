"""백그라운드 Job 실행 + 이미지 저장 관리."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image


# ─── Job 데이터 ──────────────────────────────────────────────────────────────


@dataclass
class PageResult:
    page_index: int
    source_text: str
    image_url: str
    elapsed: float


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued → analyzing → generating → completed / failed
    story_text: str = ""
    protagonist_type: str = "other"
    theme: str = "FOREST_NATURE"
    seed: int | None = 42
    lora_key: str = "raw_200"
    total_pages: int = 0
    completed_pages: list[int] = field(default_factory=list)
    page_results: list[PageResult] = field(default_factory=list)
    error: str | None = None
    total_elapsed: float = 0.0


class JobStore:
    """인메모리 Job 저장소. 최근 N개만 유지."""

    def __init__(self, max_jobs: int = 20):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._max = max_jobs
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Job:
        job = Job(job_id=str(uuid.uuid4()), **kwargs)
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


# ─── 이미지 저장소 (추상화) ──────────────────────────────────────────────────


class ImageStorage(ABC):
    """이미지 저장소 인터페이스. LocalStorage / S3Storage 교체 가능."""

    @abstractmethod
    def save(self, job_id: str, page_index: int, image: Image.Image) -> str:
        """이미지를 저장하고 접근 URL을 반환한다."""

    @abstractmethod
    def get_url(self, job_id: str, page_index: int) -> str:
        """저장된 이미지의 URL을 반환한다."""


class LocalStorage(ImageStorage):
    """로컬 디스크 저장소. FastAPI StaticFiles로 서빙."""

    def __init__(self, base_dir: str = "outputs/jobs", url_prefix: str = "/images"):
        self.base_dir = Path(base_dir)
        self.url_prefix = url_prefix

    def save(self, job_id: str, page_index: int, image: Image.Image) -> str:
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        filename = f"page_{page_index:02d}.png"
        filepath = job_dir / filename
        image.save(str(filepath), format="PNG")
        return f"{self.url_prefix}/{job_id}/{filename}"

    def get_url(self, job_id: str, page_index: int) -> str:
        filename = f"page_{page_index:02d}.png"
        return f"{self.url_prefix}/{job_id}/{filename}"


# ─── Job 실행기 ──────────────────────────────────────────────────────────────


class JobExecutor:
    """GPU 독점 단일 워커. 한 번에 하나의 job만 실행한다."""

    def __init__(
        self,
        store: JobStore,
        storage: ImageStorage,
        max_queue: int = 3,
    ):
        self._store = store
        self._storage = storage
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._callbacks: dict[str, Callable] = {}
        self._callbacks_lock = threading.Lock()

    # ── 공개 API ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def submit(self, job_id: str, on_event: Callable | None = None) -> bool:
        """job을 큐에 추가한다. 큐가 가득 차면 False를 반환한다."""
        try:
            self._queue.put_nowait(job_id)
            if on_event:
                with self._callbacks_lock:
                    self._callbacks[job_id] = on_event
            return True
        except queue.Full:
            return False

    # ── 내부: 워커 스레드 ────────────────────────────────────────────

    def _emit(self, job_id: str, event: str, data: dict) -> None:
        """SSE 이벤트를 콜백으로 전달한다."""
        with self._callbacks_lock:
            cb = self._callbacks.get(job_id)
        if cb:
            try:
                cb(event, data)
            except Exception:
                pass

    def _worker(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self._store.get(job_id)
            if not job:
                continue
            try:
                self._run_job(job)
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                self._emit(job_id, "error", {"message": str(e)})
            finally:
                with self._callbacks_lock:
                    self._callbacks.pop(job_id, None)

    # ── 내부: 파이프라인 실행 ────────────────────────────────────────

    def _run_job(self, job: Job) -> None:
        from api.pipeline.model_manager import ModelManager
        from api.pipeline.story_pipeline import build_story_plan

        start_total = time.time()

        # ── 1. 스토리 분석 (GPT-4o) ──────────────────────────────────────
        job.status = "analyzing"

        story_plan = build_story_plan(job.story_text)

        # 사용자가 선택한 theme으로 world profile 오버라이드
        if job.theme != story_plan.world.theme:
            from api.pipeline.story_pipeline import build_world_profile, build_scene_plans
            story_plan.world = build_world_profile(job.theme)
            story_plan.scenes = build_scene_plans(
                story_plan.story_input, story_plan.world, story_plan.character_bible,
            )

        job.total_pages = len(story_plan.scenes)

        self._emit(job.job_id, "analyzing", {"total_pages": job.total_pages})

        # ── 2. 페이지별 이미지 생성 ─────────────────────────────────────
        job.status = "generating"
        generator = ModelManager.get().generator

        ref_image = None
        failed_pages: list[int] = []

        for i, scene in enumerate(story_plan.scenes):
            try:
                current_ref = None
                ip_scale_for_this_page = None

                if i > 0 and ref_image is not None:
                    scene_char_count = len(scene.character_ids) if scene.character_ids else 0
                    if scene_char_count >= 2:
                        ip_scale_for_this_page = 0.05
                    current_ref = ref_image

                page_seed = (job.seed + scene.page_index) if job.seed is not None else None

                images, elapsed = generator.generate(
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    prompt_2=scene.prompt_2,
                    seed=page_seed,
                    width=1024,
                    height=1024,
                    ip_adapter_image=current_ref,
                    ip_adapter_scale=ip_scale_for_this_page,
                    lora_scale=getattr(scene, "lora_scale", None),
                )

                if i == 0 and images:
                    ref_image = images[0]

                image_url = self._storage.save(
                    job.job_id, scene.page_index, images[0]
                )

                result = PageResult(
                    page_index=scene.page_index,
                    source_text=scene.source_text,
                    image_url=image_url,
                    elapsed=elapsed,
                )
                job.page_results.append(result)
                job.completed_pages.append(scene.page_index)

                self._emit(job.job_id, "page_complete", {
                    "page_index": scene.page_index,
                    "image_url": image_url,
                    "source_text": scene.source_text,
                    "elapsed": elapsed,
                })

            except Exception as exc:
                failed_pages.append(scene.page_index)
                self._emit(job.job_id, "page_error", {
                    "message": str(exc),
                    "page_index": scene.page_index,
                })

        job.total_elapsed = time.time() - start_total
        job.status = "completed"
        self._emit(job.job_id, "complete", {
            "total_pages": job.total_pages,
            "completed_pages": len(job.completed_pages),
            "failed_pages": failed_pages,
            "total_elapsed": job.total_elapsed,
        })
