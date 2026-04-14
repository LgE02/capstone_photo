"""Klein 백그라운드 Job 실행기.
FLUX.2-klein-4B + 캐릭터 레퍼런스 기반 동화 삽화 생성.
"""

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
    total_pages: int = 0
    completed_pages: list[int] = field(default_factory=list)
    page_results: list[PageResult] = field(default_factory=list)
    character_reference_url: str | None = None
    error: str | None = None
    total_elapsed: float = 0.0


class JobStore:
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


# ─── 이미지 저장소 ───────────────────────────────────────────────────────────


class ImageStorage(ABC):
    @abstractmethod
    def save(self, job_id: str, page_index: int, image: Image.Image) -> str:
        pass

    @abstractmethod
    def get_url(self, job_id: str, page_index: int) -> str:
        pass


class LocalStorage(ImageStorage):
    def __init__(self, base_dir: str = "outputs/klein_jobs", url_prefix: str = "/images"):
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


class KleinJobExecutor:
    """GPU 독점 단일 워커. Klein 모델 전용."""

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

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def submit(self, job_id: str, on_event: Callable | None = None) -> bool:
        try:
            self._queue.put_nowait(job_id)
            if on_event:
                with self._callbacks_lock:
                    self._callbacks[job_id] = on_event
            return True
        except queue.Full:
            return False

    def _emit(self, job_id: str, event: str, data: dict) -> None:
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

    def _run_job(self, job: Job) -> None:
        from api_klein.pipeline.story_pipeline import build_story_plan, build_scene_plans, build_world_profile
        from api_klein.pipeline.model_manager import KleinModelManager

        start_total = time.time()

        # ── 1. 스토리 분석 ───────────────────────────────────────────────
        job.status = "analyzing"
        story_plan = build_story_plan(job.story_text)

        if job.theme != story_plan.world.theme:
            story_plan.world = build_world_profile(job.theme)
            story_plan.scenes = build_scene_plans(
                story_plan.story_input, story_plan.world, story_plan.character_bible,
            )

        job.total_pages = len(story_plan.scenes)
        self._emit(job.job_id, "analyzing", {"total_pages": job.total_pages})

        # ── 2. 캐릭터 레퍼런스 생성 ─────────────────────────────────────
        generator = KleinModelManager.get().generator

        main_char_id = story_plan.story_input.characters[0].id if story_plan.story_input.characters else None
        char_prompt = story_plan.character_bible.get(main_char_id, "") if main_char_id else ""

        if char_prompt:
            ref_seed = job.seed if job.seed is not None else 42
            ref_image = generator.generate_character_reference(char_prompt, seed=ref_seed)
            ref_url = self._storage.save(job.job_id, 0, ref_image)
            job.character_reference_url = ref_url
            self._emit(job.job_id, "character_reference", {"image_url": ref_url})

        # ── 3. 페이지별 삽화 생성 ────────────────────────────────────────
        job.status = "generating"
        failed_pages: list[int] = []

        for scene in story_plan.scenes:
            try:
                page_seed = (job.seed + scene.page_index) if job.seed is not None else None

                images, elapsed = generator.generate(
                    prompt=scene.prompt,
                    seed=page_seed,
                    width=1024,
                    height=1024,
                    use_reference=True,
                )

                image_url = self._storage.save(job.job_id, scene.page_index, images[0])

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
