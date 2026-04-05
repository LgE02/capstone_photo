"""비동기 이미지 생성 파이프라인 — preload 지원.

설계 원칙:
  - 사용자가 페이지를 넘기는 동안 다음 페이지가 이미 생성되어 있어야 함
  - 이미지 생성은 백그라운드 스레드에서 순차 실행
  - get_page() 는 캐시에 있으면 즉시 반환, 없으면 생성 완료까지 블로킹 대기
  - 에러가 나도 다른 페이지 생성은 계속 진행 (플레이스홀더로 대체)

사용 흐름:
    pipeline = AsyncGenerationPipeline(generator)
    pipeline.start(scene_plans)           # 백그라운드 생성 시작
    img, elapsed = pipeline.get_page(1)   # 1페이지 (없으면 대기)
    img, elapsed = pipeline.get_page(2)   # 2페이지 (이미 생성되어 있을 확률 높음)
    pipeline.shutdown()                   # 정리
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from PIL import Image

if TYPE_CHECKING:
    from runtime_pipeline.generator import FairytaleImageGenerator
    from runtime_pipeline.story_pipeline import ScenePlan, StoryPlan


# 생성 실패 시 보여줄 플레이스홀더 이미지 크기
_PLACEHOLDER_SIZE = (512, 512)
_PLACEHOLDER_COLOR = (230, 220, 215)  # 연한 베이지 — 동화 분위기


@dataclass
class PageResult:
    """한 페이지의 생성 결과."""

    page_index: int
    image: Optional[Image.Image] = None
    elapsed: float = 0.0
    error: Optional[str] = None
    ready: bool = False


def _make_placeholder(page_index: int) -> Image.Image:
    """생성 실패 시 사용할 단색 플레이스홀더."""
    img = Image.new("RGB", _PLACEHOLDER_SIZE, color=_PLACEHOLDER_COLOR)
    return img


class AsyncGenerationPipeline:
    """백그라운드 스레드에서 이미지를 순차 생성하고 페이지 단위 캐시를 제공한다.

    Args:
        generator: 로드 완료된 FairytaleImageGenerator 인스턴스
        seed: 고정 시드 (None이면 랜덤)
        num_images: 장면당 생성 이미지 수
        width / height: 생성 해상도 (None이면 모델 기본값)
    """

    def __init__(
        self,
        generator: "FairytaleImageGenerator",
        seed: Optional[int] = None,
        num_images: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self._generator = generator
        self._seed = seed
        self._num_images = num_images
        self._width = width
        self._height = height

        # 페이지 인덱스 → PageResult
        self._cache: dict[int, PageResult] = {}
        # 각 페이지가 준비되면 set() 되는 이벤트
        self._events: dict[int, threading.Event] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self, scene_plans: list["ScenePlan"]) -> None:
        """백그라운드 생성 스레드를 시작한다.

        scene_plans 의 순서대로 이미지를 생성한다.
        이미 실행 중이면 중지 후 재시작.
        """
        self.shutdown()
        self._stop_event.clear()

        with self._lock:
            self._cache.clear()
            self._events.clear()
            for scene in scene_plans:
                self._events[scene.page_index] = threading.Event()
                self._cache[scene.page_index] = PageResult(page_index=scene.page_index)

        self._thread = threading.Thread(
            target=self._worker,
            args=(scene_plans,),
            daemon=True,
            name="fairytale-img-gen",
        )
        self._thread.start()
        print(f"[AsyncPipeline] 생성 시작: {len(scene_plans)}페이지")

    def _worker(self, scene_plans: list["ScenePlan"]) -> None:
        """생성 워커 — 백그라운드에서 순서대로 실행된다."""
        for scene in scene_plans:
            if self._stop_event.is_set():
                break

            page_idx = scene.page_index
            start = time.time()
            try:
                images, elapsed = self._generator.generate(
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    seed=self._seed,
                    num_images=self._num_images,
                    width=self._width,
                    height=self._height,
                )
                result = PageResult(
                    page_index=page_idx,
                    image=images[0] if images else _make_placeholder(page_idx),
                    elapsed=elapsed,
                    ready=True,
                )
            except Exception as exc:
                elapsed = time.time() - start
                print(f"[AsyncPipeline] 페이지 {page_idx} 생성 실패: {exc}")
                result = PageResult(
                    page_index=page_idx,
                    image=_make_placeholder(page_idx),
                    elapsed=elapsed,
                    error=str(exc),
                    ready=True,
                )

            with self._lock:
                self._cache[page_idx] = result

            # 대기 중인 get_page() 호출을 깨운다
            self._events[page_idx].set()
            print(f"[AsyncPipeline] 페이지 {page_idx} 완료 ({result.elapsed:.1f}s)")

    def get_page(self, page_index: int, timeout: float = 180.0) -> tuple[Image.Image, float]:
        """page_index 에 해당하는 이미지를 반환한다.

        이미 생성된 경우 즉시 반환.
        아직 생성 중이면 완료될 때까지 최대 timeout 초 대기.

        Returns:
            (PIL Image, elapsed_seconds)
        Raises:
            TimeoutError: timeout 초 안에 생성이 완료되지 않은 경우
            KeyError: 존재하지 않는 page_index
        """
        event = self._events.get(page_index)
        if event is None:
            raise KeyError(f"page_index {page_index} 가 없습니다.")

        # 캐시에 이미 있는지 먼저 확인
        with self._lock:
            cached = self._cache.get(page_index)
            if cached and cached.ready:
                return cached.image, cached.elapsed

        # 없으면 대기
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"페이지 {page_index} 생성 타임아웃 ({timeout}초)")

        with self._lock:
            result = self._cache[page_index]

        return result.image, result.elapsed

    def is_ready(self, page_index: int) -> bool:
        """해당 페이지가 이미 생성 완료됐는지 확인한다 (논블로킹)."""
        with self._lock:
            result = self._cache.get(page_index)
            return result is not None and result.ready

    def ready_pages(self) -> list[int]:
        """생성 완료된 페이지 인덱스 목록을 반환한다."""
        with self._lock:
            return [idx for idx, r in self._cache.items() if r.ready]

    def shutdown(self) -> None:
        """진행 중인 생성을 중단하고 스레드를 종료한다."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None


# ─────────────────────────────────────────────────────────────────────────────
# 캐릭터 초상화 생성
# ─────────────────────────────────────────────────────────────────────────────
def generate_character_portraits(
    generator: "FairytaleImageGenerator",
    story_plan: "StoryPlan",
    output_dir: str,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Image.Image]:
    """각 캐릭터의 초상화(레퍼런스 이미지)를 생성한다.

    이 초상화는 이후 장면 생성 시 IP-Adapter 참고 이미지로 사용되어
    캐릭터 디자인의 일관성을 유지한다.

    Returns:
        {character_id: PIL.Image} 매핑
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    portraits: dict[str, Image.Image] = {}

    # StoryPlan에서 필요한 정보 추출
    style_positive = "ftbookstyle, child picture book illustration, flat cartoon illustration, clean bold line art, chibi proportions, vivid flat pastel colors"
    world_positive = story_plan.world.positive_hint
    neg_parts = [
        "multiple characters, crowd scene, many figures",
        "plain white background, empty background",
        "realistic photo, 3d render, dark mood, harsh shadows",
        "text, watermark, signature",
        story_plan.world.negative_hint,
    ]
    negative_prompt = ", ".join(neg_parts)

    for ch in story_plan.story_input.characters:
        bible_desc = story_plan.character_bible.get(ch.id, "")
        if not bible_desc:
            continue

        # 초상화 프롬프트: 스타일 + 캐릭터 묘사 + 구도
        portrait_prompt = (
            f"{style_positive}, "
            f"{bible_desc}, "
            f"solo, centered portrait, large character, facing viewer, "
            f"soft illustrated background"
        )

        portrait_seed = seed + hash(ch.id) % 1000 if seed is not None else None

        print(f"  [Portrait] {ch.id} 생성 중...")
        try:
            images, elapsed = generator.generate(
                prompt=portrait_prompt,
                negative_prompt=negative_prompt,
                seed=portrait_seed,
                width=width,
                height=height,
            )
            if images:
                portraits[ch.id] = images[0]
                # 초상화 저장
                generator.save_images(
                    images,
                    output_dir=output_dir,
                    prefix=f"portrait_{ch.id}",
                )
                print(f"  [Portrait] {ch.id} 완료 ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"  [Portrait] {ch.id} 실패: {exc}")

    return portraits


# ─────────────────────────────────────────────────────────────────────────────
# 단순 동기 실행 헬퍼 — async가 필요 없을 때 사용
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_pages_sync(
    generator: "FairytaleImageGenerator",
    scene_plans: list["ScenePlan"],
    output_dir: str,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    ip_adapter_image: Optional[Image.Image] = None,
    use_first_page_as_ref: bool = False,
    character_portraits: Optional[dict[str, Image.Image]] = None,
) -> list[dict]:
    """모든 페이지를 동기적으로 생성하고 결과 목록을 반환한다.

    Args:
        ip_adapter_image: 모든 페이지에 적용할 참고 이미지 (단일)
        use_first_page_as_ref: True면 1페이지 결과를 2페이지부터 참고 이미지로 사용
        character_portraits: 캐릭터별 레퍼런스 초상화 {id: Image}
                             이것이 있으면 장면별로 등장 캐릭터의 초상화를
                             IP-Adapter에 투입한다.

    Returns:
        [{"page": int, "path": str, "elapsed": float}, ...]
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    results = []
    ref_image = ip_adapter_image  # 외부 참고 이미지 또는 None

    for i, scene in enumerate(scene_plans):
        try:
            # IP-Adapter 참고 이미지 결정 (우선순위)
            current_ref = None

            # 1순위: 캐릭터 초상화
            if character_portraits and scene.character_ids:
                scene_char_count = len(scene.character_ids)
                if scene_char_count == 1:
                    # 단일 캐릭터: 해당 캐릭터 초상화 사용 → 일관성 극대화
                    cid = scene.character_ids[0]
                    if cid in character_portraits:
                        current_ref = character_portraits[cid]
                    else:
                        current_ref = next(iter(character_portraits.values()), None)
                else:
                    # 다중 캐릭터: IP-Adapter 사용하지 않음
                    # 한 캐릭터 초상화만 넣으면 다른 캐릭터도 그 모습으로 변함
                    # (예: 토끼 초상화 → 나무꾼도 토끼처럼)
                    current_ref = None

            # 2순위: 외부 참고 이미지
            elif ip_adapter_image is not None:
                current_ref = ip_adapter_image

            # 3순위: 1페이지 결과 참고
            elif use_first_page_as_ref and i > 0 and ref_image is not None:
                current_ref = ref_image

            # 페이지마다 다른 seed 사용 → 다른 초기 노이즈 → 다른 장면
            page_seed = (seed + scene.page_index) if seed is not None else None

            images, elapsed = generator.generate(
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                prompt_2=scene.prompt_2,
                seed=page_seed,
                width=width,
                height=height,
                ip_adapter_image=current_ref,
            )

            # use_first_page_as_ref: 첫 페이지 결과를 참고 이미지로 저장
            if use_first_page_as_ref and i == 0 and images:
                ref_image = images[0]
                print(f"  [IP-Adapter] 1페이지를 이후 참고 이미지로 설정")

            saved = generator.save_images(
                images,
                output_dir=output_dir,
                prefix=f"page_{scene.page_index:02d}",
            )
            results.append({
                "page": scene.page_index,
                "path": saved[0] if saved else "",
                "elapsed": elapsed,
                "source_text": scene.source_text,
            })
            print(f"  Page {scene.page_index} 완료 ({elapsed:.1f}s): {scene.source_text[:30]}")
        except Exception as exc:
            print(f"  Page {scene.page_index} 실패: {exc}")
            results.append({
                "page": scene.page_index,
                "path": "",
                "elapsed": 0.0,
                "source_text": scene.source_text,
                "error": str(exc),
            })

    return results
