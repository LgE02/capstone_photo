"""삽화 생성 워커 핵심 로직.

카프카 메시지 1건을 받아 처리. 메시지 타입은 **토픽 이름**으로 판별:

  토픽 KAFKA_TOPIC_INIT (예: fairytale_created)
    payload: {fairytaleId, setting, char_species, characters}
    → bible.json + 역할별 레퍼런스 이미지를 S3 에 생성. 결과 publish 없음.

  토픽 KAFKA_TOPIC_PAGE (예: fairytale_paragraph)
    payload: {fairytaleId, pageNo, sentences}
      ※ sentences 는 '\\n' 으로 구분된 단일 문자열로 옴 (Spring 측 합의).
        list 로 들어오면 그대로 사용.
    → bible 로드 + 페이지 삽화 생성 → S3 → Kafka publish (KAFKA_TOPIC_RESULT).

PAGE 가 INIT 보다 먼저 도착하면 BibleNotReadyError 를 raise — consumer 가
commit 보류하고 재시도. 두 토픽이 분리돼 있어 INIT/PAGE 간 순서 보장이 없으므로
Spring 측에서 "동화 생성 직후 INIT publish → 잠시 후 PAGE publish" 흐름을
지켜주는 게 권장. 그래도 race 가 생기면 워커가 자동 재시도함.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from api_klein.pipeline.llm_prompt_extractor import (
    extract_character_bible,
    extract_page_scene,
)
from api_klein.pipeline.model_manager import KleinModelManager
from api_klein.pipeline.story_pipeline import build_world_profile
from api_klein.publisher import KafkaResultPublisher
from api_klein.storage import S3Storage


STYLE_PREFIX = (
    "children's book illustration, soft cell shading with smooth color gradients, "
    "warm teal and golden tones, glowing light particles scattered in the background, "
    "painterly digital art style, no hard outlines, lush soft background. "
)

REFERENCE_SEED_OFFSET = {
    "HERO": 0,
    "VILLAIN": 100,
    "DISPATCHER": 200,
    "HELPER": 300,
    "FALSE_HERO": 400,
    "DONOR": 500,
}


class BibleNotReadyError(RuntimeError):
    """PAGE 메시지가 도착했는데 해당 fairytale 의 bible.json 이 아직 없음.

    INIT 메시지가 아직 처리되지 않았을 때 발생. consumer 는 이 예외를 보면
    commit 을 보류하고 짧은 백오프 후 같은 메시지를 재시도해야 함.
    """

    def __init__(self, fairytale_id: int):
        super().__init__(f"bible.json not found for fairytale {fairytale_id}")
        self.fairytale_id = fairytale_id


def _parse_sentences(raw: Any) -> list[str]:
    """Spring 측이 sentences 를 '\\n' 구분 단일 문자열로 보냄.

    호환성 차원에서 list/tuple 도 받아 그대로 처리.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split("\n") if s.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    raise TypeError(f"sentences 는 string 또는 list 여야 함, got {type(raw).__name__}")


def _pick(message: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """여러 후보 키 중에서 처음 발견되는 값을 반환.

    Spring 측이 camelCase/snake_case 둘 다 섞어 보내는 상황을 흡수하기 위함.
    예: _pick(msg, 'fairytaleId', 'fairytale_id')
    """
    for k in keys:
        if k in message and message[k] is not None:
            return message[k]
    return default


@dataclass
class FairytaleProcessor:
    """워커 단위 핵심 처리기 — 카프카 1메시지 = handle_message 1회 호출."""

    storage: S3Storage
    publisher: KafkaResultPublisher
    base_seed: int = 42

    def __post_init__(self) -> None:
        # fairytale_id → role → reference image (메모리 LRU 같은 hot cache, 단순 dict)
        self._ref_cache: dict[int, dict[str, Image.Image]] = {}
        # fairytale_id → bible payload
        self._bible_cache: dict[int, dict[str, Any]] = {}
        # fairytale_id → 과거 페이지 장면 히스토리 (공간 일관성 유지용)
        # 각 항목: {"page_no": int, "sentences": list[str], "narrative_hint": str, "focus_roles": list[str]}
        # 페이지 N 처리 시 1..N-1 의 항목을 GPT-5.5 에 함께 전달.
        # 워커 메모리 한정 — 재시작 시 손실 (발표 데모 범위).
        self._scene_history: dict[int, list[dict[str, Any]]] = {}

    # ── 진입점 ─────────────────────────────────────────────────────────────

    def handle_message(self, topic: str, message: dict[str, Any]) -> str | None:
        """카프카 메시지 1건 처리.

        Args:
            topic: 메시지가 들어온 카프카 토픽 이름. INIT/PAGE 분기에 사용.
            message: JSON 디코드된 payload.

        Returns:
            INIT: None
            PAGE: 생성·재사용된 페이지 이미지의 S3 URL.

        Raises:
            BibleNotReadyError: PAGE 인데 bible.json 없음 (consumer 가 재시도).
        """
        # Spring 이 camelCase / snake_case 둘 다 섞어 보내고 있어서 전부 흡수.
        fairytale_id_raw = _pick(message, "fairytaleId", "fairytale_id")
        if fairytale_id_raw is None:
            raise KeyError(f"fairytaleId / fairytale_id 둘 다 없음: keys={list(message.keys())}")
        fairytale_id = int(fairytale_id_raw)

        # 토픽 이름으로 INIT/PAGE 판별. 환경변수로 토픽명을 받아 비교.
        topic_init = os.environ.get("KAFKA_TOPIC_INIT", "fairytale_created").strip()
        topic_page = os.environ.get("KAFKA_TOPIC_PAGE", "fairytale_paragraph").strip()

        if topic == topic_init:
            characters = _pick(message, "characters")
            if not characters:
                raise ValueError(
                    f"INIT 메시지에 'characters' 가 없음: fairytale {fairytale_id}"
                )
            self._ensure_initialized(
                fairytale_id=fairytale_id,
                setting=str(_pick(message, "setting", default="FOREST_NATURE")),
                char_species=str(
                    _pick(message, "char_species", "charSpecies", default="ETC")
                ).upper(),
                characters=dict(characters),
            )
            return None

        if topic != topic_page:
            # 알 수 없는 토픽 — 설정 오류일 가능성. 실패시켜 commit 안 되게 함
            raise ValueError(f"알 수 없는 토픽: {topic} (INIT={topic_init}, PAGE={topic_page})")

        # ── PAGE 처리 ─────────────────────────────────────────────────────
        page_no_raw = _pick(message, "pageNo", "page")
        if page_no_raw is None:
            raise KeyError(f"pageNo / page 둘 다 없음: keys={list(message.keys())}")
        page_no = int(page_no_raw)
        sentences = _parse_sentences(_pick(message, "sentences", "text"))

        # 멱등성: 이미 생성된 페이지면 재사용 (재시도/중복 메시지 안전장치).
        # publish 는 다시 보냄 — Spring 이 첫 메시지 때 결과를 못 받았을 수 있음.
        if self.storage.page_exists(fairytale_id, page_no):
            s3_url = self.storage.page_url(fairytale_id, page_no)
            self.publisher.publish_page_complete(fairytale_id, page_no, s3_url)
            print(f"[processor] page {page_no} 이미 존재, skip → {s3_url}")
            return s3_url

        bible_payload = self._load_bible(fairytale_id)
        if bible_payload is None:
            raise BibleNotReadyError(fairytale_id)

        return self._process_page(
            fairytale_id=fairytale_id,
            page_no=page_no,
            sentences=sentences,
            bible_payload=bible_payload,
        )

    # ── init ───────────────────────────────────────────────────────────────

    def _ensure_initialized(
        self,
        fairytale_id: int,
        setting: str,
        char_species: str,
        characters: dict[str, str],
    ) -> None:
        """입력에 명시된 역할들의 레퍼런스 + bible.json 이 S3에 있는지 확인하고 없으면 생성.

        idempotent — 이미 있으면 즉시 리턴.
        역할 수는 입력 dict에 따라 가변 (Propp 형태론 5~6개 가능).
        """
        roles = list(characters.keys())
        existing_bible = self.storage.download_bible(fairytale_id)
        all_refs_present = self.storage.references_exist(fairytale_id, roles)

        if existing_bible and all_refs_present:
            self._bible_cache[fairytale_id] = existing_bible
            print(f"[processor] init skip — fairytale {fairytale_id} 이미 초기화됨")
            return

        print(f"[processor] init 시작 — fairytale {fairytale_id} (역할 {len(roles)}개: {roles})")
        init_start = time.time()

        # bible 없으면 GPT-5.5로 생성
        if existing_bible:
            bible_payload = existing_bible
            visual_descriptions = bible_payload["visual_descriptions"]
        else:
            bible_t0 = time.time()
            visual_descriptions = extract_character_bible(
                setting=setting,
                char_species=char_species,
                characters=characters,
            )
            bible_payload = {
                "fairytaleId": fairytale_id,
                "setting": setting,
                "char_species": char_species,
                "characters": characters,
                "visual_descriptions": visual_descriptions,
            }
            self.storage.upload_bible(fairytale_id, bible_payload)
            print(f"[processor]  bible.json 생성·업로드 완료 ({time.time() - bible_t0:.1f}s)")

        self._bible_cache[fairytale_id] = bible_payload

        # 누락된 역할만 레퍼런스 생성
        generator = KleinModelManager.get().generator
        ref_cache = self._ref_cache.setdefault(fairytale_id, {})

        for role in roles:
            if self.storage.object_exists(self.storage.reference_key(fairytale_id, role)):
                continue

            char_data = visual_descriptions.get(role, {}) or {}
            visual_desc = (char_data.get("visual_description") or "").strip()
            if not visual_desc:
                print(f"[processor]  {role} visual_description 없음, 레퍼런스 skip")
                continue

            seed = self.base_seed + REFERENCE_SEED_OFFSET.get(role, 0) + fairytale_id
            print(f"[processor]  {role} 레퍼런스 생성 중 (seed={seed})")
            ref_t0 = time.time()
            ref_image = generator.generate_character_reference(visual_desc, seed=seed)
            self.storage.upload_reference(fairytale_id, role, ref_image)
            ref_cache[role] = ref_image
            print(f"[processor]  {role} 레퍼런스 완료 ({time.time() - ref_t0:.1f}s)")

        print(f"[processor] init 완료 — fairytale {fairytale_id} (총 {time.time() - init_start:.1f}s)")

    # ── 페이지 처리 ─────────────────────────────────────────────────────────

    def _process_page(
        self,
        fairytale_id: int,
        page_no: int,
        sentences: list[str],
        bible_payload: dict[str, Any],
    ) -> str:
        visual_descriptions = bible_payload["visual_descriptions"]
        setting = bible_payload.get("setting", "FOREST_NATURE")

        # 과거 페이지 히스토리 — 공간/상황 일관성 유지용
        previous_scenes = self._scene_history.get(fairytale_id, [])

        # 장면 분석 — GPT 호출 시간 측정 (사용 모델/비용 추적 + 단축 효과 검증용)
        analysis_t0 = time.time()
        scene = extract_page_scene(
            sentences=sentences,
            character_bible=visual_descriptions,
            setting=setting,
            previous_scenes=previous_scenes,
        )
        analysis_elapsed = time.time() - analysis_t0
        focus_roles: list[str] = scene["focus_roles"]
        narrative_hint: str = scene["narrative_hint"]
        print(f"[processor] page {page_no} 분석 ({analysis_elapsed:.1f}s) — 등장:{focus_roles}")
        print(f"[processor]   narrative_hint = {narrative_hint!r}")

        # 이 페이지를 히스토리에 추가 (생성 성공 여부와 무관하게 분석 결과는 기록)
        self._scene_history.setdefault(fairytale_id, []).append({
            "page_no": page_no,
            "sentences": sentences,
            "narrative_hint": narrative_hint,
            "focus_roles": focus_roles,
        })

        # 등장 역할의 레퍼런스 로드
        refs: list[Image.Image] = []
        for role in focus_roles:
            ref = self._load_reference(fairytale_id, role)
            if ref is not None:
                refs.append(ref)

        # FLUX multi-image 가이던스가 레퍼런스 4장 이상에서 캐릭터 분산/복제가
        # 심해지는 경향 확인됨. 페이지당 최대 3장으로 제한 (focus_roles 순서 우선).
        # 4명+ 등장 페이지는 드물어 일상 페이지엔 영향 없는 안전망 역할.
        MAX_REFERENCES_PER_PAGE = 3
        if len(refs) > MAX_REFERENCES_PER_PAGE:
            refs = refs[:MAX_REFERENCES_PER_PAGE]

        # 프롬프트 빌드
        prompt = self._build_page_prompt(
            narrative_hint=narrative_hint,
            focus_roles=focus_roles,
            visual_descriptions=visual_descriptions,
            setting=setting,
        )

        # FLUX 호출
        generator = KleinModelManager.get().generator
        seed = self.base_seed + fairytale_id * 1000 + page_no
        images, elapsed = generator.generate(
            prompt=prompt,
            seed=seed,
            reference_images=refs if refs else None,
        )

        # S3 업로드 → Redis publish
        s3_url = self.storage.upload_page(fairytale_id, page_no, images[0])
        self.publisher.publish_page_complete(fairytale_id, page_no, s3_url)
        print(f"[processor] page {page_no} 완료 ({elapsed:.1f}s) → {s3_url}")
        return s3_url

    # ── 캐시 helpers ───────────────────────────────────────────────────────

    def _load_bible(self, fairytale_id: int) -> dict[str, Any] | None:
        if fairytale_id in self._bible_cache:
            return self._bible_cache[fairytale_id]
        bible = self.storage.download_bible(fairytale_id)
        if bible:
            self._bible_cache[fairytale_id] = bible
        return bible

    def _load_reference(self, fairytale_id: int, role: str) -> Image.Image | None:
        cache = self._ref_cache.setdefault(fairytale_id, {})
        if role in cache:
            return cache[role]
        img = self.storage.download_reference(fairytale_id, role)
        if img is not None:
            cache[role] = img
        return img

    # ── 프롬프트 빌드 ───────────────────────────────────────────────────────

    @staticmethod
    def _build_page_prompt(
        narrative_hint: str,
        focus_roles: list[str],
        visual_descriptions: dict[str, dict[str, str]],
        setting: str,
    ) -> str:
        """Klein/Qwen3용 자연어 프롬프트 — 레퍼런스 이미지가 외형을 책임지므로
        텍스트는 장면 + 위치/배경 + 간단한 캐릭터 식별 위주.
        """
        world = build_world_profile(setting)

        char_parts: list[str] = []
        for i, role in enumerate(focus_roles):
            entry = visual_descriptions.get(role, {}) or {}
            desc = (entry.get("visual_description") or "").strip()
            ctype = (entry.get("type") or "ETC").upper()
            if not desc:
                desc = f"a {role.lower()} character"

            if ctype == "ANIMAL":
                # visual_description 키워드로 새 anatomy 안전망 — bible 가드가 못 잡은
                # 케이스(과거에 만들어진 깨진 bible)도 페이지 단계에서 한 번 더 잡음.
                desc_lower = desc.lower()
                if any(k in desc_lower for k in ("feather", "beak", "wing", "talon")):
                    desc = (
                        f"{desc} — this character has full bird anatomy: "
                        f"eyes on the sides of the head (not centered like a human face), "
                        f"no smile-line and no lip corners beside the beak, "
                        f"feathers not fur"
                    )
                else:
                    desc = f"{desc} — this character has a full animal body"
            elif ctype == "HUMAN":
                desc = f"{desc} — this character is a human person"

            if len(focus_roles) == 2:
                position = "on the left side" if i == 0 else "on the right side"
                desc = f"{desc}, positioned {position}"

            char_parts.append(desc)

        if len(char_parts) == 1:
            char_sentence = f"The main character is {char_parts[0]}."
        elif len(char_parts) == 2:
            char_sentence = (
                f"There are two separate characters in this scene. "
                f"First: {char_parts[0]}. Second: {char_parts[1]}."
            )
        elif char_parts:
            numbered = " ".join(f"Character {i+1}: {p}." for i, p in enumerate(char_parts))
            char_sentence = f"There are {len(char_parts)} characters. {numbered}"
        else:
            char_sentence = "A storybook character."

        action = f"Scene: {narrative_hint}." if narrative_hint else ""
        background = f"Background: {world.positive_hint}."

        # 그림책 톤 — medium shot 기본. cinematic angle 안 씀.
        visibility = (
            "Medium shot — characters shown full-body or upper-body, "
            "clearly readable, surroundings visible."
        )

        # 말풍선/텍스트 금지 — 모델이 임의로 그리는 garbled letter 방지.
        no_text = (
            "Absolutely no speech bubbles, dialogue balloons, thought bubbles, "
            "captions, signs, posters, banners, or any letters, words, or numbers "
            "anywhere in the image."
        )

        parts = [STYLE_PREFIX, char_sentence]
        if action:
            parts.append(action)
        parts.append(background)
        parts.append(visibility)
        parts.append(no_text)
        return " ".join(parts)
