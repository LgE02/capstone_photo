"""Klein 전용 스토리 파이프라인 (api/ 독립 복사본).

api/pipeline/story_pipeline.py를 기반으로 완전히 독립적으로 복사.
Klein(FLUX.2-klein-4B, Qwen3 40k토큰) 전용 프롬프트 빌더 포함.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from api_klein.pipeline.llm_prompt_extractor import extract_story_prompts as _llm_extract
from api_klein.pipeline.story_analyzer import analyze_story as _analyze_story
from api_klein.pipeline.config import (
    CHARACTER_TYPE_HINTS,
    JOB_HINTS,
    SPECIES_FALLBACK_TEMPLATE,
    SPECIES_HINTS,
    THEME_CHARACTER_COSTUME_HINTS,
    THEME_EXPANSIONS,
)

# ── Klein 스타일 프리픽스 ──────────────────────────────────────────────────────

STYLE_PREFIX = (
    "Children's picture book illustration, "
    "semi-painterly digital art with soft cel shading and warm color gradients, "
    "ALL characters drawn in chibi-style proportions: "
    "large round head, big glossy expressive eyes with highlight sparkles, rosy cheeks, "
    "small rounded nose, simplified cute features, short compact body, "
    "warm and approachable storybook aesthetic, "
    "NOT anime, NOT realistic, NOT photorealistic. "
    "Richly detailed backgrounds with atmospheric depth and warm ambient lighting. "
)


# ── 데이터 모델 ───────────────────────────────────────────────────────────────

@dataclass
class QualityGateReport:
    original_line_count: int
    cleaned_line_count: int
    recovered: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class StoryCharacter:
    id: str
    type: str = "other"
    species: str | None = None
    job: str | None = None
    traits: list[str] = field(default_factory=list)
    visual_hint: str | None = None


@dataclass
class WorldProfile:
    theme: str
    expansion_key: str
    positive_hint: str
    negative_hint: str
    character_costume_hints: dict[str, str] = field(default_factory=dict)


@dataclass
class StoryInput:
    theme: str
    characters: list[StoryCharacter]
    lines: list[str]
    style_preset: str = "fairytale_pastel"
    scene_prompts: list[str] = field(default_factory=list)
    scene_focus_characters: list[list[str]] = field(default_factory=list)


@dataclass
class SceneSpec:
    page_index: int
    source_text: str
    focus_character_ids: list[str]
    narrative_hint: str
    staging_hint: str


@dataclass
class ScenePlan:
    page_index: int
    source_text: str
    scene_spec: SceneSpec
    prompt: str
    character_ids: list[str]


@dataclass
class StoryPlan:
    story_input: StoryInput
    quality_gate: QualityGateReport
    world: WorldProfile
    character_bible: dict[str, str]
    scenes: list[ScenePlan]


# ── 메인 엔트리 ──────────────────────────────────────────────────────────────


def build_story_plan(story_data: str | dict[str, Any], max_scenes: int = 10) -> StoryPlan:
    """Klein 전용 스토리 플랜 빌더."""
    story_input = normalize_story_input(story_data, max_scenes=max_scenes)
    cleaned_lines, report = run_quality_gate(story_input.lines, max_scenes=max_scenes)
    story_input.lines = cleaned_lines
    world = build_world_profile(story_input.theme)
    character_bible = build_character_bible(story_input.characters)
    scenes = build_scene_plans(story_input, world, character_bible)
    return StoryPlan(
        story_input=story_input,
        quality_gate=report,
        world=world,
        character_bible=character_bible,
        scenes=scenes,
    )


# ── 스토리 입력 정규화 ───────────────────────────────────────────────────────


def normalize_story_input(story_data: str | list[str] | dict[str, Any], max_scenes: int = 10) -> StoryInput:
    """원본 텍스트나 JSON 입력을 내부 스키마로 정규화한다."""
    if isinstance(story_data, list) and story_data and isinstance(story_data[0], str):
        joined = "\n".join(line.strip() for line in story_data if isinstance(line, str) and line.strip())
        return normalize_story_input(joined, max_scenes=max_scenes)

    if isinstance(story_data, str):
        stripped = story_data.strip()
        if stripped.startswith("{"):
            return normalize_story_input(json.loads(stripped), max_scenes=max_scenes)

        # ── LLM 분석 (우선) ──────────────────────────────────────────────
        try:
            analyzed = _llm_extract(stripped, max_scenes=max_scenes)
            print("[KleinPipeline] LLM 프롬프트 추출 성공")
            return normalize_story_input(analyzed, max_scenes=max_scenes)
        except Exception as e:
            print(f"[KleinPipeline] LLM 추출 실패, 키워드 매핑으로 fallback: {e}")

        # ── 키워드 매핑 fallback ─────────────────────────────────────────
        analyzed = _analyze_story(stripped)
        return normalize_story_input(analyzed, max_scenes=max_scenes)

    theme = str(story_data.get("theme") or "FOREST_NATURE").strip()
    lines = [
        normalize_line(line)
        for line in story_data.get("lines", [])
        if isinstance(line, str) and normalize_line(line)
    ]
    characters = [
        normalize_character(item, index)
        for index, item in enumerate(story_data.get("characters", []), start=1)
    ]
    if not characters:
        characters = [StoryCharacter(id="main_character", type="other", traits=["cute", "storybook"])]

    scene_prompts = [
        str(p).strip() for p in story_data.get("scene_prompts", [])
        if str(p).strip()
    ]

    scene_focus_characters = [
        list(fc) if isinstance(fc, (list, tuple)) else []
        for fc in story_data.get("scene_focus_characters", [])
    ]

    return StoryInput(
        theme=theme,
        characters=characters,
        lines=lines,
        style_preset="fairytale_pastel",
        scene_prompts=scene_prompts,
        scene_focus_characters=scene_focus_characters,
    )


def normalize_character(data: Any, index: int) -> StoryCharacter:
    if not isinstance(data, dict):
        return StoryCharacter(id=f"character_{index}")

    traits = [str(trait).strip() for trait in data.get("traits", []) if str(trait).strip()]
    return StoryCharacter(
        id=str(data.get("id") or f"character_{index}").strip(),
        type=normalize_character_type(data.get("type")),
        species=optional_text(data.get("species")),
        job=optional_text(data.get("job")),
        traits=traits,
        visual_hint=optional_text(data.get("visual_hint")),
    )


# ── Quality Gate ─────────────────────────────────────────────────────────────


def run_quality_gate(lines: list[str], max_scenes: int = 10) -> tuple[list[str], QualityGateReport]:
    """중복 줄과 너무 짧은 줄을 정리한다."""
    issues: list[str] = []
    cleaned_lines: list[str] = []
    recovered = False
    previous = None
    buffer = ""

    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            continue
        if line == previous:
            issues.append(f"Removed duplicate line: {line[:40]}")
            continue
        previous = line

        if len(line) < 8:
            buffer = f"{buffer} {line}".strip()
            recovered = True
            continue

        if buffer:
            line = f"{buffer} {line}".strip()
            buffer = ""
            recovered = True

        cleaned_lines.append(line)

    if buffer:
        if cleaned_lines:
            cleaned_lines[-1] = f"{cleaned_lines[-1]} {buffer}".strip()
        else:
            cleaned_lines.append(buffer)
        recovered = True

    trimmed = cleaned_lines[:max_scenes]
    if len(cleaned_lines) > max_scenes:
        issues.append(f"Trimmed scene count from {len(cleaned_lines)} to {max_scenes}.")

    return trimmed, QualityGateReport(
        original_line_count=len(lines),
        cleaned_line_count=len(trimmed),
        recovered=recovered,
        issues=issues,
    )


# ── 세계관 / 캐릭터 Bible ────────────────────────────────────────────────────


def build_world_profile(theme: str) -> WorldProfile:
    """선택된 세계관 키를 실제 프롬프트 힌트로 변환한다."""
    normalized = theme.strip().lower()
    matched_key = "FOREST_NATURE"
    for key, value in THEME_EXPANSIONS.items():
        aliases = [key.lower()] + [alias.lower() for alias in value.get("aliases", [])]
        if normalized in aliases:
            matched_key = key
            break

    config = THEME_EXPANSIONS[matched_key]
    costume_hints = THEME_CHARACTER_COSTUME_HINTS.get(matched_key, {})
    return WorldProfile(
        theme=theme,
        expansion_key=matched_key,
        positive_hint=config["positive"],
        negative_hint=config["negative"],
        character_costume_hints=costume_hints,
    )


def build_character_bible(characters: list[StoryCharacter]) -> dict[str, str]:
    """캐릭터 메타데이터를 시각 묘사로 압축한다.

    LLM이 생성한 visual_hint (era-accurate, detailed) 우선 사용.
    LLM 미사용 시에만 config의 fallback 매핑 사용.
    """
    # 종(species)별 해부학적 특징 강조 (이미지 모델이 헷갈리지 않도록)
    SPECIES_ANATOMY: dict[str, str] = {
        "rabbit": "rabbit with LONG upright ears (NOT dog ears), short round cottontail, rabbit snout with split lip, hind legs longer than front legs",
        "tiger": "tiger with striped fur, round ears, feline face with whiskers, thick tail",
        "fox": "fox with pointed ears, bushy thick tail, narrow pointed snout",
        "bear": "bear with round ears, broad flat face, stocky body",
        "cat": "cat with pointy triangular ears, narrow face, long thin tail",
        "dog": "dog with floppy or upright ears, broad snout, wagging tail",
        "turtle": "turtle with domed shell on back, stubby legs, short neck",
        "frog": "frog with wide flat head, bulging eyes, webbed feet, no tail",
        "dragon": "dragon with horns, scales, wings, clawed feet, long tail",
        "deer": "deer with antlers (if male), slender legs, large gentle eyes",
    }

    bible: dict[str, str] = {}
    for character in characters:
        # LLM visual_hint가 충분하면 그대로 사용 + species 해부학 앞에 추가
        if character.visual_hint and len(character.visual_hint.split()) >= 5:
            vh = character.visual_hint
            species_key = (character.species or "").lower()
            # species가 visual_hint에 없으면 앞에 추가
            if character.species and character.species.lower() not in vh.lower():
                vh = f"{character.species}, {vh}"
            # 해부학적 특징 강조 추가 (이미지 모델 혼동 방지)
            if species_key in SPECIES_ANATOMY:
                vh = f"{vh}. ANATOMY: {SPECIES_ANATOMY[species_key]}"
            bible[character.id] = vh
            continue

        # fallback: SPECIES_HINTS / JOB_HINTS 매핑
        parts: list[str] = []
        if character.species:
            species_key = character.species.lower()
            if species_key in SPECIES_HINTS:
                parts.append(SPECIES_HINTS[species_key])
            else:
                parts.append(SPECIES_FALLBACK_TEMPLATE.format(species=character.species))
        else:
            parts.append(CHARACTER_TYPE_HINTS.get(character.type, "storybook character"))

        if character.job:
            parts.append(JOB_HINTS.get(character.job.lower(), character.job))
        if character.visual_hint:
            parts.append(character.visual_hint)

        cleaned_parts = [p.rstrip(", ") for p in parts if p and p.strip()]
        bible[character.id] = ", ".join(cleaned_parts)
    return bible


# ── 장면 계획 빌드 ───────────────────────────────────────────────────────────


def build_scene_plans(
    story_input: StoryInput,
    world: WorldProfile,
    character_bible: dict[str, str],
) -> list[ScenePlan]:
    """스토리 한 줄마다 하나의 Klein 자연어 프롬프트를 만든다."""
    char_by_id = {ch.id: ch for ch in story_input.characters}
    scenes: list[ScenePlan] = []
    llm_prompts = story_input.scene_prompts
    llm_focus = story_input.scene_focus_characters

    for index, line in enumerate(story_input.lines, start=1):
        llm_hint = llm_prompts[index - 1] if index - 1 < len(llm_prompts) else ""
        llm_chars = llm_focus[index - 1] if index - 1 < len(llm_focus) else []

        scene_spec = build_scene_spec(
            page_index=index,
            line=line,
            characters=story_input.characters,
            character_bible=character_bible,
            llm_scene_prompt=llm_hint,
            llm_focus_characters=llm_chars,
        )

        scene_chars = [char_by_id[cid] for cid in scene_spec.focus_character_ids if cid in char_by_id]
        if not scene_chars:
            scene_chars = [story_input.characters[0]]

        prompt = build_prompt(
            scene_spec=scene_spec,
            world=world,
            character_bible=character_bible,
            scene_chars=scene_chars,
        )

        scenes.append(ScenePlan(
            page_index=index,
            source_text=line,
            scene_spec=scene_spec,
            prompt=prompt,
            character_ids=scene_spec.focus_character_ids,
        ))
    return scenes


def _get_scene_character_ids(llm_scene_prompt: str, characters: list[StoryCharacter]) -> list[str]:
    if not llm_scene_prompt:
        return [characters[0].id] if characters else []

    prompt_lower = llm_scene_prompt.lower()
    found: list[str] = []
    for ch in characters:
        matched = False
        if ch.species and ch.species.lower() in prompt_lower:
            matched = True
        if ch.job and ch.job.lower() in prompt_lower:
            matched = True
        if ch.id not in ("protagonist", "main_character") and ch.id.lower() in prompt_lower:
            matched = True
        if matched and ch.id not in found:
            found.append(ch.id)

    return found if found else ([characters[0].id] if characters else [])


def build_scene_spec(
    page_index: int,
    line: str,
    characters: list[StoryCharacter],
    character_bible: dict[str, str],
    llm_scene_prompt: str = "",
    llm_focus_characters: list[str] | None = None,
) -> SceneSpec:
    """한 줄을 최소 구조의 장면 명세로 변환한다."""
    if llm_scene_prompt:
        narrative_hint = llm_scene_prompt
    else:
        narrative_hint = extract_scene_keywords(line)

    if llm_focus_characters:
        focus_character_ids = [
            cid for cid in llm_focus_characters
            if any(ch.id == cid for ch in characters)
        ]
    else:
        focus_character_ids = []

    if not focus_character_ids:
        focus_character_ids = _get_scene_character_ids(narrative_hint, characters)

    if not focus_character_ids:
        focus_character_ids = [characters[0].id] if characters else []

    is_multi = len(focus_character_ids) >= 2
    if is_multi:
        staging_hint = (
            "Both characters are clearly separate individuals. "
            "Do NOT merge or blend them into one figure."
        )
    else:
        staging_hint = "The character is prominently placed in the scene."

    return SceneSpec(
        page_index=page_index,
        source_text=line,
        focus_character_ids=focus_character_ids,
        narrative_hint=narrative_hint,
        staging_hint=staging_hint,
    )


# ── Klein 자연어 프롬프트 빌더 ────────────────────────────────────────────────


def build_prompt(
    scene_spec: SceneSpec,
    world: WorldProfile,
    character_bible: dict[str, str],
    scene_chars: list[StoryCharacter],
) -> str:
    """FLUX.2-klein-4B Qwen3 인코더용 자연어 프롬프트.

    핵심 원칙:
    - LLM이 생성한 scene_prompt(동적 구도 + 감정 + 카메라 앵글)를 그대로 사용
    - LLM이 생성한 visual_hint(시대 고증된 묘사)를 그대로 사용
    - 프롬프트 빌더는 구조를 잡는 역할만, 내용은 LLM 결과를 최대한 보존
    """
    # 1. 스타일
    style = STYLE_PREFIX

    # 2. 캐릭터 묘사 — LLM visual_hint 우선 (시대 고증 포함)
    char_parts = []
    for i, ch in enumerate(scene_chars):
        if ch.visual_hint and len(ch.visual_hint.split()) >= 3:
            desc = ch.visual_hint
        elif character_bible.get(ch.id):
            desc = character_bible[ch.id]
        else:
            if ch.species:
                desc = f"a {ch.species}"
            elif ch.job:
                desc = f"a {ch.job}"
            else:
                desc = "a storybook character"

        # 동물/사람 체형 명시 (LLM이 놓쳤을 때 보강)
        if ch.type == "animal" or ch.species:
            if "animal body" not in desc and "full animal" not in desc:
                desc = f"{desc} (full animal body, NOT human anatomy)"
        elif ch.type == "human":
            desc = f"{desc} (storybook-style human: rounded face, large expressive eyes, warm simplified features, NOT realistic)"

        char_parts.append(desc)

    # 캐릭터 수에 따른 문장 구성
    if len(char_parts) == 1:
        char_sentence = f"Character: {char_parts[0]}."
    elif len(char_parts) == 2:
        char_sentence = (
            f"Two characters in this scene — "
            f"LEFT: {char_parts[0]}. "
            f"RIGHT: {char_parts[1]}. "
            f"They are SEPARATE individuals, do NOT merge them."
        )
    else:
        numbered = " ".join(f"[{i+1}] {p}." for i, p in enumerate(char_parts))
        char_sentence = f"{len(char_parts)} characters: {numbered}"

    # 3. 장면 — LLM의 동적 구도/감정/카메라 앵글 그대로 사용
    action = f"Scene: {scene_spec.narrative_hint}." if scene_spec.narrative_hint else ""

    # 4. 배경/세계관
    background = f"Setting: {world.positive_hint}."

    # 5. 다중 캐릭터 분리 힌트 (multi-char일 때만)
    staging = scene_spec.staging_hint if len(scene_chars) >= 2 else ""

    parts = [style, char_sentence]
    if action:
        parts.append(action)
    parts.append(background)
    if staging:
        parts.append(staging)

    return " ".join(parts)


# ── 키워드 추출 (LLM fallback용) ─────────────────────────────────────────────


def extract_scene_keywords(line: str) -> str:
    """한국어/영어 문장에서 장면 묘사용 영어 키워드를 추출한다 (LLM 실패 시 fallback)."""
    LOCATION_MAP: list[tuple[str, str]] = [
        ("궁궐", "palace courtyard"), ("관아", "government office"),
        ("기와집", "tiled-roof house"), ("초가집", "thatched cottage"),
        ("한옥", "hanok house"), ("장터", "busy marketplace"),
        ("우물", "stone well"), ("어귀", "village entrance"),
        ("마을", "village"), ("절", "temple"),
        ("논", "rice paddy field"), ("밭", "countryside field"),
        ("돌다리", "stone bridge"), ("나무다리", "wooden bridge"),
        ("고을", "countryside town"),
        ("숲속", "deep forest"), ("숲", "forest"),
        ("강가", "riverside"), ("강", "river"),
        ("개울", "stream"), ("산속", "mountain forest"),
        ("산", "mountain"), ("들판", "open meadow"),
        ("동굴", "cave"), ("바닷가", "ocean shore"),
        ("바다", "ocean"), ("연못", "pond"),
        ("호수", "lake"), ("나무 위", "up in the tree"),
        ("나무", "under the trees"), ("꽃밭", "flower field"),
        ("구름", "among the clouds"),
        ("노을", "warm sunset sky"), ("달빛", "moonlit night"),
        ("밤", "quiet night"), ("새벽", "early dawn"),
        ("아침", "bright morning"), ("낮", "sunny daytime"),
        ("저녁", "golden evening"),
        ("봄", "spring blossoms"), ("여름", "lush summer"),
        ("가을", "autumn leaves"), ("겨울", "snowy winter"),
    ]

    ACTION_MAP: list[tuple[str, str]] = [
        ("걸어", "walking"), ("걸었", "walking"),
        ("뛰어", "running"), ("뛰었", "running"),
        ("달려", "running"), ("달렸", "running"),
        ("날아", "flying"), ("날았", "flying"),
        ("헤엄", "swimming"),
        ("올라", "climbing"), ("올랐", "climbing"),
        ("내려", "descending"), ("내렸", "descending"),
        ("앉아", "sitting"), ("앉았", "sitting"),
        ("누워", "lying down"), ("누웠", "lying down"),
        ("서서", "standing"), ("서있", "standing"),
        ("바라보", "gazing"), ("울고", "crying"),
        ("웃으", "smiling"), ("웃었", "smiling"),
        ("놀라", "surprised expression"), ("놀랐", "surprised expression"),
        ("기뻐", "joyful expression"), ("슬퍼", "sad expression"),
        ("화가", "angry expression"), ("두려워", "frightened expression"),
        ("고민", "thinking pose"), ("생각", "thinking pose"),
        ("춤을", "dancing"), ("춤추", "dancing"),
        ("말하", "talking"), ("외치", "shouting"), ("속삭", "whispering"),
    ]

    def _best_match(text: str, mapping: list[tuple[str, str]]) -> str | None:
        best_pos = len(text) + 1
        best_english = None
        for keyword, english in mapping:
            idx = text.find(keyword)
            if idx != -1 and idx < best_pos:
                best_pos = idx
                best_english = english
        return best_english

    found_keywords: list[str] = []
    loc = _best_match(line, LOCATION_MAP)
    if loc:
        found_keywords.append(loc)
    act = _best_match(line, ACTION_MAP)
    if act:
        found_keywords.append(act)

    return ", ".join(found_keywords)


# ── 유틸리티 ─────────────────────────────────────────────────────────────────


def split_story_lines(text: str) -> list[str]:
    return [
        normalize_line(part)
        for part in re.split(r"[\r\n]+|(?<=[.!?])\s+", text)
        if normalize_line(part)
    ]


def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    return re.sub(r"[\"'`]+", "", line)


def optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def normalize_character_type(value: Any) -> str:
    normalized = str(value or "other").strip().lower()
    if normalized in {"human", "animal", "other"}:
        return normalized
    return "other"


def shorten_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
