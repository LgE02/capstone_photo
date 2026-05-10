"""Klein 스토리 파이프라인.

동화 텍스트를 GPT-4o로 분석해 캐릭터/장면 메타데이터를 만들고,
FLUX.2-klein-4B Qwen3 인코더용 자연어 프롬프트로 조립한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from api_klein.pipeline.config import THEME_CHARACTER_COSTUME_HINTS, THEME_EXPANSIONS
from api_klein.pipeline.llm_prompt_extractor import extract_story_prompts as _llm_extract


# ── 데이터 모델 ──────────────────────────────────────────────────────────────


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


STYLE_PREFIX = (
    "children's book illustration, soft cell shading with smooth color gradients, "
    "warm teal and golden tones, glowing light particles scattered in the background, "
    "painterly digital art style, no hard outlines, lush soft background. "
)


# ── 메인 엔트리 ──────────────────────────────────────────────────────────────


def build_story_plan(story_data: str | dict[str, Any], max_scenes: int = 10) -> StoryPlan:
    """동화 입력을 장면별 Klein 프롬프트 계획으로 변환."""
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


# ── 입력 정규화 ──────────────────────────────────────────────────────────────


def normalize_story_input(
    story_data: str | list[str] | dict[str, Any],
    max_scenes: int = 10,
) -> StoryInput:
    """원본 텍스트나 JSON 입력을 내부 스키마로 정규화 (GPT-4o 단일 경로)."""
    if isinstance(story_data, list) and story_data and isinstance(story_data[0], str):
        joined = "\n".join(line.strip() for line in story_data if isinstance(line, str) and line.strip())
        return normalize_story_input(joined, max_scenes=max_scenes)

    if isinstance(story_data, str):
        stripped = story_data.strip()
        if stripped.startswith("{"):
            return normalize_story_input(json.loads(stripped), max_scenes=max_scenes)

        analyzed = _llm_extract(stripped, max_scenes=max_scenes)
        print("[StoryPipeline] LLM 프롬프트 추출 성공")
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
    """테마 키를 실제 프롬프트 힌트로 변환."""
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
    """캐릭터 메타데이터를 시각 묘사로 압축.

    GPT-4o가 visual_hint를 충분히 채워 주는 단일 경로 전제 →
    visual_hint가 있으면 그대로 사용. 없을 때만 species/job 기반 짧은 기본값.
    """
    bible: dict[str, str] = {}
    for character in characters:
        if character.visual_hint:
            vh = character.visual_hint
            if character.species and character.species.lower() not in vh.lower():
                vh = f"cute {character.species}, {vh}"
            bible[character.id] = vh
            continue

        if character.species:
            bible[character.id] = f"cute {character.species} character"
        elif character.job:
            bible[character.id] = f"a {character.job}"
        else:
            bible[character.id] = "storybook character"
    return bible


# ── 장면 계획 ────────────────────────────────────────────────────────────────


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


def build_scene_spec(
    page_index: int,
    line: str,
    characters: list[StoryCharacter],
    character_bible: dict[str, str],
    llm_scene_prompt: str = "",
    llm_focus_characters: list[str] | None = None,
) -> SceneSpec:
    """한 줄을 최소 구조의 장면 명세로 변환."""
    narrative_hint = llm_scene_prompt or extract_scene_keywords(line)

    if llm_focus_characters:
        focus_character_ids = [
            cid for cid in llm_focus_characters
            if any(ch.id == cid for ch in characters)
        ]
    else:
        focus_character_ids = []

    if not focus_character_ids:
        focus_character_ids = _infer_scene_character_ids(narrative_hint, characters)

    if not focus_character_ids:
        focus_character_ids = [characters[0].id] if characters else []

    if len(focus_character_ids) >= 2:
        staging_hint = (
            "Both characters are large and clearly visible in the foreground. "
            "They are separate individuals, not merged."
        )
    else:
        staging_hint = "The character is large and prominently placed in the foreground."

    return SceneSpec(
        page_index=page_index,
        source_text=line,
        focus_character_ids=focus_character_ids,
        narrative_hint=narrative_hint,
        staging_hint=staging_hint,
    )


def _infer_scene_character_ids(llm_scene_prompt: str, characters: list[StoryCharacter]) -> list[str]:
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


# ── Klein 프롬프트 빌더 (Qwen3 인코더, 40,960 토큰) ───────────────────────────


def build_prompt(
    scene_spec: SceneSpec,
    world: WorldProfile,
    character_bible: dict[str, str],
    scene_chars: list[StoryCharacter],
) -> str:
    """FLUX.2-klein-4B Qwen3 인코더용 자연어 프롬프트.

    토큰 한도 40,960 → 상세하고 자연스러운 문장으로 작성.
    """
    char_parts: list[str] = []
    for i, ch in enumerate(scene_chars):
        if ch.visual_hint and len(ch.visual_hint.split()) >= 3:
            desc = ch.visual_hint
        elif character_bible.get(ch.id):
            desc = character_bible[ch.id]
        elif ch.species:
            desc = f"a cute {ch.species}"
        elif ch.job:
            desc = f"a {ch.job}"
        else:
            desc = "a storybook character"

        if ch.type == "animal" or ch.species:
            desc = f"{desc} — this character has a full animal body"
        elif ch.type == "human":
            desc = f"{desc} — this character is a human person"

        if len(scene_chars) == 2:
            position = "on the left side" if i == 0 else "on the right side"
            desc = f"{desc}, positioned {position}"

        char_parts.append(desc)

    if len(char_parts) == 1:
        char_sentence = f"The main character is {char_parts[0]}."
    elif len(char_parts) == 2:
        char_sentence = (
            f"There are two separate characters in this scene. "
            f"First: {char_parts[0]}. "
            f"Second: {char_parts[1]}."
        )
    else:
        numbered = " ".join(f"Character {i+1}: {p}." for i, p in enumerate(char_parts))
        char_sentence = f"There are {len(char_parts)} characters. {numbered}"

    action = f"Scene: {scene_spec.narrative_hint}." if scene_spec.narrative_hint else ""
    background = f"Background: {world.positive_hint}."
    composition = (
        "Wide establishing shot showing both the character and the background clearly. "
        + scene_spec.staging_hint
    )

    parts = [STYLE_PREFIX, char_sentence]
    if action:
        parts.append(action)
    parts.append(background)
    parts.append(composition)

    return " ".join(parts)


# ── 키워드 추출 (LLM이 narrative_hint를 못 줄 때만) ───────────────────────────


def extract_scene_keywords(line: str) -> str:
    """한국어/영어 문장에서 장면 묘사용 영어 키워드를 추출한다."""
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

    found: list[str] = []
    loc = _best_match(line, LOCATION_MAP)
    if loc:
        found.append(loc)
    act = _best_match(line, ACTION_MAP)
    if act:
        found.append(act)

    return ", ".join(found)


# ── 유틸 ─────────────────────────────────────────────────────────────────────


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
