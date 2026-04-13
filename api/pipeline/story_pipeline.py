"""동화 스토리를 삽화용 프롬프트로 변환하는 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from api.pipeline.llm_prompt_extractor import extract_story_prompts as _llm_extract
from api.pipeline.story_analyzer import analyze_story as _analyze_story
from api.pipeline.config import (
    CHARACTER_TYPE_HINTS,
    JOB_HINTS,
    SPECIES_FALLBACK_TEMPLATE,
    SPECIES_HINTS,
    STYLE_PRESETS,
    THEME_CHARACTER_COSTUME_HINTS,
    THEME_EXPANSIONS,
)


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
    subject_hint: str
    narrative_hint: str
    staging_hint: str


@dataclass
class ScenePlan:
    page_index: int
    source_text: str
    scene_spec: SceneSpec
    prompt: str
    prompt_2: str
    negative_prompt: str
    character_ids: list[str]
    lora_scale: float = 0.8


@dataclass
class StoryPlan:
    story_input: StoryInput
    quality_gate: QualityGateReport
    world: WorldProfile
    character_bible: dict[str, str]
    scenes: list[ScenePlan]


def build_story_plan(story_data: str | dict[str, Any], max_scenes: int = 10) -> StoryPlan:
    """구조화된 스토리 입력을 장면별 프롬프트 계획으로 변환한다."""
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
            print("[StoryPipeline] LLM 프롬프트 추출 성공")
            return normalize_story_input(analyzed, max_scenes=max_scenes)
        except Exception as e:
            print(f"[StoryPipeline] LLM 추출 실패, 키워드 매핑으로 fallback: {e}")

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
    """캐릭터 메타데이터를 시각 묘사로 압축한다."""
    bible: dict[str, str] = {}
    for character in characters:
        if character.visual_hint and len(character.visual_hint.split()) >= 5:
            vh = character.visual_hint
            if character.species and character.species.lower() not in vh.lower():
                vh = f"cute {character.species}, {vh}"
            bible[character.id] = shorten_text(vh, max_words=20).rstrip(", ")
            continue

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
        bible[character.id] = shorten_text(", ".join(cleaned_parts), max_words=20).rstrip(", ")
    return bible


def build_scene_plans(
    story_input: StoryInput,
    world: WorldProfile,
    character_bible: dict[str, str],
) -> list[ScenePlan]:
    """스토리 한 줄마다 하나의 장면 프롬프트를 만든다."""
    style = STYLE_PRESETS["fairytale_pastel"]
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

        is_multi = len(scene_chars) >= 2

        if is_multi:
            character_prompt = build_character_prompt(
                scene_chars, character_bible, world.character_costume_hints, include_all=True,
            )
        else:
            character_prompt = build_character_prompt(
                scene_chars, character_bible, world.character_costume_hints, include_all=False,
            )

        prompt, prompt_2 = build_prompt_from_scene_spec(
            scene_spec=scene_spec,
            style_positive=style["positive"],
            world_positive=world.positive_hint,
            character_prompt=character_prompt,
            scene_chars=scene_chars,
        )

        neg_style = style["negative"]
        if is_multi:
            for remove_phrase in [
                "multiple characters, ",
                "crowd scene, ",
                "many figures, ",
                "group of characters, ",
            ]:
                neg_style = neg_style.replace(remove_phrase, "")

        neg_parts = [neg_style, world.negative_hint]
        neg_parts.append("picture frame, book page, border, frame, page layout, collage")

        if is_multi:
            neg_parts.append("hybrid creature, merged characters, character fusion, fused body")

        has_animal = any(ch.species for ch in scene_chars)
        has_human = any(ch.type == "human" for ch in scene_chars)
        all_animal = all(ch.type == "animal" or ch.species for ch in scene_chars)

        if all_animal:
            neg_parts.append("human, person, human face, human body, humanoid, girl, boy, woman, man")
        elif has_animal and has_human:
            neg_parts.append("anthropomorphic, animal ears on human, kemonomimi, furry humanoid")

        negative_prompt = ", ".join(part for part in neg_parts if part)
        scenes.append(
            ScenePlan(
                page_index=index,
                source_text=line,
                scene_spec=scene_spec,
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                character_ids=scene_spec.focus_character_ids,
                lora_scale=0.70 if is_multi else 0.8,
            )
        )
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
        staging_hint = "two characters, large in foreground"
    else:
        staging_hint = "solo, large character in foreground"

    return SceneSpec(
        page_index=page_index,
        source_text=line,
        focus_character_ids=focus_character_ids,
        subject_hint="",
        narrative_hint=narrative_hint,
        staging_hint=staging_hint,
    )


def build_prompt_from_scene_spec(
    scene_spec: SceneSpec,
    style_positive: str,
    world_positive: str,
    character_prompt: str,
    scene_chars: list["StoryCharacter"] | None = None,
) -> tuple[str, str]:
    """장면 명세를 SDXL 듀얼 프롬프트로 조립한다."""
    scene_chars = scene_chars or []

    style_short = "ftbookstyle, flat cartoon illustration"
    prompt_parts = [style_short, character_prompt]
    prompt = ", ".join(p.strip().rstrip(", ") for p in prompt_parts if p and p.strip())

    char_type_hint = _build_char_type_hint(scene_chars)
    prompt_2_parts = [
        style_positive,
        world_positive,
        char_type_hint,
        scene_spec.narrative_hint,
        scene_spec.staging_hint,
    ]
    prompt_2 = ", ".join(p.strip().rstrip(", ") for p in prompt_2_parts if p and p.strip())

    return prompt, prompt_2


def _build_char_type_hint(chars: list["StoryCharacter"]) -> str:
    if not chars:
        return ""
    parts = []
    for ch in chars:
        if ch.species:
            color = ""
            if ch.visual_hint:
                for c in ["white", "brown", "black", "gray", "green", "blue",
                           "red", "yellow", "orange", "pink", "golden"]:
                    if c in ch.visual_hint.lower():
                        color = c + " "
                        break
            parts.append(f"{color}{ch.species} animal")
        elif ch.type == "human" and ch.job:
            parts.append(f"human {ch.job}")
        elif ch.type == "human":
            parts.append("human character")
        else:
            job_str = ch.job or "magical character"
            parts.append(job_str)
    if len(parts) == 1:
        return f"a {parts[0]}"
    elif len(parts) == 2:
        return f"a {parts[0]} and a {parts[1]}"
    else:
        return ", ".join(f"a {p}" for p in parts)


def _short_character_desc(ch: StoryCharacter) -> str:
    parts: list[str] = []
    if ch.species:
        parts.append(f"cute {ch.species}")
    elif ch.type == "human":
        parts.append("cute character")
    else:
        parts.append("storybook character")

    if ch.job:
        job_key = ch.job.lower()
        hint = JOB_HINTS.get(job_key, "")
        if hint:
            first_phrase = hint.split(",")[0].replace("wearing ", "in ")
            parts.append(first_phrase)
        else:
            parts.append(ch.job)

    return " ".join(parts)


def build_character_prompt(
    characters: list[StoryCharacter],
    character_bible: dict[str, str],
    costume_hints: dict[str, str] | None = None,
    include_all: bool = False,
) -> str:
    """캐릭터 정보와 테마 의상 힌트를 결합해 프롬프트 조각으로 만든다."""
    if not characters:
        return ""

    costume_hints = costume_hints or {}

    if include_all and len(characters) > 1:
        multi_parts = []
        for ch in characters:
            bible_desc = character_bible.get(ch.id, "")
            if bible_desc:
                desc = shorten_text(bible_desc, max_words=10)
            else:
                desc = _short_character_desc(ch)
            if ch.species:
                desc = f"{desc}, {ch.species} animal NOT human"
            elif ch.type == "human":
                desc = f"{desc}, human person NOT animal"
            multi_parts.append(desc)
        return " BREAK ".join(multi_parts)

    primary = characters[0]
    result_parts: list[str] = []
    bible_desc = character_bible.get(primary.id, "")
    if bible_desc:
        result_parts.append(bible_desc)

    has_job_costume = bool(primary.job and JOB_HINTS.get(primary.job.lower()))
    if not has_job_costume:
        costume = costume_hints.get(primary.type, "")
        if costume:
            result_parts.append(costume)

    if not result_parts:
        return "storybook character, cute proportions, round face"

    cleaned = [p.rstrip(", ") for p in result_parts if p and p.strip()]
    return ", ".join(cleaned)


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
        ("palace", "palace"), ("forest", "forest"), ("village", "village"),
        ("river", "river"), ("mountain", "mountain"), ("ocean", "ocean"),
        ("night", "night"), ("sunset", "sunset"), ("morning", "morning"),
        ("field", "field"), ("marketplace", "marketplace"),
        ("well", "stone well"), ("temple", "temple"), ("bridge", "bridge"),
    ]

    ACTION_MAP: list[tuple[str, str]] = [
        ("걸어", "walking"), ("걸었", "walking"),
        ("뛰어", "running"), ("뛰었", "running"),
        ("달려", "running"), ("달렸", "running"),
        ("날아", "flying"), ("날았", "flying"),
        ("헤엄", "swimming"),
        ("올라", "climbing"), ("올랐", "climbing"),
        ("내려", "descending"), ("내렸", "descending"),
        ("건너", "crossing"), ("건넜", "crossing"),
        ("폴짝", "leaping"), ("기어", "crawling"),
        ("앉아", "sitting"), ("앉았", "sitting"),
        ("누워", "lying down"), ("누웠", "lying down"),
        ("엎드", "crouching"),
        ("쉬고", "resting"), ("쉬었", "resting"),
        ("서서", "standing"), ("서있", "standing"),
        ("바라보", "gazing"), ("들여다", "peering into"),
        ("쳐다보", "looking up"), ("내려다", "looking down"),
        ("둘러보", "looking around"),
        ("울어", "crying"), ("울었", "crying"), ("울고", "crying"),
        ("웃으", "smiling"), ("웃었", "smiling"), ("웃으며", "smiling"),
        ("놀라", "surprised expression"), ("놀랐", "surprised expression"),
        ("기뻐", "joyful expression"), ("기뻤", "joyful expression"),
        ("기쁘", "joyful expression"),
        ("슬퍼", "sad expression"), ("슬펐", "sad expression"),
        ("화가", "angry expression"),
        ("무서워", "frightened expression"), ("두려워", "frightened expression"),
        ("반가워", "happy greeting"), ("반가웠", "happy greeting"),
        ("고민", "thinking pose"), ("생각", "thinking pose"),
        ("춤을", "dancing"), ("춤추", "dancing"),
        ("노래", "singing"),
        ("말하", "talking"), ("외치", "shouting"), ("속삭", "whispering"),
        ("먹으", "eating"), ("먹었", "eating"), ("드셨", "eating"),
        ("마시", "drinking"),
        ("건네", "handing over"), ("건넸", "handing over"),
        ("받아", "receiving"), ("받았", "receiving"),
        ("주었", "giving"), ("드렸", "giving"),
        ("안아", "hugging"), ("안고", "hugging"),
        ("싸우", "fighting"),
        ("도와", "helping"), ("도왔", "helping"),
        ("찾아", "searching"), ("찾았", "searching"),
        ("숨어", "hiding"), ("숨었", "hiding"),
        ("도망", "fleeing"),
        ("열어", "opening"), ("열었", "opening"),
        ("닫아", "closing"), ("닫았", "closing"),
        ("절하", "bowing"), ("인사", "greeting"),
        ("walking", "walking"), ("running", "running"), ("sitting", "sitting"),
        ("crying", "crying"), ("smiling", "smiling"), ("sleeping", "sleeping"),
        ("dancing", "dancing"), ("singing", "singing"),
        ("hiding", "hiding"), ("searching", "searching"),
        ("hugging", "hugging"), ("laughing", "laughing"),
    ]

    OBJECT_MAP: list[tuple[str, str]] = [
        ("바구니", "holding basket"), ("꽃다발", "holding bouquet"),
        ("편지", "holding letter"), ("책", "holding book"),
        ("등불", "glowing lantern"), ("횃불", "glowing torch"),
        ("부채", "holding fan"), ("지팡이", "with walking stick"),
        ("우산", "holding umbrella"), ("보따리", "carrying bundle"),
        ("상자", "with wooden box"), ("열쇠", "holding key"),
        ("두루마리", "holding scroll"), ("붓", "with ink brush"),
        ("약", "holding medicine bottle"), ("보자기", "with cloth bundle"),
        ("가마", "riding palanquin"), ("연", "holding kite"),
        ("꽃", "holding flower"), ("사과", "holding apple"),
        ("물고기", "with fish"), ("음식", "with food"),
        ("밥", "with rice bowl"), ("떡", "with rice cake"),
        ("과일", "with fruit"),
        ("basket", "holding basket"), ("flower", "holding flower"),
        ("letter", "holding letter"), ("lantern", "with glowing lantern"),
        ("umbrella", "holding umbrella"), ("key", "holding key"),
        ("book", "holding book"), ("scroll", "holding scroll"),
    ]

    ATMOSPHERE_MAP: list[tuple[str, str]] = [
        ("비가", "rainy weather"), ("빗속", "in the rain"), ("비", "light rain"),
        ("눈이", "snowy weather"), ("눈송이", "falling snowflakes"), ("눈", "snowy"),
        ("바람", "windy"), ("폭풍", "stormy"),
        ("안개", "misty foggy"),
        ("맑", "clear sunny sky"), ("화창", "bright sunny"),
        ("어둡", "dark gloomy"), ("캄캄", "pitch dark"),
        ("따뜻", "warm golden light"), ("포근", "cozy warm"),
        ("추", "cold chilly"), ("서늘", "cool breeze"),
        ("반짝", "sparkling light"), ("빛나", "glowing light"),
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

    obj = _best_match(line, OBJECT_MAP)
    if obj:
        found_keywords.append(obj)

    atm = _best_match(line, ATMOSPHERE_MAP)
    if atm:
        found_keywords.append(atm)

    return ", ".join(found_keywords)


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
