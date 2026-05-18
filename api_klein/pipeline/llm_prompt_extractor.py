"""GPT-4o를 사용해 동화 텍스트에서 삽화 프롬프트를 추출한다 (Klein 전용).

기존 api/pipeline/llm_prompt_extractor.py 독립 복사본.
주요 변경:
  - 하드코딩된 특정 직업 번역 규칙 제거
    → LLM이 스토리 맥락(시대, 문화)을 스스로 파악하여 시대 고증된 묘사 생성
  - FLUX.2-klein-4B Qwen3 40k 토큰 인코더에 맞춰 자연어 문장 강조
  - 동적 구도·만화적 과장 표현 규칙 포함
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Kafka 기반 파이프라인용 — Propp 민담 형태론 역할 스키마 (가변)
# ─────────────────────────────────────────────────────────────────────────────

# 알려진 역할들 (Propp 형태론). 메시지에 어느 부분집합이든 들어올 수 있음.
KNOWN_ROLES = ("HERO", "VILLAIN", "HELPER", "DISPATCHER", "FALSE_HERO", "DONOR")
ROLES = KNOWN_ROLES  # 이전 import 호환용


_CHARACTER_BIBLE_SYSTEM = """\
You generate visual descriptions of fairytale characters for a children's picture-book
illustration generator (FLUX.2-klein-4B, Qwen3 text encoder).

You will receive:
- setting: cultural/historical setting (e.g. KOREAN_TRADITIONAL, EUROPEAN_MEDIEVAL)
- character_type: applies to ALL roles in this fairytale. One of HUMAN / ANIMAL / ETC.
  - HUMAN: every character is rendered as a human child/person
  - ANIMAL: every character is anthropomorphic — animal body, expressive
            child-like proportions, animal-appropriate cultural attire
  - ETC: every character is a spirit / dokkaebi / fairy / celestial being
  This determines the *type* field for EVERY role in the output. Do NOT mix types.
  If a Korean name suggests a different type (e.g. character_type=ANIMAL but
  the name is "할아버지"), still honor character_type — render "할아버지" as an
  elderly anthropomorphic animal (e.g. a noble old rabbit in hanbok). The cast
  must be visually coherent.
- characters: dict mapping Propp morphology role → Korean character name.
  Roles can be any subset of: HERO, VILLAIN, HELPER, DISPATCHER, FALSE_HERO, DONOR.

For EACH role, produce:
- type: ALWAYS equal to the input character_type (HUMAN/ANIMAL/ETC).
        Do NOT infer per-role types from Korean names — the entire cast
        shares the same type.
- visual_description: a vivid English description with:
    1. Body / skin / fur / scale appearance
    2. Era-accurate clothing or covering matching the setting
    3. ONE distinctive accessory or prop
    4. ONE unique physical feature
    5. Size descriptor (small, large, stocky, slim, ...)

IMPORTANT — do NOT include emotion or expression words in visual_description.
Avoid: smiling, friendly expression, cheerful, menacing, glaring, angry, sad,
       sparkling eyes, soft smile, etc.
The reference image generated from this description is reused across ALL pages, so
the character must look neutral and emotion-free in the reference. Per-page emotion
is controlled separately by the scene prompt.

NEVER include injury or violence-related details as physical features.
This applies to EVERY role (HERO/VILLAIN/HELPER/DISPATCHER/FALSE_HERO/DONOR)
and EVERY type (HUMAN/ANIMAL/ETC). VILLAIN does NOT get a free pass for this.
- NO scars, wounds, cuts, bruises, blood, broken/missing limbs, eyepatches over injuries
- NO "fierce look from a battle scar", "weathered face from old fights", or similar
This is a children's picture book — every character appears visually whole and unharmed.

Good safe choices for "ONE unique physical feature" (item 4) — pick whichever fits the type:

For HUMAN / ETC (people, dokkaebi, fairies, spirits):
- distinctive hair (long braid, curly bangs, twin ponytails, gray streak, hair color)
- specific eye color (golden, emerald, deep blue)
- a single small mole or freckle (cosmetic only, not injury)
- glasses or monocle (optional)
- horn shape (for dokkaebi/oni-like) or pointed ear tip (for fairy/elf-like)
- a glowing halo or magical aura (for celestial/spirit characters)

For ANIMAL:
- distinctive ear shape (tufted ears, drooping ears, oversized ears)
- distinctive tail (bushy, ringed pattern, extra long, short pom-pom)
- fur/feather/scale color variation (white chest blaze, dark mask around eyes, color-tipped tail)
- oversized paws / webbed feet / fluffy chest fur / shell pattern
- natural markings (stripes, spots, swirls — natural patterns only)
- specific eye color (golden, emerald, amber)

Universal options (any type):
- a single small accessory-feature like a flower tucked behind the ear, a leaf in the hair

Cultural accuracy: use your knowledge of the setting to choose the right attire.
- Joseon Korea: hanbok (jeogori + baji/chima), gat hat for nobility, straw sandals for commoners,
  rough hemp work clothes + white headband for woodcutter (나무꾼),
  flowing celestial robe with ribbon ornament for celestial maiden (선녀, NO wings)
- Medieval Europe: tunic, hose, cloak; knight = chainmail; witch = black robe + pointed hat
- Fantasy: world-appropriate fantasy outfit
NOT: Chinese cross-collar hanfu (wrong culture), modern clothes, generic anime proportions

Output strictly valid JSON, no markdown. Keys MUST match input role names exactly:
{
  "<ROLE_NAME>": { "name": "<Korean name>", "type": "ANIMAL|HUMAN|ETC", "visual_description": "..." },
  ...
}
Include an entry for every role in the input. Do NOT add roles not in the input.
"""


def extract_character_bible(
    setting: str,
    character_type: str,
    characters: dict[str, str],
    model: str = "gpt-4o",
) -> dict[str, dict[str, str]]:
    """동화 1편 시작 시 1회 — 역할별 visual_description 생성.

    Args:
        setting: 예 "KOREAN_TRADITIONAL"
        character_type: 전체 캐릭터의 의인화 타입. "HUMAN" / "ANIMAL" / "ETC"
                        — 입력 dict의 모든 role이 이 타입으로 통일 렌더링됨.
        characters: {역할: 한국어 이름} 형태. 역할 수는 가변.
                    예: {"HERO": "개구리", "VILLAIN": "늑대", "HELPER": "자라",
                         "DISPATCHER": "사자", "FALSE_HERO": "여우"}
        model: 사용할 OpenAI 모델

    Returns:
        {
          역할명: {"name": "...", "type": "...", "visual_description": "..."},
          ...
        }
        입력 characters 의 모든 역할에 대해 항목이 생성됨.
    """
    from openai import OpenAI
    client = OpenAI()

    role_lines = "\n".join(f"  {role}: {name}" for role, name in characters.items())
    user_prompt = (
        f"setting: {setting}\n"
        f"character_type (applies to ALL roles — they all share this type): {character_type}\n"
        f"characters:\n{role_lines}\n\n"
        f"Generate the JSON character bible following the system instructions. "
        f"The JSON must have exactly these keys: {list(characters.keys())}."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CHARACTER_BIBLE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    forced_type = character_type.upper()
    bible: dict[str, dict[str, str]] = {}
    for role in characters.keys():
        entry = parsed.get(role, {}) or {}
        bible[role] = {
            "name": str(entry.get("name", characters.get(role, ""))),
            "type": forced_type,
            "visual_description": str(entry.get("visual_description", "")),
        }
    return bible


_PAGE_SCENE_SYSTEM = """\
You analyze a single page (2-3 Korean sentences) of a fairytale and extract scene info.

Inputs:
- setting (e.g. KOREAN_TRADITIONAL)
- character_bible: pre-existing visual descriptions keyed by role name
- sentences: Korean text

For EACH page, produce:
1. focus_roles — which roles APPEAR visually in this page (subset of bible keys, non-empty).
   "Mentioning" a name doesn't count — the character must physically be in the depicted moment.
2. narrative_hint — single English scene sentence (max ~50 words) including:
   - what is happening (action, body language, emotion)
   - location/environment hint
   - camera angle (close-up / wide shot / low angle / over-the-shoulder / action / bird's-eye)
   Do NOT include art-style words ("illustration", "pastel", "cartoon").
   Do NOT redescribe character outfits — that's in the bible.

━━━ EMOTION SYMBOLS (optional accent — graphic only, NO text) ━━━
For wordless strong moments you MAY place a SINGLE graphic emotion shape near the
character's head. These are pure visual icons (the model cannot render readable text).

  Shock / surprise   → "a bold red exclamation mark '!' shape above [character]'s head"
  Confusion          → "a curling question mark '?' shape beside [character]'s head"
  Realization        → "a glowing yellow exclamation '!' shape above [character]'s head"
  Shock + confusion  → "interrobang '?!' shape bursting above [character]'s head"
  Affection          → "small pink heart shapes '♥' floating around [character]'s head"
  Wonder             → "small star shapes '★' twinkling around [character]'s head"
  Anger              → "manga cross-hatched anger lines '#' on [character]'s forehead"
  Embarrassment      → "a single large teardrop sweatdrop next to [character]'s temple"
  Joy / music        → "small musical note shapes '♪' floating around [character]"

Rules:
- ONE symbol max, only when body language alone wouldn't convey the strong moment.
- Describe as a SHAPE/ICON, never as containing text or words.
- Quiet scenes → no symbol. Body language alone.

━━━ MULTI-CHARACTER POSITIONING ━━━
If 2+ roles are in focus_roles, place them spatially:
"[role A] on the left, [role B] on the right"
Each character has their OWN action — abilities don't transfer between characters.

━━━ DO NOT INTRODUCE EXTRA CHARACTERS ━━━
The illustration must contain ONLY the characters in focus_roles. Never describe other
animals, people, or creatures in the scene — even if the Korean text mentions them.

If the Korean text uses vague group words like "친구들" (friends), "동물들" (animals),
"무리" (group), 사람들 (people), DO NOT depict specific companions. Instead:
- Treat the scene as the named roles' personal moment
- Describe the SETTING / ATMOSPHERE instead of a crowd
- BAD: "the frog plays with his friends in the pond"
       (model invents random bears, rabbits, ducks)
- GOOD: "the frog hops joyfully alone among lily pads, dragonflies fluttering above the calm pond"
       (specific safe atmospheric details — no character invention)
- BAD: "the lion warns the frog while other animals watch"
       (model adds extra animals)
- GOOD: "the lion stands beside the frog at the edge of the forest, autumn leaves drifting nearby"

This rule is critical — diffusion models will hallucinate any character mentioned
in the prompt, breaking story consistency.

Output strictly valid JSON, no markdown:
{ "narrative_hint": "...", "focus_roles": ["<ROLE>", ...] }
Use literal role name strings (HERO, VILLAIN, ...).
"""


def extract_page_scene(
    sentences: list[str],
    character_bible: dict[str, dict[str, str]],
    setting: str,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """페이지(2-4문장) → 장면 묘사 + 등장 역할 추출.

    Returns:
        { "narrative_hint": "...", "focus_roles": ["<ROLE>", ...] }
    """
    from openai import OpenAI
    client = OpenAI()

    sentence_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences) if s.strip())
    bible_json = json.dumps(character_bible, ensure_ascii=False, indent=2)

    user_prompt = (
        f"setting: {setting}\n\n"
        f"character_bible:\n{bible_json}\n\n"
        f"sentences (Korean):\n{sentence_lines}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PAGE_SCENE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    focus = parsed.get("focus_roles") or []
    if isinstance(focus, str):
        focus = [focus]
    available = set(character_bible.keys())
    focus = [r for r in focus if r in available]
    if not focus:
        # 안전망: bible에 HERO 있으면 HERO, 아니면 첫 역할
        focus = ["HERO"] if "HERO" in available else [next(iter(available), "HERO")]

    return {
        "narrative_hint": str(parsed.get("narrative_hint", "")),
        "focus_roles": focus,
    }
