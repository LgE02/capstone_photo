"""OpenAI Chat Completion API를 사용해 동화 텍스트에서 삽화 프롬프트를 추출한다 (Klein 전용).
모델 ID는 .env의 OPENAI_MODEL 로 제어 (기본값 gpt-5.5).

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


# OpenAI 모델 ID — .env 의 OPENAI_MODEL 로 교체 가능. 미설정 시 기본값 사용.
# 워커 시작 시 1회 평가되므로 모델 교체 후엔 워커 재시작 필요.
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")


def _completion_kwargs_for(model: str, max_tokens: int) -> dict:
    """모델별 호환성 파라미터 분기.

    - GPT-5 / o1 / o3 계열 (reasoning 모델):
        max_completion_tokens + reasoning_effort, temperature 자유 지정 불가.
    - 그 외 (gpt-4o, gpt-4o-mini, gpt-4-turbo 등 즉답형):
        max_tokens + temperature.

    .env 의 OPENAI_MODEL 만 바꿔도 호환성 자동 분기 — GPT-4o↔GPT-5.5 A/B 비교에 사용.
    """
    if model.startswith(("gpt-5", "o1", "o3")):
        return {
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
        }
    return {
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }


_CHARACTER_BIBLE_SYSTEM = """\
You generate visual descriptions of fairytale characters for a children's picture-book
illustration generator (FLUX.2-klein-4B, Qwen3 text encoder).

━━━ INPUTS ━━━
- setting: cultural/historical setting (KOREAN_TRADITIONAL, EUROPEAN_MEDIEVAL, etc.)
- char_species: HUMAN / ANIMAL / ETC — applies to ALL roles. Whole cast shares
  this same type for visual coherence.
  - HUMAN: human child/person, picture-book chibi proportions
  - ANIMAL: anthropomorphic — animal body, animal-appropriate cultural attire
  - ETC: spirit / dokkaebi / fairy / celestial being
  If a Korean name conflicts with char_species (e.g. ANIMAL + "할아버지"),
  honor char_species — render as elderly anthropomorphic animal in setting attire.
- characters: dict mapping Propp role → Korean name. Roles are a subset of:
  HERO, VILLAIN, HELPER, DISPATCHER, FALSE_HERO, DONOR.

━━━ OUTPUT (strict JSON, no markdown) ━━━
{
  "<ROLE>": {"name": "<Korean>", "type": "<HUMAN|ANIMAL|ETC>", "visual_description": "..."},
  ...
}
Include every input role. Add none beyond input.

━━━ VISUAL_DESCRIPTION STRUCTURE ━━━
A vivid English description containing:
1. Body / skin / fur / scale appearance
2. Era-accurate clothing or covering matching the setting
3. ONE distinctive accessory or prop that fits the archetype
4. ONE unique physical feature
5. Size descriptor (small, large, stocky, slim, ...)

The reference image is REUSED across all pages, so write a NEUTRAL appearance.
Do NOT include emotion words (smiling, glaring, friendly, menacing, sad).
Per-page emotion is handled separately by the scene prompt.

━━━ FACE & SAFETY RULES (every role, every type) ━━━
- Faces ALWAYS fully visible — NO mask, eyepatch, hood-covering-face, or any
  face-concealing accessory. Eyes clearly readable against surrounding fur/skin.
- For raccoon-like animals: write "raccoon-style natural fur pattern" — never the
  word "mask".
- NO injuries (scars, wounds, cuts, bruises, blood, missing limbs).

━━━ VILLAIN TONE — DARKER, NOT HORROR ━━━
VILLAIN should look DARKER and slightly intimidating — the reader senses unease
at first glance — but not horror-movie scary.

ENCOURAGED for VILLAIN:
- Darker color palette than HERO/HELPER (deep grey, deep brown, deep purple,
  forest green, midnight blue, charcoal — NOT pure jet black, NOT pastel)
- Imposing / watchful / weighty posture and silhouette
- One iconic archetype cue: witch's pointed hat / wolf's sharp profile and
  visible canines / bear's broad imposing build / tiger's bold stripes
- Descriptors like "imposing", "watchful", "weighty", "stern" are OK
  (these describe physique, not emotion)

FORBIDDEN for VILLAIN (these tip into horror):
- Glowing red eyes, dripping fangs bared at viewer
- Visible shadow aura around the body
- Skull motifs, spiked armor, chains, rope/leather warrior wrappings
- Stacking many darkness markers at once — pick AT MOST one or two natural ones
- Emotion words like "sinister", "evil-looking", "menacing", "creepy"
  (describe physique, not feelings)

The story's menace comes from PAGE ACTIONS (growling, blocking the path, looming)
— the reference just sets the dark tone.

━━━ CULTURAL ACCURACY QUICK GUIDE ━━━
- Joseon Korea: hanbok (jeogori + baji/chima), gat hat for nobility, straw
  sandals for commoners. Woodcutter: hemp clothes + white headband. Celestial
  maiden: flowing robe with ribbon ornament, NO wings.
- Medieval Europe: tunic + hose + cloak. Knight: chainmail. Witch: deep-color
  robe + pointed hat (single iconic cue).
- Fantasy: world-appropriate fantasy outfit.
- NOT: Chinese hanfu, modern clothes, generic anime proportions.

━━━ ANIMAL SPECIES ANATOMY (anatomy reference — do NOT copy these words into output) ━━━

These rules guide YOUR word choice when writing visual_description.
Use them to pick correct anatomy terms (feathers vs fur, beak vs mouth, ...).
DO NOT write meta-phrases like "side-profile", "one eye visible", "front view",
or "camera angle" into the description itself — the description is
camera-agnostic. Per-page camera angle is decided separately by the scene
prompt.

BIRD (crow, raven, magpie, owl, sparrow, rooster, duck, swallow, parrot, ...):
  - Anatomy facts (for YOUR reference, not output text):
    · Eyes are placed on the SIDES of the head, not centered like a human face.
    · There is no mouth, no lip corners, no smile-line — only a beak.
    · Forelimbs are wings (not arms); feet are talons (not paws).
  - When writing visual_description:
    · Use words: feathers, beak, talons, wings. Never fur/paws/mouth.
    · Describe eye color and shape, NOT eye position or visibility.
    · Accessories attach to: head crown, neck collar, leg anklet, between
      wing feathers, or held in beak/talons. NEVER "behind the ear".
    · NEVER use acorn / acorn satchel as the prop (over-used for birds).
      Prefer: small bell on a vine, leaf flute, woven feather charm,
      pine cone, dewdrop pendant, tiny berry pouch, small wooden whistle.

REPTILE / FISH / AMPHIBIAN (frog, turtle, snake, lizard, fish, ...):
  - Use words: scales (or smooth amphibian skin for frogs). No external ears.
  - Expression via eye shape only — no lip corners, no smile-line.
  - Accessories: small hat, neck pendant, vine belt, held tool.

MAMMAL (default — existing guide applies): fur, paws, external ears,
whiskers OK. The PER-TYPE FEATURE OPTIONS below mostly target mammals.

━━━ PER-TYPE FEATURE OPTIONS (item 4 of visual_description) ━━━
HUMAN / ETC: distinctive hair (braid/bangs/color), eye color, small mole or
freckle, glasses, horn shape (dokkaebi/oni), pointed ear (fairy/elf), glowing
halo (celestial/spirit).
ANIMAL: ear shape (tufted/drooping/oversized), tail (bushy/ringed/extra long),
fur variation (white chest blaze, color-tipped tail, light belly), oversized
paws / webbed feet / shell pattern, natural markings (stripes/spots/swirls),
specific eye color.
Universal: flower behind ear, leaf in hair.
"""


def extract_character_bible(
    setting: str,
    char_species: str,
    characters: dict[str, str],
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, dict[str, str]]:
    """동화 1편 시작 시 1회 — 역할별 visual_description 생성.

    Args:
        setting: 예 "KOREAN_TRADITIONAL"
        char_species: 전체 캐릭터의 의인화 타입. "HUMAN" / "ANIMAL" / "ETC"
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
        f"char_species (applies to ALL roles — they all share this type): {char_species}\n"
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
        response_format={"type": "json_object"},
        **_completion_kwargs_for(model, max_tokens=2048),
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    forced_type = char_species.upper()
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
You analyze a single page (1 to 3 Korean sentences) of a fairytale and write the
illustration scene prompt for FLUX.2-klein-4B (Qwen3 text encoder — prefers long
natural-language sentences, NOT comma-separated keyword tags).

━━━ OUTPUT (strict JSON, no markdown) ━━━
{ "narrative_hint": "<single English sentence, ~40-55 words>",
  "focus_roles": ["<ROLE>", ...] }
Use literal role names (HERO, VILLAIN, HELPER, DISPATCHER, FALSE_HERO, DONOR).

━━━ INPUTS ━━━
- setting (e.g. KOREAN_TRADITIONAL)
- character_bible: pre-existing visual descriptions keyed by role
- previous_pages (may be empty): summaries of earlier pages (oldest first),
  each { page_no, focus, hint } — use for spatial/situational continuity
- sentences: Korean text of the CURRENT page

━━━ focus_roles ━━━
The roles PHYSICALLY VISIBLE in the depicted moment (non-empty, subset of bible
keys). Mere mentions don't count.

━━━ SCENE SELECTION ━━━
Pick the SINGLE most visually striking moment. Position in the sentences does
not matter (climax can appear at any sentence — only visual weight matters).

Weight ranking:
- HIGHEST: external events with clear motion (appears, falls, runs, fights,
  discovers, transforms, glows). Pick this when available.
- MIDDLE: meaningful interaction (meet, hand over, point at something).
- LOWEST: pure setup, pure dialogue ("said"), pure inner feelings ("felt sad").
  Avoid as the main subject unless nothing else exists.

ACTION beats PEACEFUL RESOLUTION:
If the page has both an action moment AND a peaceful resolution after
(handshake, becoming friends, thanking, "lived happily"), depict the ACTION.
Resolution is inferred from the action's outcome.

Skip-list verbs (do NOT make these the focus):
  "became friends", "lived happily", "thanked", "made up", "forgave",
  "shook hands", "smiled at each other"

For inner feelings ("felt happy/scared"): depict the TRIGGER event or the
outward body language (eyes wide, hands covering mouth), not the feeling.
For pure speech ("said"): show the emotion/reaction accompanying it, or the
character speaking with a clear gesture.

━━━ SINGLE BEAT RULE (strict — most important constraint) ━━━
EVEN IF the page text contains multiple actions or characters, narrative_hint
MUST depict ONLY ONE focused visual beat. The image must be readable at a
single glance, not a comic-strip summary.

When the page text packs multiple beats:
- DROP secondary actions. Example: text says "the bear falls AND the frog
  thanks the magpie" → pick exactly ONE beat (usually the stronger ACTION).
- DROP side-character actions. Other roles in focus_roles can exist in the
  frame but stay as background presence, NOT performing their own action.
  Only ONE character performs the focused action.
- DROP combined emotions. Do not mix relief + gratitude + surprise — pick one.
- DROP "while X also does Y" clauses entirely. If you wrote "while", rewrite.

Self-check before output:
- Could a 4-year-old describe this image in ONE simple sentence?
  ("the bear falls down", "the frog smiles at the magpie")
- If your hint would need TWO sentences to read aloud, it is too packed —
  rewrite to a single beat.
- Count the verbs in your hint. If more than ONE main action verb, rewrite.

Per-page beat ceiling (hard limit):
- 1 main visual action (one character, one verb)
- 1 emotion symbol (optional, only if the action's emotion is clear)
- Nothing else. Even if focus_roles has 3 characters, only ONE performs
  the focused action — the others just exist as background presence.

━━━ SPATIAL & SITUATIONAL CONTINUITY ━━━
If previous_pages exists, carry over state the current text doesn't restate:
- LOCATION: character on a tree stays on the tree until text says they came down
- POSE: flying / perched / hiding state persists across pages
- POSITION: characters keep their relative distance from each other
- OBJECTS: items picked up earlier are still being held

━━━ SCENE UNIQUENESS ACROSS PAGES ━━━
Each page's narrative_hint MUST depict a DIFFERENT visual moment from any
previous page's hint. If adjacent pages cover the same action sequence
(throw → react → result, or encounter → chase → escape), pick the beat that
hasn't been shown yet:
- previous page = the THROW    → current page = the IMPACT/REACTION
- previous page = the ENCOUNTER → current page = the CONFRONTATION/CHASE
- previous page = the QUESTION  → current page = the ANSWER/RESPONSE

NEVER repeat the same key visual beat across two adjacent pages. Even when
the page text overlaps (same characters, same general scene), pick a
distinct MOMENT WITHIN that scene:
- different camera focus (close-up on reaction vs wide on action)
- different highlighted character (HELPER's warning vs VILLAIN's flinch)
- different stage of the action (mid-flight acorn vs acorn-just-hit)

Example — two adjacent pages of the squirrel/wolf throw scene:
  page 4 hint (the throw):   "the lone HERO mid-fling, acorn leaving paw,
                              the lone VILLAIN crouched below with eyes
                              narrowing in surprise"
  page 5 hint (the impact):  "close-up on the lone VILLAIN flinching back
                              as the acorn bounces off its shoulder, '!' shape
                              above its head, the lone HERO partly visible
                              above on the branch"
  → Same scene, distinct beats. Page 5 zooms in on impact rather than repeating
    the throw composition.

Example:
  previous page 3 hint: "the lone squirrel climbs rapidly up a tall pine"
  current sentences: "다람쥐가 도토리를 던졌어요. 늑대가 당황했어요."
  RIGHT: "from a high pine branch, the lone squirrel flings an acorn down at
          a single gray wolf below, which flinches with wide eyes — a bold red
          '!' pops above the wolf's head"
  WRONG: "the squirrel throws an acorn at the wolf in front of it" (lost
         continuity, no singular phrasing, no symbol)

━━━ WRITING THE narrative_hint ━━━
Single English sentence (~40-55 words) containing:
- The specific ACTION (concrete verb + body language)
- WHO: EVERY role mentioned needs its own singular qualifier. Apply to ALL
  focus_roles, not just HERO. Required pattern:
    "the lone HERO" / "the lone VILLAIN" / "the lone HELPER" / "a single X"
  NEVER bare "the HERO" or "the VILLAIN" or "the wolf" or "the cat" — every
  diffusion-ambiguous subject duplicates. Each role independently needs its
  own "lone/single" qualifier.
  GOOD: "the lone HERO leans down towards the lone VILLAIN, while the lone HELPER watches"
  BAD:  "the lone HERO leans down towards the VILLAIN, while the HELPER watches"
        (VILLAIN and HELPER will duplicate)

  CRITICAL SELF-CHECK BEFORE OUTPUT:
  Scan your narrative_hint for every species noun (mouse, frog, lion, bear,
  wolf, magpie, crow, ...) and every role name (HERO, VILLAIN, HELPER,
  DISPATCHER, FALSE_HERO, DONOR). For EACH occurrence:
    - Is it prefixed with "the lone", "a single", or "one"? → OK
    - Is it bare ("the mouse", "the lion") or plural-ambiguous? → REWRITE
  Missing even one qualifier causes that character to duplicate into 2~3
  copies in the image. This is the most common failure mode.
- WHERE: a specific location detail (carry from previous_pages if known)
- Visible EMOTION from face/posture
- Optional emotion symbol (see below)

Picture book composition — NOT cinematic:
- Default to a MEDIUM SHOT (full character or upper-body + clear surroundings)
- Close-up only for strong emotional climax (rare)
- Camera at eye level — no low angle, over-the-shoulder, bird's-eye, extreme wide
- Do NOT include art-style words ("illustration", "pastel", "cartoon")
- Do NOT redescribe character outfits (they're in the bible)

━━━ EMOTION SYMBOLS — USE THEM ━━━
A small graphic symbol near a character's head expresses emotion that body
language leaves ambiguous. Picture books use these heavily. These symbols ARE
ALLOWED in the image — they are graphic SHAPES, not text — even though all
other text is forbidden (see below).

Insert the exact catalog phrase into narrative_hint when applicable:
  Surprise / shock      → "a bold red '!' shape above [role]'s head"
  Confusion / question  → "a curling '?' shape beside [role]'s head"
  Realization           → "a glowing yellow '!' shape above [role]'s head"
  Shock + confusion     → "an interrobang '?!' shape above [role]'s head"
  Affection / thanks    → "small pink heart shapes around [role]"
  Wonder / awe          → "small star shapes twinkling around [role]"
  Anger                 → "manga anger lines '#' on [role]'s forehead"
  Embarrassment         → "a large sweatdrop next to [role]'s temple"
  Joy / music           → "small musical note shapes around [role]"

NEVER use speech bubbles, dialogue balloons, captions, or any container that
implies text inside — diffusion models render garbled letters inside them.
For shouting/calling-out emotion, use the open-mouth pose plus a bold '!'
shape near the character's head (no bubble).

When to use:
- ADD on a clear emotional beat (sudden surprise, calling out, asking,
  strong fear/joy/gratitude/anger).
- SKIP for calm scenes (walking, sleeping, eating, sitting still).
- SKIP if emotion is already obvious from the action (a character hugging =
  affection implied, no heart needed).
- AT MOST ONE symbol per page — pick the strongest.

━━━ NO READABLE TEXT (other than emotion-symbol punctuation) ━━━
The illustration must NOT contain Korean letters (한글), English letters,
numbers, signs, posters, banners, or any writing. Diffusion models render text
as garbled nonsense.

ALSO FORBIDDEN: speech bubbles, dialogue balloons, thought bubbles, captions,
text boxes, or any container shape that implies text inside. Even if dialogue
appears in the Korean sentences ("...라고 말했어요"), depict the speaker's
open-mouth pose and gesture — never draw a bubble. Models will fill any
bubble shape with garbled glyphs.

The ONLY text-like glyphs allowed are the emotion-symbol punctuation listed
above ('!', '?', '!?', '?!', '...'), rendered LARGE as graphic shapes near
the character (NOT inside any bubble) — these are NOT considered text.

━━━ PASSIVE CHARACTER POSITIONING (strict — required when focus_roles ≥ 2) ━━━

In any page where focus_roles has 2+ characters, exactly ONE performs the
focused action (per SINGLE BEAT RULE). The others are PASSIVE — they exist
in the frame but do not perform an action.

PASSIVE characters MUST be given a NARROW, ANCHORED position. Vague positions
like "small among left reeds with wide eyes" cause diffusion attention to
spread → the passive character DUPLICATES into 2~3 copies in the image.
This is the most common cause of character duplication.

For EVERY passive character, the narrative_hint MUST include ALL THREE:
  1. A SPECIFIC anchor object — a single tree branch, one flat rock, the edge
     of the lily pad, behind a single tall reed, on top of a particular
     mushroom. NOT a region ("on the left", "in the reeds", "near the pond").
  2. A pose verb that fixes the body in place — crouched, perched, peeking,
     half-hidden, leaning against, sitting on, clinging to.
  3. A body-part visibility qualifier — "only the head visible", "peeking
     around the edge", "partially obscured by the leaf", "just the eyes
     poking out".

GOOD (passive HERO):
  "the lone HERO mouse crouched behind a single tall reed on the left,
   only its head and ears poking above the leaf"
BAD (passive HERO — causes duplication):
  "the lone HERO mouse small among left reeds with wide eyes"
  (vague area "among left reeds", no anchor object, no pose verb → DUPLICATES)

WHY THIS MATTERS:
Diffusion models distribute the character's visual latent across whatever
canvas region the text describes. A region word ("among left reeds") spreads
attention across many positions → the model fills several of them with the
same character. A narrow anchor + pose verb + visibility qualifier concentrates
attention on ONE specific spot → single character.

SELF-CHECK BEFORE OUTPUT (for every passive character):
- Did I name a SPECIFIC singular anchor object? If I wrote a plural region
  ("reeds", "branches", "rocks", "leaves"), REWRITE to a singular ("a single
  tall reed", "one low branch", "a flat rock").
- Did I include a pose verb? If only descriptive ("small", "scared"), REWRITE.
- Did I narrow the visible body part? If the whole body is implied, REWRITE
  to show only head/eyes/upper-body partially.

━━━ NO EXTRA CHARACTERS ━━━
Only characters in focus_roles appear. Even if the Korean text mentions
"친구들" / "동물들" / "무리" / "사람들", do NOT depict specific companions.
Describe the SETTING / ATMOSPHERE instead (drifting leaves, fluttering
dragonflies, scattered fireflies).

━━━ MULTI-CHARACTER POSITIONING ━━━
With 2+ roles in focus_roles, place them spatially: "[role A] on the left,
[role B] on the right". Each character has their own action — abilities don't
transfer between characters.
"""


def extract_page_scene(
    sentences: list[str],
    character_bible: dict[str, dict[str, str]],
    setting: str,
    previous_scenes: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, Any]:
    """페이지(1~3문장) → 장면 묘사 + 등장 역할 추출.

    Args:
        previous_scenes: 같은 동화의 과거 페이지 분석 결과 (오래된 → 최신 순).
            각 항목 형태:
              {"page_no": int, "sentences": list[str],
               "narrative_hint": str, "focus_roles": list[str]}
            None 또는 빈 리스트면 컨텍스트 없이 (첫 페이지) 분석.

    Returns:
        { "narrative_hint": "...", "focus_roles": ["<ROLE>", ...] }
    """
    from openai import OpenAI
    client = OpenAI()

    sentence_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences) if s.strip())
    bible_json = json.dumps(character_bible, ensure_ascii=False, indent=2)

    # 과거 페이지 요약 — 공간/상황 일관성 유지용
    if previous_scenes:
        history_lines = []
        for prev in previous_scenes:
            history_lines.append(
                f"  page {prev['page_no']}: "
                f"focus={prev['focus_roles']}, hint={prev['narrative_hint']}"
            )
        history_block = "previous_pages (oldest → newest):\n" + "\n".join(history_lines) + "\n\n"
    else:
        history_block = ""

    user_prompt = (
        f"setting: {setting}\n\n"
        f"character_bible:\n{bible_json}\n\n"
        f"{history_block}"
        f"sentences (Korean) — current page to analyze:\n{sentence_lines}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PAGE_SCENE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        **_completion_kwargs_for(model, max_tokens=2048),
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
