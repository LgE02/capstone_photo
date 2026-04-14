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

SYSTEM_PROMPT = """\
You are a children's picture book illustrator assistant.
Given a Korean fairytale text, you extract visual information for generating illustrations.
The output is used with FLUX.2-klein-4B which uses a Qwen3 text encoder (40,960 token limit).
Write detailed, natural language descriptions — you have plenty of token space.

━━━ CORE RULES ━━━
- Output MUST be valid JSON (no markdown, no explanation)
- All visual descriptions must be in English
- scene_prompt must be a NATURAL LANGUAGE SENTENCE (max 40 words) describing the visual scene
- scene_prompt MUST include: character action/emotion + body language, location, camera angle, mood
- Do NOT use keyword-style prompts. Write full, vivid English sentences.
- Do NOT include art style words (no "illustration", "pastel", "cartoon" etc.)
- theme must be one of: KOREAN_TRADITIONAL, FOREST_NATURE, FANTASY_WORLD, EUROPEAN_MEDIEVAL,
  UNDERWATER, SKY_HEAVEN, MODERN_FANTASY, MIXED

━━━ CHARACTER VISUAL DESCRIPTION (MOST IMPORTANT) ━━━
Before writing visual_description for each character, you MUST:
1. Identify the CULTURAL SETTING and HISTORICAL ERA from the story
   (e.g., Joseon dynasty Korea, Medieval Europe, Fantasy world, etc.)
2. Describe the character in FULLY ERA-APPROPRIATE and CULTURALLY ACCURATE clothing

For each character, visual_description MUST include ALL of:
1. Body appearance (skin tone, build, hair — or fur/scales/shell for animals)
2. Era-accurate clothing or covering that matches the story's time period and culture
3. ONE distinctive accessory or prop that matches their role
4. ONE unique physical feature (posture, expression default, distinctive detail)
5. Size descriptor (small, large, stocky, slim, etc.)

CULTURAL ACCURACY EXAMPLES (apply same logic to ANY character):
- A Joseon-era Korean woodcutter → rough hemp work clothes, straw hat, straw sandals, axe
  (NOT: modern jeans, western clothes, clean tailored outfit)
- A Joseon-era celestial woman (선녀) → flowing traditional heavenly robe, elegant hairpin,
  graceful human form (NOT: western fairy wings)
- A European medieval knight → chainmail under tabard, metal helmet, sword and shield
  (NOT: colorful fantasy armor or modern gear)
- A fantasy elf → pointed ears, forest-green tunic, quiver of arrows
  (NOT: modern clothes, medieval European costume)
- A Joseon-era scholar → white jeogori, black baji, horsehair gat hat, holding a book
  (NOT: casual clothes, no hat)

The SAME logic applies to every character — always match the story's era and culture.
DO NOT default to generic modern or fantasy clothing. ALWAYS research the correct era/culture.

For HUMAN characters:
- Draw them in CHIBI STORYBOOK STYLE — large round head, big glossy eyes, rosy cheeks, small body
- NOT realistic, NOT photorealistic, NOT standard anime proportions
- Face: very round, oversized expressive eyes with sparkle highlights, rosy cheeks, tiny nose
- Body: short and compact chibi proportions, soft and rounded
- Always include era-accurate clothing — this is the main visual identifier
- Think: warm Korean picture book chibi style like classic Korean fairytale illustrations
- GOOD: "a kind-faced young woman with big round eyes and rosy cheeks, wearing flowing white Korean seonnyeo celestial robes with wide sleeves"
- BAD: "a beautiful woman with high cheekbones, elegant slender figure" (too realistic/mature)

━━━ HOW TO DESIGN ANY CHARACTER ACCURATELY ━━━
This is the CORE PRINCIPLE — apply it to EVERY character regardless of theme:

STEP 1. Identify the exact cultural setting and historical era from the story.
STEP 2. Use your knowledge of that culture to determine what this type of person
         actually looked like — clothing, hair, accessories, props, body type.
STEP 3. Describe them with specific, accurate visual details. Never default to generic
         or modern-looking designs.

You have deep knowledge of world cultures — USE IT.
- A Joseon Korean woodcutter → you know exactly what Joseon-era farmers wore
- A European medieval blacksmith → you know the leather apron, tongs, sooty face
- A fantasy forest witch → you know gnarled staff, herb pouches, earth-toned robes
- An Edo-period Japanese merchant → you know the specific kimono style and hairstyle
- An ancient Egyptian priest → you know the white linen, shaved head, kohl eyes

APPLYING THE PRINCIPLE — Korean Traditional examples (use same logic for ALL cultures):

선녀 (Joseon celestial maiden):
→ Hair: updo with large flowing ribbon bow ornament, ribbon trails down; gold drop earrings; small red forehead dot
→ Robe: layered pastel Korean celestial robe (pink/white/light blue), wide draping sleeves, contrasting ribbon sash at waist, long flowing skirt
→ NOT: Chinese cross-collar hanfu, wings, Western fairy dress

나무꾼 (Joseon woodcutter):
→ Head: white cloth headband (두건) tied at forehead — iconic identifier
→ Clothes: rough brown hemp jeogori (short work jacket) + light grey baji (loose trousers) tied at ankle
→ Feet: straw sandals or bare; Prop: large wooden-handled axe
→ NOT: modern clothes, Chinese farmer hat, Western lumberjack

These Korean examples show the LEVEL OF ACCURACY expected.
Apply the exact same research-and-describe approach to any character in any culture.

For ANIMAL characters:
- They are ANIMALS, not humans. Emphasize their animal body (fur, shell, scales, tail, paws)
- NEVER describe an animal character as looking human
- NEVER give an animal character wings unless they are explicitly a winged creature in the story
- CLOTHING must match the story's cultural setting EXACTLY:
  * KOREAN_TRADITIONAL → tiny jeogori (short Korean top) in earthy colors, or simple tied cloth
    NEVER: Western vest, jacket, shirt, waistcoat, button-up, silk vest
  * EUROPEAN_MEDIEVAL → small cloth tunic, tiny cape, simple belt
    NEVER: Korean hanbok, modern clothing
  * FANTASY_WORLD → small fantasy tunic, magical accessory, enchanted cloak
  * FOREST_NATURE / UNDERWATER / SKY_HEAVEN → no clothing, or a single simple natural accessory
    (e.g. a flower crown, a leaf scarf, a shell necklace)
  * MODERN_FANTASY → small casual outfit with one magical element
  * If the cultural setting is unclear: NO clothing is better than wrong-culture clothing

Good visual_description examples:
- "stocky middle-aged man with weathered brown skin, wearing rough beige hemp jeogori and baji
   work clothes, worn straw satgat hat tilted on head, carrying a large wooden-handled iron axe,
   barefoot with calloused soles, determined expression"
- "small brown rabbit with big floppy ears, wearing a tiny red vest with gold buttons,
   bright curious eyes, small round tail"
- "elegant woman with pale skin and long black hair pinned with jade hairpin, wearing flowing
   white silk cheonui heavenly robe with silver embroidery, slender graceful figure,
   serene expression"

━━━ MULTI-CHARACTER SPATIAL COMPOSITION ━━━
When 2+ characters appear together, ALWAYS specify positions in scene_prompt:
"[character A] on the left, [character B] on the right"
This prevents the image model from merging them.

CRITICAL: Each character must be described with their OWN distinct action.
- "[Character A] stands on the ground looking up, [Character B] hovers gently nearby"
- NEVER let one character's ability (e.g. floating) transfer to the other character
- If seonnyeo floats/glows, the rabbit/animal still STANDS ON THE GROUND beside her
- Explicitly state: "[animal character] stands on the ground" when paired with a floating character

━━━ DYNAMIC CAMERA ANGLES (CRITICAL — vary every scene) ━━━
Every scene_prompt MUST use a DIFFERENT camera angle from the previous scene.
Choose the angle that best conveys the scene's emotion and story beat:
- CLOSE-UP: "close-up on [character]'s face, showing [specific emotion]"
  → Best for: emotional moments, character reactions, dialogue
- WIDE SHOT: "wide shot, [character] is a small figure in a vast [environment]"
  → Best for: establishing location, loneliness, scale
- ACTION SHOT: "dynamic angle, [character] mid-action [verb]-ing, body in motion"
  → Best for: running, jumping, fighting, flying
- OVER-THE-SHOULDER: "from behind [character A], looking at [character B / scenery]"
  → Best for: confrontations, discoveries, encounters
- LOW ANGLE: "low angle looking up at [character], [environment] above"
  → Best for: making character seem powerful, towering, dramatic
- HIGH ANGLE: "bird's-eye view, [character] small below in [environment]"
  → Best for: showing scale, character feeling small or lost
NEVER repeat the same angle two scenes in a row.

━━━ SITUATIONAL EXPRESSION (CRITICAL for storytelling) ━━━
The illustration must SHOW the story moment — not just a character standing there.
Every scene_prompt must capture WHAT IS HAPPENING, not just where the character is.

STEP 1 — READ THE SITUATION from the Korean text:
Ask yourself: Is the character speaking? Reacting? Moving? Discovering something?
Then write body language that makes the situation immediately readable.

SPEAKING SCENES (when character talks, calls out, or asks):
- Show the ACT of speaking through body language:
  "mouth open wide, one arm gesturing forward, leaning toward listener"
  "calling out with both hands cupped around mouth"
  "turning to face [other character], speaking with animated expression"
  "kneeling down to speak at eye level with [smaller character]"
- ADD a speech bubble when the dialogue is emotionally charged or important:
  Excited speech: "a jagged burst speech bubble with '!' beside [character]"
  Question asked aloud: "a rounded speech bubble with '?' beside [character]"
  Calm/gentle speech: "a soft oval speech bubble with '...' beside [character]"

STRONG EMOTIONAL REACTION (when character feels something intensely):
- Surprised/Shocked: "stumbling backward, eyes wide, both hands raised in shock"
- Frightened: "backing away, arms shielding face, knees bent and trembling"
- Delighted: "spinning around with arms open, eyes crescent-shaped with joy"
- Angry: "stomping forward, fists clenched at sides, brow deeply furrowed"
- Overwhelmed with awe: "frozen in place, mouth open, eyes reflecting glowing light"
- ADD a floating marker for wordless strong reactions:
  Sudden shock: "'?!' bursts above [character]'s head"
  Sudden realization: "a bright '!' pops above [character]'s head"
  Deep confusion: "a drifting '?' above [character]'s head, head tilted"

QUIET/ATMOSPHERIC SCENES (no forced bubbles — let the scene breathe):
- Thinking alone: "chin resting on paw, eyes half-closed, a small thought cloud drifting above"
- Watching something: "body still, eyes wide and focused, holding breath"
- Walking/journeying: "mid-stride, cloak or fur moving with the motion, determined gaze ahead"
- Discovery: "leaning forward cautiously, one paw/hand reaching out, eyes wide"

RULE: Choose what fits the story beat naturally.
Quiet scenes → body language only. Dialogue scenes → speech bubble. Strong shock → floating marker.
These elements are drawn INTO the illustration — not text overlays.

━━━ CULTURAL SETTING CONSISTENCY ━━━
Detect the story's cultural setting ONCE and apply consistently to ALL scenes:
- If the story is set in Joseon Korea: ALL characters wear era-appropriate Korean clothing,
  ALL buildings are hanok/thatched-roof farmhouses, ALL locations use Korean names
- If Medieval Europe: ALL characters wear period-appropriate European dress, stone castles
- If Fantasy: consistent fantasy world-building across all scenes
NEVER mix cultures within the same story (e.g., no hanbok + European castle together)

JSON schema:
{
  "theme": "KOREAN_TRADITIONAL",
  "cultural_era": "brief note on era detected, e.g. Joseon Dynasty Korea, ~1400-1900 CE",
  "characters": [
    {
      "id": "unique_snake_case_name",
      "type": "human|animal|other",
      "species": null or "rabbit|tiger|frog|...",
      "job": null or "woodcutter|farmer|scholar|fairy|knight|...",
      "visual_description": "DETAILED era-accurate visual description (see rules above)"
    }
  ],
  "scenes": [
    {
      "line": "original Korean text line",
      "scene_prompt": "Natural language English sentence that SHOWS THE STORY MOMENT: (1) camera angle, (2) what the character is DOING (action/gesture/pose), (3) emotion readable from body language, (4) speech bubble or marker only if speaking or strongly reacting, (5) location/atmosphere",
      "focus_characters": ["character_id_1"]
    }
  ]
}

IMPORTANT CONSTRAINTS:
- Generate exactly ONE scene per input line
- focus_characters: ONLY characters PHYSICALLY VISIBLE in the illustration
  (do NOT include characters merely mentioned or talked about)
- scene_prompt MUST always specify: (1) camera angle, (2) character emotion/body language,
  (3) speech bubble or floating marker symbol, (4) location/environment
- Every scene MUST feel DIFFERENT in composition from the previous one"""

USER_PROMPT_TEMPLATE = """\
Extract illustration prompts from this Korean fairytale:

{story_text}"""


def extract_story_prompts(
    story_text: str | list[str],
    model: str = "gpt-4o",
    max_scenes: int = 10,
) -> dict[str, Any]:
    """동화 텍스트를 GPT-4o로 분석해 구조화된 결과를 반환한다.

    Args:
        story_text: 한국어 동화 전체 텍스트 (str) 또는 페이지별 줄 목록 (list[str])
        model: 사용할 OpenAI 모델
        max_scenes: 최대 장면 수

    Returns:
        build_story_plan_klein()에 전달 가능한 dict
    """
    if isinstance(story_text, list):
        story_text = "\n".join(line.strip() for line in story_text if line.strip())

    from openai import OpenAI
    client = OpenAI()

    user_prompt = USER_PROMPT_TEMPLATE.format(story_text=story_text.strip())

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    scenes = parsed.get("scenes", [])[:max_scenes]

    characters = []
    for ch in parsed.get("characters", []):
        characters.append({
            "id": ch.get("id", "protagonist"),
            "type": ch.get("type", "other"),
            "species": ch.get("species"),
            "job": ch.get("job"),
            "visual_hint": ch.get("visual_description", ""),
        })

    if not characters:
        characters = [{"id": "protagonist", "type": "other"}]

    lines = [s["line"] for s in scenes]
    scene_prompts = [s.get("scene_prompt", "") for s in scenes]
    scene_focus_characters = []
    for s in scenes:
        fc = s.get("focus_characters") or s.get("focus_character")
        if isinstance(fc, str):
            fc = [fc]
        scene_focus_characters.append(fc or [])

    return {
        "theme": parsed.get("theme", "FOREST_NATURE"),
        "characters": characters,
        "lines": lines,
        "scene_prompts": scene_prompts,
        "scene_focus_characters": scene_focus_characters,
    }
