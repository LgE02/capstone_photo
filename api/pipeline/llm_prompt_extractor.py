"""GPT-4o를 사용해 동화 텍스트에서 삽화 프롬프트를 추출한다.

기존 story_analyzer.py (키워드 매핑)를 대체한다.
동화 전체를 1회 API 호출로 분석하여:
  - 캐릭터 정보 (종, 직업, 외형)
  - 세계관/테마
  - 장면별 영어 이미지 프롬프트 (자연어 문장)
를 일괄 반환한다.

사용 흐름:
    from api.pipeline.llm_prompt_extractor import extract_story_prompts
    result = extract_story_prompts("옛날에 토끼가 산에서 살았습니다...")
    # result는 build_story_plan()에 전달 가능한 dict
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
The output will be used with a FLUX T5 text encoder that understands natural language sentences.

RULES:
- Output MUST be valid JSON (no markdown, no explanation)
- All visual descriptions must be in English
- scene_prompt must be a NATURAL LANGUAGE SENTENCE (max 30 words) describing the visual scene
- scene_prompt MUST include: character action/emotion, location, and mood
- scene_prompt MUST clearly describe character EMOTIONS and EXPRESSIONS (e.g., "looking frightened", "smiling joyfully", "gazing curiously", "deep in thought")
- scene_prompt MUST specify the location/environment clearly (e.g., "in a deep forest clearing", "at a Korean hanok village")
- Do NOT use keyword-style prompts. Write full, descriptive English sentences.
- Do NOT include art style words (no "illustration", "pastel", "cartoon" etc.)
- theme must be one of: KOREAN_TRADITIONAL, FOREST_NATURE, FANTASY_WORLD, EUROPEAN_MEDIEVAL, UNDERWATER, SKY_HEAVEN, MODERN_FANTASY, MIXED

CHARACTER VISUAL DESCRIPTION - CRITICAL:
You must design each character with SPECIFIC, FIXED visual features that make them
instantly recognizable across all pages. This is the MOST IMPORTANT part.

For each character, visual_description MUST include ALL of these:
1. Body color/fur color (e.g., "brown fur", "green shell", "pale blue skin")
2. ONE distinctive accessory or clothing item (e.g., "wearing a small red vest", "golden crown", "blue scarf")
3. ONE unique physical feature (e.g., "big floppy ears", "round shell with star pattern", "long white beard")
4. Size descriptor (e.g., "small", "large", "tiny")

Example visual_descriptions:
- "small brown rabbit with big floppy ears, wearing a red vest with gold buttons"
- "green turtle with a round shell decorated with blue swirl patterns, wearing a tiny blue hat"
- "large sea dragon king with pale blue scales, long white whiskers, golden coral crown on head"
- "old man with long white beard, wearing dark blue hanbok with silver embroidery"

BAD visual_descriptions (too vague - DO NOT do this):
- "a cute rabbit with big eyes" (no color, no clothing, no unique feature)
- "a wise turtle" (no visual info at all)
- "a majestic dragon" (no specific details)

For ANIMAL characters:
- They are ANIMALS, not humans. Emphasize their animal body (fur, shell, scales, tail, paws)
- If they wear clothes, specify the clothing ON the animal body
- NEVER describe an animal character as looking human

MULTI-CHARACTER SPATIAL COMPOSITION (2-3 characters max):
- When 2+ characters appear together, ALWAYS specify their positions in scene_prompt:
  "character A on left side, character B on right side"
- This prevents the image generator from merging characters into one

SCENE_PROMPT EXAMPLES (natural language, with emotion):
GOOD:
- "A small white rabbit sits alone in a forest clearing, looking worried and deep in thought"
- "The woodcutter stands at the edge of the forest, shouting excitedly with arms raised"
- "The rabbit on the left looks up at a mystical fairy on the right, eyes wide with wonder"
- "A girl in blue hanbok kneels by a glowing crystal ball in a dark forest, looking amazed"

BAD (keyword-style - DO NOT do this):
- "white rabbit, forest, thinking pose, NOT human"
- "woodcutter shouting, forest edge, animal body"

Korean traditional CHARACTER translation rules:
- 선녀 → "a beautiful celestial woman in flowing white Korean cheonui (heavenly robe), long black hair pinned with a jade hairpin, NO wings, NO fairy wings, human form" (Korean seonnyeo is NOT a Western fairy. She is a graceful human woman from heaven.)
- 나무꾼 → "a sturdy Joseon-era Korean woodcutter man wearing rough beige hemp jeogori and baji (traditional work clothes), a worn straw hat (satgat), barefoot or straw sandals, carrying a large wooden-handled axe" (NOT modern clothes. Joseon period farmer-style.)
- 도깨비 → "a Korean dokkaebi goblin with wild hair and a small horn on its head, wearing a tiger-patterned loincloth, holding a magic spiked club (bangmangi), mischievous grin"
- 용왕 → "Korean Dragon King with pale blue scales, long white whiskers, golden coral crown, wearing ornate blue sea-king robes"
- 산신령 → "old Korean mountain spirit with a long white beard, white eyebrows, wearing white hanbok with a staff, serene expression"
- 도사 → "Joseon Taoist sage with white topknot hair, long white hanbok robe, holding a wooden staff"

Korean traditional element translation rules (ONLY apply when theme is KOREAN_TRADITIONAL):
- 초가집 → "Korean straw-thatched roof farmhouse with wooden walls and dirt floor"
- 기와집 → "Korean hanok house with dark curved tiled roof and wooden pillars"
- 한옥 → "Korean hanok house with dancheong painted eaves and ondol floor"
- 궁궐 → "Korean Joseon royal palace with dancheong painted hanok buildings and stone steps"
- 마을 → "Korean traditional village with hanok houses and dirt paths"
- 장터 → "Korean traditional open-air market with cloth canopies and wooden stalls"
- 임금님 → "Korean king wearing red Joseon gonryongpo dragon robe and ikseongwan black crown"
- 도령 → "young Korean nobleman in white jeogori and black baji, wearing black gat hat"
- ALWAYS use "Korean hanok" for buildings, NEVER just "palace" or "temple"
- ALWAYS use "hanbok" for clothing, NEVER "robe" or "dress" or "gown"
- NEVER use generic Asian/Chinese terms - always prefix with "Korean" or "Joseon"

DYNAMIC SCENE COMPOSITION RULES (CRITICAL for lively illustrations):
- Every scene_prompt MUST vary the camera angle and composition:
  * Close-up: "close-up shot of [character]'s face showing [emotion]"
  * Wide shot: "wide shot of [character] small in a large [environment]"
  * Action shot: "dynamic shot of [character] [action verb]-ing with motion"
  * Over-the-shoulder: "viewed from behind as [character] faces [something]"
  * Low angle: "low angle shot looking up at [character]"
- NEVER use the same composition twice in a row

EXAGGERATED CARTOON EXPRESSION RULES:
- Use exaggerated cartoon-style emotions for lively illustrations:
  * Surprised: "eyes wide as saucers, jaw dropped open, sweat drop on forehead"
  * Scared: "trembling, hiding face in paws/hands, big teary eyes"
  * Happy: "jumping with joy, arms wide open, sparkling eyes"
  * Confused: "tilting head sideways, question mark floating above head"
  * Excited: "bouncing with energy, fists pumped, shining eyes"
  * Thinking: "chin resting on paw/hand, thought bubble floating above"
- When a character speaks, add: "speech bubble with exclamation mark"
- When a character is confused or wondering, add: "question mark floating above head"
- When something magical happens, add: "sparkles and glowing light effects around"

JSON schema:
{
  "theme": "KOREAN_TRADITIONAL",
  "characters": [
    {
      "id": "unique_name",
      "type": "human|animal|other",
      "species": null or "rabbit|tiger|frog|...",
      "job": null or "king|princess|farmer|scholar|woodcutter|fairy|...",
      "visual_description": "DETAILED visual description with color, accessory, unique feature (see rules above)"
    }
  ],
  "scenes": [
    {
      "line": "original Korean text",
      "scene_prompt": "Natural language English sentence describing the scene with character emotions and actions",
      "focus_characters": ["character_id_1", "character_id_2"]
    }
  ]
}

IMPORTANT:
- Generate exactly ONE scene per input line. If the story has 10 lines, output 10 scenes.
- focus_characters: ONLY characters who are PHYSICALLY VISIBLE in the illustration.
  Do NOT include characters who are merely MENTIONED or TALKED ABOUT.
- scene_prompt MUST clearly state the environment (underwater, on land, in forest, etc.)
- scene_prompt MUST include character EMOTION or EXPRESSION in every scene"""

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
        story_text: 한국어 동화 전체 텍스트 (str) 또는
                    DB에서 페이지별로 가져온 줄 목록 (list[str])
        model: 사용할 OpenAI 모델
        max_scenes: 최대 장면 수

    Returns:
        build_story_plan()에 전달 가능한 dict
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
        max_tokens=4000,
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
