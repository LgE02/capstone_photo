"""GPT-4o-mini를 사용해 동화 텍스트에서 삽화 프롬프트를 추출한다.

기존 story_analyzer.py (키워드 매핑)를 대체한다.
동화 전체를 1회 API 호출로 분석하여:
  - 캐릭터 정보 (종, 직업, 외형)
  - 세계관/테마
  - 장면별 영어 이미지 프롬프트
를 일괄 반환한다.

사용 흐름:
    from runtime_pipeline.llm_prompt_extractor import extract_story_prompts
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

RULES:
- Output MUST be valid JSON (no markdown, no explanation)
- All visual descriptions must be in English
- Each scene_prompt must be a SHORT English phrase (max 15 words) describing the visual scene
- scene_prompt should focus on: character action, location, key objects, mood/weather
- scene_prompt MUST specify the location/environment clearly (e.g., "underwater coral palace", "grassy hillside", "dark forest path")
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
- They are ANIMALS, not humans. Always emphasize their animal body (fur, shell, scales, tail, paws)
- If they wear clothes, specify the clothing ON the animal body
- NEVER describe an animal character as looking human

CRITICAL - Korean traditional element translation rules (SDXL confuses Korean with Chinese):
- 초가집 → "Korean straw-thatched roof farmhouse with wooden walls" (NOT "thatched cottage")
- 기와집 → "Korean hanok house with dark curved tiled roof and wooden pillars"
- 한옥 → "Korean hanok house with dancheong painted eaves and ondol floor"
- 궁궐 → "Korean Joseon royal palace, hanok style with dark tiled curved roof"
- 마을 → "Korean traditional village with hanok houses"
- 장터 → "Korean traditional open-air market with cloth canopies"
- 임금님 → "Korean king wearing Joseon gonryongpo dragon robe and ikseongwan crown"
- 도령 → "young Korean nobleman in white hanbok with black gat hat"
- ALWAYS use "Korean hanok" for buildings, NEVER just "palace" or "temple"
- ALWAYS use "hanbok" for clothing, NEVER "robe" or "dress" or "gown"
- NEVER use generic Asian terms - always prefix with "Korean" or "Joseon"
- Include "dancheong painted" for palace/temple details (unique Korean feature)

JSON schema:
{
  "theme": "KOREAN_TRADITIONAL",
  "characters": [
    {
      "id": "unique_name",
      "type": "human|animal|other",
      "species": null or "rabbit|tiger|frog|...",
      "job": null or "king|princess|farmer|scholar|...",
      "visual_description": "DETAILED visual description with color, accessory, unique feature (see rules above)"
    }
  ],
  "scenes": [
    {
      "line": "original Korean text",
      "scene_prompt": "English visual description with clear location AND character action",
      "focus_characters": ["character_id_1", "character_id_2"]
    }
  ]
}

IMPORTANT:
- Generate exactly ONE scene per input line. If the story has 10 lines, output 10 scenes.
- focus_characters: ONLY characters who are PHYSICALLY VISIBLE in the illustration.
  Do NOT include characters who are merely MENTIONED or TALKED ABOUT.
  Example: "의원이 토끼의 간이 필요하다고 말했다" → the rabbit is only talked about, NOT visible.
  So focus_characters should be ["dragon_king"] or ["dragon_king", "doctor"], NOT ["rabbit"].
  Example: "자라가 토끼를 데려오겠다고 했다" → only the turtle is visible, rabbit is NOT present.
  So focus_characters should be ["turtle"], NOT ["turtle", "rabbit"].
- scene_prompt MUST clearly state the environment (underwater, on land, in forest, etc.)"""

USER_PROMPT_TEMPLATE = """\
Extract illustration prompts from this Korean fairytale:

{story_text}"""


def extract_story_prompts(
    story_text: str | list[str],
    model: str = "gpt-4o-mini",
    max_scenes: int = 10,
) -> dict[str, Any]:
    """동화 텍스트를 GPT-4o-mini로 분석해 구조화된 결과를 반환한다.

    Args:
        story_text: 한국어 동화 전체 텍스트 (str) 또는
                    DB에서 페이지별로 가져온 줄 목록 (list[str])
        model: 사용할 OpenAI 모델
        max_scenes: 최대 장면 수

    Returns:
        build_story_plan()에 전달 가능한 dict:
        {
            "theme": str,
            "characters": [...],
            "lines": [str, ...],
            "scene_prompts": [str, ...],  # 장면별 영어 프롬프트
        }
    """
    # DB에서 줄 목록으로 들어올 경우 하나의 텍스트로 합친다
    # (1회 API 호출로 전체 맥락을 파악해야 캐릭터 일관성 유지 가능)
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

    # 장면 수 제한
    scenes = parsed.get("scenes", [])[:max_scenes]

    # build_story_plan()이 기대하는 형식으로 변환
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
    # 장면별 등장 캐릭터 ID 목록 (LLM이 명시적으로 반환)
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
