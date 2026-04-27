"""Klein 전용 스토리 파이프라인.
기존 story_pipeline.py를 상속하되 프롬프트 빌더를 Klein(Qwen3, 40960토큰)에 맞게 오버라이드.
"""

from __future__ import annotations

from api.pipeline.story_pipeline import (
    StoryInput,
    WorldProfile,
    StoryCharacter,
    SceneSpec,
    ScenePlan,
    StoryPlan,
    build_story_plan as _base_build_story_plan,
    build_world_profile,
    build_character_bible,
    build_scene_spec,
    normalize_story_input,
    run_quality_gate,
)

STYLE_PREFIX = (
    "children's book illustration, soft cell shading with smooth color gradients, "
    "warm teal and golden tones, glowing light particles scattered in the background, "
    "painterly digital art style, no hard outlines, lush soft background. "
)


def build_prompt_klein(
    scene_spec: SceneSpec,
    world: WorldProfile,
    character_bible: dict[str, str],
    scene_chars: list[StoryCharacter],
) -> str:
    """FLUX.2-klein-4B Qwen3 인코더용 자연어 프롬프트.
    토큰 한도 40,960 → 상세하고 자연스러운 문장으로 작성.
    """
    # 1. 스타일
    style = STYLE_PREFIX

    # 2. 캐릭터 묘사
    char_parts = []
    for i, ch in enumerate(scene_chars):
        if ch.visual_hint and len(ch.visual_hint.split()) >= 3:
            desc = ch.visual_hint
        elif character_bible.get(ch.id):
            desc = character_bible[ch.id]
        else:
            if ch.species:
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

    # 3. 장면 행동/상황
    action = f"Scene: {scene_spec.narrative_hint}." if scene_spec.narrative_hint else ""

    # 4. 배경
    background = f"Background: {world.positive_hint}."

    # 5. 구도
    composition = (
        "Wide establishing shot showing both the character and the background clearly. "
        + scene_spec.staging_hint
    )

    parts = [style, char_sentence]
    if action:
        parts.append(action)
    parts.append(background)
    parts.append(composition)

    return " ".join(parts)


def build_scene_plans_klein(
    story_input: StoryInput,
    world: WorldProfile,
    character_bible: dict[str, str],
) -> list[ScenePlan]:
    """Klein 전용 프롬프트 빌더로 장면 계획 생성."""
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

        prompt = build_prompt_klein(
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


def build_story_plan_klein(story_data, max_scenes: int = 10) -> StoryPlan:
    """Klein 전용 스토리 플랜 빌더."""
    story_input = normalize_story_input(story_data, max_scenes=max_scenes)
    cleaned_lines, report = run_quality_gate(story_input.lines, max_scenes=max_scenes)
    story_input.lines = cleaned_lines
    world = build_world_profile(story_input.theme)
    character_bible = build_character_bible(story_input.characters)
    scenes = build_scene_plans_klein(story_input, world, character_bible)

    return StoryPlan(
        story_input=story_input,
        quality_gate=report,
        world=world,
        character_bible=character_bible,
        scenes=scenes,
    )
