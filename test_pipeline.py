"""
동화 삽화 파이프라인 테스트 스크립트

실행 방법:
    cd fairytale_lora
    python test_pipeline.py
    python test_pipeline.py --story-a   # Story A: 동물 주인공
    python test_pipeline.py --story-b   # Story B: 사람 주인공
    python test_pipeline.py --seed 42
    python test_pipeline.py --low-memory
    python test_pipeline.py --keep-loaded
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

# ── 테스트 동화 텍스트 ────────────────────────────────────────────────────────
DEFAULT_STORY = """옛날에 개구리 대군님이 한옥 마을에서 살았습니다.
어느 날 대군님은 장터에서 편지를 받았습니다.
비가 내리는 궁궐 앞에서 기뻐하며 춤을 추었습니다.
임금님께서 등불을 들고 기다리고 있었습니다."""

# ── Story A: 동물 주인공 (토끼 + 나무꾼 + 선녀) ────────────────────────────
STORY_A = """옛날 옛날 한 옛날에, 어느 작은 마을에 용감한 마음을 가진 토끼 한 마리가 살았답니다.
토끼는 용감한 마음을 가졌지만, 평소에는 숲을 뛰어다니며 친구들과 놀기만 했답니다.
하루는 토끼가 숲속에서 친구들과 신나게 놀고 있을 때, 나무꾼이 나타나서 "이게 웬일이에요! 오늘은 특별한 일이 일어날 것 같아요!"라고 외쳤답니다.
그런데 이를 어쩌지요! 나무꾼은 숲속에서 토끼에게 큰일이 날 것 같다는 이야기를 하면서, 믿을 수 없는 소문을 전했답니다.
아이고! 나무꾼의 이야기를 듣고 토끼는 마음속으로 두려운 생각이 들었지만, 용감한 마음을 가진 만큼 친구들에게는 아무 말도 하지 않았답니다.
그날 오후, 토끼는 나무꾼의 말을 잊지 않으려 애쓰며 숲속에서 혼자 사색에 잠겼답니다.
그런데 갑자기 토끼의 귀에 바람 소리가 들려왔고, 그 소리가 마치 누군가의 속삭임처럼 느껴졌답니다.
그런데 갑자기 바람 소리 너머에서 누군가의 모습이 보였고, 그건 바로 숲속의 신비로운 선녀였답니다!
선녀는 토끼에게 "너는 용감한 마음을 가지고 있구나, 그래서 나는 너에게 특별한 일을 부탁하고 싶어!"라고 말했답니다.
토끼는 선녀의 부탁을 듣고 "어떤 특별한 일을 부탁하실 건가요?"라고 궁금해했지요.
선녀는 토끼에게 "네가 용감하게 나무꾼의 이야기를 전해 주면, 모든 것이 잘 해결될 거란다!"라고 강조했답니다.
토끼는 선녀의 말에 용기를 내어 "그럼 제가 나무꾼에게 전해볼게요!"라고 결심했답니다.
토끼는 나무꾼에게 다가가서 "나무꾼 아저씨, 선녀가 저에게 특별한 일을 부탁하셨어요!"라고 용감하게 이야기했답니다.
그러자 나무꾼은 놀란 눈으로 토끼를 바라보며 "정말로 선녀가 그런 말을 했단 말이냐?"라고 물었답니다.
나무꾼은 호기심 가득한 목소리로 "그렇다면 선녀님께서 어떤 특별한 일인지 말씀해주셨는지 궁금하구나, 정말로 이게 웬일이에요!"라고 말했답니다.
선녀는 토끼에게 "네가 나무꾼에게 전해주면, 그가 용기를 내어 마을 사람들에게 알려줄 수 있을 거란다!"라고 덧붙였지요.
나무꾼은 토끼의 말을 듣고 "아이고, 선녀님께서 나에게 특별한 일을 맡기셨다니, 이게 웬일이에요!"라고 감격했답니다.
토끼는 나무꾼의 얼굴을 바라보며 "아저씨, 선녀님께서 말씀하신 대로 용기를 내어 마을 사람들에게 이 이야기를 알려주셔야 해요!"라고 힘주어 이야기했답니다.
나무꾼은 토끼의 말을 듣고 "좋다, 내 용기를 모아 마을 사람들에게 이 특별한 이야기를 전해보겠노라!"라고 결심했답니다.
그렇게 나무꾼은 용기를 내어 마을 사람들을 모으고 "여러분, 오늘은 특별한 이야기를 전해드릴게요!"라고 외쳤답니다."""

# ── Story B: 사람 주인공 (릴리아, 유럽 중세) ────────────────────────────────
STORY_B = """옛날 옛적 유럽의 한 작은 마을에 사는 용감한 소녀 릴리아가 있었답니다.
릴리아는 매일 아침 해가 뜨기 전에 일어나 마을 근처의 숲속에서 신비로운 생물들과 놀며 즐거운 시간을 보냈답니다.
릴리아는 숲속에서 만나던 친구들인 작은 요정들과 함께 날마다 새로운 모험을 꿈꾸었답니다.
그러던 어느 날, 릴리아는 숲속 깊은 곳에서 반짝이는 수정 구슬을 발견했답니다.
릴리아는 그 수정 구슬에서 나오는 부드러운 빛을 보며, 마치 자신에게 특별한 일이 일어날 것 같은 기대감에 가슴이 두근거렸답니다.
그 순간, 릴리아는 자신이 마법의 세계로 들어갈 수 있는 용기를 가져야 한다는 사실을 깨달았답니다.
릴리아는 이제 그 수정 구슬을 어떻게 사용할지 고민하며, 용기 있게 다음 발걸음을 내딛기로 결심했답니다.
릴리아는 마음속 깊이 그 용기를 찾기 위해 수정 구슬을 손에 쥐고, 신비로운 숲속의 소리들을 귀 기울여 듣기 시작했답니다.
릴리아는 숲속의 모든 소리들이 마치 그녀에게 용기를 주는 듯 느껴지며, 이제는 그 수정 구슬을 통해 새로운 모험이 시작될 것이라는 설렘에 가슴이 벅차올랐답니다.
릴리아는 수정 구슬을 통해 새로운 세계로 나아가는 용기를 내기 위해 마음속에 간직한 꿈을 떠올리며, 그 꿈이 얼마나 아름다울지를 상상했답니다.
릴리아는 그 수정 구슬이 마치 자신의 용기를 시험하는 듯한 느낌을 주는 신비로운 존재라는 것을 깨달았답니다.
그때, 릴리아는 숲속의 친구들에게도 이 특별한 모험을 함께 나누고 싶다는 생각이 들어, 마음속의 용기를 더욱 키워야겠다고 다짐했답니다.
릴리아는 친구들과 함께하는 이 특별한 모험을 위해 용기를 내어 숲속 깊은 곳으로 들어가기로 마음먹었답니다.
릴리아는 그날 아침, 친구들에게 용기를 잃지 않고 함께 나아가자고 다짐하며 행복한 미소를 지었답니다.
릴리아는 용기 있게 친구들을 이끌고 숲속의 깊은 곳으로 들어가며, 그곳에서 마법의 세계가 펼쳐질 것이라는 설렘에 가슴이 뛰었답니다.
릴리아는 친구들과 함께 그 수정 구슬을 손에 쥐며, 깊은 숲속으로 나아가기로 결심했답니다.
릴리아는 친구들과 함께 깊은 숲속으로 들어가면서 그곳에서 마법의 세계가 어떻게 펼쳐질지에 대한 기대감에 가득 차 있었답니다.
릴리아는 친구들과 함께 깊은 숲속으로 들어가면서 불빛이 반짝이는 수정 구슬이 그들의 용기를 더욱 북돋아줄 것이라는 믿음에 가슴이 두근거렸답니다.
숲속 깊은 곳에서 마법의 세계가 펼쳐질 준비를 하고 있는 릴리아와 친구들은 그 순간의 설렘에 가득 차 있었답니다.
하지만 그때, 갑자기 숲속의 요정들이 나타나면서 "이 깊은 숲속에는 결코 들어오지 말라는 금지령이 있단다!"라며 릴리아와 친구들을 엄중히 경고했답니다."""


def print_divider(title: str = "", width: int = 70) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def print_plan_debug(plan) -> None:
    """스토리 분석 결과와 생성될 프롬프트를 상세하게 출력한다."""
    print_divider("스토리 분석 결과")
    print(f"  테마:      {plan.world.theme}  →  {plan.world.expansion_key}")
    print(f"  배경 힌트: {plan.world.positive_hint}")
    print(f"  배경 네거: {plan.world.negative_hint}")
    print()

    print_divider("캐릭터 Bible")
    for cid, desc in plan.character_bible.items():
        print(f"  [{cid}]  {desc}")
    print()

    for ch in plan.story_input.characters:
        parts = []
        if ch.species:
            parts.append(f"species={ch.species}")
        if ch.job:
            parts.append(f"job={ch.job}")
        if ch.type:
            parts.append(f"type={ch.type}")
        if ch.traits:
            parts.append(f"traits={ch.traits}")
        if ch.visual_hint:
            parts.append(f"visual_hint={ch.visual_hint}")
        print(f"  캐릭터 [{ch.id}]: {', '.join(parts)}")
    print()

    print_divider("장면별 프롬프트 (완전 출력)")
    for scene in plan.scenes:
        words = scene.prompt.split()
        print(f"\n  ▶ P{scene.page_index}  ({len(words)}단어 / 약 {int(len(words)*1.4)}토큰)")
        print(f"  원문: {scene.source_text}")
        print(f"  장면키워드: {scene.scene_spec.narrative_hint or '(없음)'}")
        print(f"  ─ POSITIVE ─")
        print(f"    {scene.prompt}")
        print(f"  ─ NEGATIVE ─")
        print(f"    {scene.negative_prompt}")
    print()


def generate_all_pages_sync(
    generator,
    scene_plans,
    output_dir: str,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_first_page_as_ref: bool = False,
) -> list[dict]:
    """모든 페이지를 동기적으로 생성하고 결과 목록을 반환한다."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    ref_image = None

    for i, scene in enumerate(scene_plans):
        try:
            current_ref = None
            ip_scale_for_this_page = None

            if i > 0 and ref_image is not None and use_first_page_as_ref:
                scene_char_count = len(scene.character_ids) if scene.character_ids else 0
                if scene_char_count >= 2:
                    ip_scale_for_this_page = 0.05
                current_ref = ref_image

            page_seed = (seed + scene.page_index) if seed is not None else None

            images, elapsed = generator.generate(
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                prompt_2=scene.prompt_2,
                seed=page_seed,
                width=width,
                height=height,
                ip_adapter_image=current_ref,
                ip_adapter_scale=ip_scale_for_this_page,
                lora_scale=getattr(scene, 'lora_scale', None),
            )

            if use_first_page_as_ref and i == 0 and images:
                ref_image = images[0]
                print(f"  [IP-Adapter] 1페이지를 이후 참고 이미지로 설정")

            saved = generator.save_images(
                images,
                output_dir=output_dir,
                prefix=f"page_{scene.page_index:02d}",
            )
            results.append({
                "page": scene.page_index,
                "path": saved[0] if saved else "",
                "elapsed": elapsed,
                "source_text": scene.source_text,
            })
            print(f"  Page {scene.page_index} 완료 ({elapsed:.1f}s): {scene.source_text[:30]}")
        except Exception as exc:
            print(f"  Page {scene.page_index} 실패: {exc}")
            results.append({
                "page": scene.page_index,
                "path": "",
                "elapsed": 0.0,
                "source_text": scene.source_text,
                "error": str(exc),
            })

    return results


def run_test(
    story_text: str,
    lora_key: str = "raw_200",
    seed: int | None = 42,
    low_memory: bool = False,
    output_dir: str = "outputs/test_raw",
    keep_loaded: bool = False,
):
    from api.pipeline.model_manager import ModelManager
    from api.pipeline.story_pipeline import build_story_plan

    print_divider("동화 삽화 파이프라인 테스트")
    print(f"  LoRA: {lora_key}  |  seed: {seed}  |  keep_loaded: {keep_loaded}")
    print_divider()

    # ── 1. 스토리 분석 & 프롬프트 생성 ───────────────────────────────────────
    print("\n[1] 스토리 분석 중...")
    plan = build_story_plan(story_text)
    print_plan_debug(plan)

    # ── 2. 모델 로드 ─────────────────────────────────────────────────────────
    print(f"[2] 모델 로드 (ModelManager) - lora={lora_key}...")
    mgr = ModelManager.get()
    mgr.load(
        base_model_key="sdxl",
        lora_key=lora_key,
        low_memory_mode=low_memory,
    )
    generator = mgr.generator

    # ── 2-1. IP-Adapter 로드 ─────────────────────────────────────────────────
    generator.load_ip_adapter(scale=0.2)
    print(f"    IP-Adapter: ON (scale=0.2, use_first_page_as_ref=True)")

    # ── 3. 전 페이지 삽화 생성 ───────────────────────────────────────────────
    print(f"\n[3] 삽화 생성 시작 ({len(plan.scenes)}페이지)...")
    start = time.time()

    results = generate_all_pages_sync(
        generator=generator,
        scene_plans=plan.scenes,
        output_dir=output_dir,
        seed=seed,
        width=1024,
        height=1024,
        use_first_page_as_ref=True,
    )

    total_elapsed = time.time() - start

    # ── 4. 결과 출력 ─────────────────────────────────────────────────────────
    print_divider("생성 결과")
    print(f"  총 소요 시간: {total_elapsed:.1f}초")
    print(f"  저장 위치: {Path(output_dir).resolve()}")
    print()

    for r in results:
        status = "OK" if r.get("path") else "FAIL"
        print(f"  P{r['page']} {status} ({r['elapsed']:.1f}s): {r['source_text'][:40]}")
        if r.get("path"):
            print(f"       → {r['path']}")
        if r.get("error"):
            print(f"       오류: {r['error']}")

    # ── 5. 모델 언로드 ───────────────────────────────────────────────────────
    if not keep_loaded:
        mgr.unload()
        print("\n[모델 언로드 완료 - 다시 쓰려면 --keep-loaded 옵션 사용]")
    else:
        print("\n[모델 유지 중 - 다음 run_test() 호출 시 즉시 재사용]")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="동화 삽화 파이프라인 테스트")
    parser.add_argument("--story", type=str, default=None, help="동화 텍스트 (기본: 개구리 왕자)")
    story_group = parser.add_mutually_exclusive_group()
    story_group.add_argument("--story-a", action="store_true", help="Story A: 동물 주인공 (토끼+나무꾼+선녀)")
    story_group.add_argument("--story-b", action="store_true", help="Story B: 사람 주인공 (릴리아, 유럽 중세)")
    parser.add_argument("--lora", type=str, default="raw_200", help="사용할 LoRA 키")
    parser.add_argument("--seed", type=int, default=42, help="시드값 (캐릭터 일관성)")
    parser.add_argument("--low-memory", action="store_true", help="저메모리 모드 (VRAM 부족 시)")
    parser.add_argument("--output", type=str, default="outputs/test_raw")
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="실행 후 모델을 언로드하지 않음 (연속 테스트 시 유용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.story_a:
        story = STORY_A
    elif args.story_b:
        story = STORY_B
    elif args.story:
        story = args.story
    else:
        story = DEFAULT_STORY
    run_test(
        story_text=story,
        lora_key=args.lora,
        seed=args.seed,
        low_memory=args.low_memory,
        output_dir=args.output,
        keep_loaded=args.keep_loaded,
    )
