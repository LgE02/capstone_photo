"""
FLUX.2-klein-4B 동화 삽화 파이프라인 테스트

실행:
    python test_pipeline_klein.py            # Story A (기본)
    python test_pipeline_klein.py --story-a  # Story A: 동물 주인공 (토끼+나무꾼+선녀)
    python test_pipeline_klein.py --story-b  # Story B: 사람 주인공 (릴리아, 유럽 중세)
    python test_pipeline_klein.py --seed 42
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

# ── 테스트 동화 텍스트 (기존 test_pipeline.py와 동일) ───────────────────────

STORY_A = """옛날 옛날 한 옛날에, 어느 작은 마을에 용감한 마음을 가진 토끼 한 마리가 살았답니다.
토끼는 용감한 마음을 가졌지만, 평소에는 숲을 뛰어다니며 친구들과 놀기만 했답니다.
하루는 토끼가 숲속에서 친구들과 신나게 놀고 있을 때, 나무꾼이 나타나서 "이게 웬일이에요! 오늘은 특별한 일이 일어날 것 같아요!"라고 외쳤답니다.
그런데 이를 어쩌지요! 나무꾼은 숲속에서 토끼에게 큰일이 날 것 같다는 이야기를 하면서, 믿을 수 없는 소문을 전했답니다.
아이고! 나무꾼의 이야기를 듣고 토끼는 마음속으로 두려운 생각이 들었지만, 용감한 마음을 가진 만큼 친구들에게는 아무 말도 하지 않았답니다.
그날 오후, 토끼는 나무꾼의 말을 잊지 않으려 애쓰며 숲속에서 혼자 사색에 잠겼답니다.
그런데 갑자기 토끼의 귀에 바람 소리가 들려왔고, 그 소리가 마치 누군가의 속삭임처럼 느껴졌답니다.
그런데 갑자기 바람 소리 너머에서 누군가의 모습이 보였고, 그건 바로 숲속의 신비로운 선녀였답니다!
선녀는 토끼에게 "너는 용감한 마음을 가지고 있구나, 그래서 나는 너에게 특별한 일을 부탁하고 싶어!"라고 말했답니다.
토끼는 선녀의 부탁을 듣고 "어떤 특별한 일을 부탁하실 건가요?"라고 궁금해했지요."""

STORY_B = """옛날 옛적 유럽의 한 작은 마을에 사는 용감한 소녀 릴리아가 있었답니다.
릴리아는 매일 아침 해가 뜨기 전에 일어나 마을 근처의 숲속에서 신비로운 생물들과 놀며 즐거운 시간을 보냈답니다.
릴리아는 숲속에서 만나던 친구들인 작은 요정들과 함께 날마다 새로운 모험을 꿈꾸었답니다.
그러던 어느 날, 릴리아는 숲속 깊은 곳에서 반짝이는 수정 구슬을 발견했답니다.
릴리아는 그 수정 구슬에서 나오는 부드러운 빛을 보며, 마치 자신에게 특별한 일이 일어날 것 같은 기대감에 가슴이 두근거렸답니다.
그 순간, 릴리아는 자신이 마법의 세계로 들어갈 수 있는 용기를 가져야 한다는 사실을 깨달았답니다.
릴리아는 이제 그 수정 구슬을 어떻게 사용할지 고민하며, 용기 있게 다음 발걸음을 내딛기로 결심했답니다.
릴리아는 마음속 깊이 그 용기를 찾기 위해 수정 구슬을 손에 쥐고, 신비로운 숲속의 소리들을 귀 기울여 듣기 시작했답니다.
릴리아는 숲속의 모든 소리들이 마치 그녀에게 용기를 주는 듯 느껴지며, 이제는 그 수정 구슬을 통해 새로운 모험이 시작될 것이라는 설렘에 가슴이 벅차올랐답니다.
하지만 그때, 갑자기 숲속의 요정들이 나타나면서 "이 깊은 숲속에는 결코 들어오지 말라는 금지령이 있단다!"라며 릴리아와 친구들을 엄중히 경고했답니다."""


def print_divider(title: str = "", width: int = 70) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def print_plan_debug(plan) -> None:
    print_divider("스토리 분석 결과")
    print(f"  테마:      {plan.world.theme}  →  {plan.world.expansion_key}")
    print(f"  배경 힌트: {plan.world.positive_hint}")
    print()

    print_divider("캐릭터 Bible")
    for cid, desc in plan.character_bible.items():
        print(f"  [{cid}]  {desc}")
    print()

    print_divider("장면별 프롬프트 (Klein)")
    for scene in plan.scenes:
        words = scene.prompt.split()
        print(f"\n  ▶ P{scene.page_index}  ({len(words)}단어)")
        print(f"  원문: {scene.source_text}")
        print(f"  ─ PROMPT ─")
        print(f"    {scene.prompt}")
    print()


def generate_all_pages(
    generator,
    scene_plans,
    output_dir: str,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    use_reference: bool = True,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for scene in scene_plans:
        try:
            page_seed = (seed + scene.page_index) if seed is not None else None

            images, elapsed = generator.generate(
                prompt=scene.prompt,
                seed=page_seed,
                width=width,
                height=height,
                use_reference=use_reference,
            )

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
            print(f"  Page {scene.page_index} 완료 ({elapsed:.1f}s): {scene.source_text[:40]}")

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
    seed: int | None = 42,
    output_dir: str = "outputs/test_klein",
    keep_loaded: bool = False,
    use_reference: bool = True,
):
    from api_klein.pipeline.model_manager import KleinModelManager
    from api_klein.pipeline.story_pipeline import build_story_plan

    print_divider("동화 삽화 파이프라인 테스트 (FLUX.2-klein-4B)")
    print(f"  seed: {seed}")
    print_divider()

    # ── 1. 스토리 분석 ────────────────────────────────────────────────────────
    print("\n[1] 스토리 분석 중...")
    plan = build_story_plan(story_text)
    print_plan_debug(plan)

    # ── 2. 모델 로드 ──────────────────────────────────────────────────────────
    print("[2] FLUX.2-klein-4B 모델 로드 중...")
    mgr = KleinModelManager.get()
    mgr.load()
    generator = mgr.generator

    # ── 3. 캐릭터 레퍼런스 생성 ───────────────────────────────────────────────
    print("\n[3] 캐릭터 레퍼런스 생성 중...")
    main_char_id = plan.story_input.characters[0].id if plan.story_input.characters else None
    char_prompt = plan.character_bible.get(main_char_id, "") if main_char_id else ""

    if char_prompt:
        ref_image = generator.generate_character_reference(char_prompt, seed=seed or 42)
        ref_dir = os.path.join(output_dir, "reference")
        generator.save_character_reference(output_dir=ref_dir)
        print(f"  캐릭터 레퍼런스 저장: {ref_dir}")
    else:
        print("  캐릭터 정보 없음 — 레퍼런스 없이 생성")

    # ── 4. 전 페이지 삽화 생성 ───────────────────────────────────────────────
    print(f"\n[4] 삽화 생성 시작 ({len(plan.scenes)}페이지)...")
    start = time.time()

    results = generate_all_pages(
        generator=generator,
        scene_plans=plan.scenes,
        output_dir=output_dir,
        seed=seed,
        width=1024,
        height=1024,
        use_reference=use_reference,
    )

    total_elapsed = time.time() - start

    # ── 5. 결과 출력 ──────────────────────────────────────────────────────────
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

    # ── 6. 모델 언로드 ────────────────────────────────────────────────────────
    if not keep_loaded:
        mgr.unload()
        print("\n[모델 언로드 완료]")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="동화 삽화 파이프라인 테스트 (FLUX.2-klein-4B)")
    story_group = parser.add_mutually_exclusive_group()
    story_group.add_argument("--story-a", action="store_true", help="Story A: 동물 주인공 (토끼+나무꾼+선녀)")
    story_group.add_argument("--story-b", action="store_true", help="Story B: 사람 주인공 (릴리아, 유럽 중세)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="출력 폴더 (미지정 시 자동)")
    parser.add_argument("--no-ref", action="store_true", help="캐릭터 레퍼런스 이미지 미사용")
    parser.add_argument("--keep-loaded", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    story_name = "story_b" if args.story_b else "story_a"
    ref_tag = "no_ref" if args.no_ref else "with_ref"

    # 출력 폴더 자동 생성: outputs/test_klein/{story_name}_{ref_tag}/
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"outputs/test_klein/{story_name}_{ref_tag}"

    story = STORY_B if args.story_b else STORY_A
    run_test(
        story_text=story,
        seed=args.seed,
        output_dir=output_dir,
        keep_loaded=args.keep_loaded,
        use_reference=not args.no_ref,
    )
