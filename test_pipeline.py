"""
동화 삽화 파이프라인 테스트 스크립트
새로 학습된 retrain_after03 LoRA로 삽화를 생성합니다.

실행 방법:
    cd fairytale_lora
    python test_pipeline.py

    # 다른 동화 텍스트로 테스트:
    python test_pipeline.py --story "호랑이 사또 이야기..."

    # 시드 고정 (캐릭터 일관성):
    python test_pipeline.py --seed 42

    # 저메모리 모드 (VRAM 부족 시):
    python test_pipeline.py --low-memory

    # 모델을 언로드하지 않고 연속 실행 (ModelManager 재사용):
    python test_pipeline.py --keep-loaded
"""

import argparse
import time
from pathlib import Path

# ── 테스트 동화 텍스트 ────────────────────────────────────────────────────────
DEFAULT_STORY = """옛날에 개구리 대군님이 한옥 마을에서 살았습니다.
어느 날 대군님은 장터에서 편지를 받았습니다.
비가 내리는 궁궐 앞에서 기뻐하며 춤을 추었습니다.
임금님께서 등불을 들고 기다리고 있었습니다."""


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

    # 캐릭터 상세
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
        # 프롬프트를 구조별로 줄바꿈해서 읽기 쉽게 출력
        print(f"    {scene.prompt}")
        print(f"  ─ NEGATIVE ─")
        print(f"    {scene.negative_prompt}")
    print()


def run_test(
    story_text: str,
    lora_key: str = "raw_200",
    seed: int | None = 42,
    low_memory: bool = False,
    output_dir: str = "outputs/test_raw",
    keep_loaded: bool = False,
):
    from runtime_pipeline.model_manager import ModelManager
    from runtime_pipeline.story_pipeline import build_story_plan
    from runtime_pipeline.async_generator import generate_all_pages_sync

    print_divider("동화 삽화 파이프라인 테스트")
    print(f"  LoRA: {lora_key}  |  seed: {seed}  |  keep_loaded: {keep_loaded}")
    print_divider()

    # ── 1. 스토리 분석 & 프롬프트 생성 ───────────────────────────────────────
    print("\n[1] 스토리 분석 중...")
    plan = build_story_plan(story_text)
    print_plan_debug(plan)

    # ── 2. 모델 로드 (이미 로드된 경우 재사용) ─────────────────────────────
    print(f"[2] 모델 로드 (ModelManager) - lora={lora_key}...")
    mgr = ModelManager.get()
    mgr.load(
        base_model_key="sdxl",
        lora_key=lora_key,
        low_memory_mode=low_memory,
    )
    generator = mgr.generator
    print(f"    LoRA: {lora_key}  |  scale: {generator._lora_scale if hasattr(generator, '_lora_scale') else 'N/A'}")

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
    )

    total_elapsed = time.time() - start

    # ── 4. 결과 출력 ─────────────────────────────────────────────────────────
    print_divider("생성 결과")
    print(f"  총 소요 시간: {total_elapsed:.1f}초")
    print(f"  저장 위치: {Path(output_dir).resolve()}")
    print()

    for r in results:
        status = "✓" if r.get("path") else "✗ 실패"
        print(f"  P{r['page']} {status} ({r['elapsed']:.1f}s): {r['source_text'][:40]}")
        if r.get("path"):
            print(f"       → {r['path']}")
        if r.get("error"):
            print(f"       오류: {r['error']}")

    # ── 5. 모델 언로드 (keep_loaded=False일 때만) ─────────────────────────
    if not keep_loaded:
        mgr.unload()
        print("\n[모델 언로드 완료 - 다시 쓰려면 --keep-loaded 옵션 사용]")
    else:
        print("\n[모델 유지 중 - 다음 run_test() 호출 시 즉시 재사용]")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="동화 삽화 파이프라인 테스트")
    parser.add_argument("--story", type=str, default=None, help="동화 텍스트 (기본: 개구리 왕자)")
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
    story = args.story or DEFAULT_STORY
    run_test(
        story_text=story,
        lora_key=args.lora,
        seed=args.seed,
        low_memory=args.low_memory,
        output_dir=args.output,
        keep_loaded=args.keep_loaded,
    )
