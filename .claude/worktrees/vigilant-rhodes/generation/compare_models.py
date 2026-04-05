"""
모델/LoRA 비교 실험 스크립트
동일한 프롬프트로 여러 모델+LoRA 조합을 테스트하고
속도 및 품질 비교 결과를 저장합니다.

사용법:
    # 추천 조합 비교
    python compare_models.py

    # 특정 프리셋으로 비교
    python compare_models.py --preset enchanted_forest

    # 커스텀 프롬프트로 비교
    python compare_models.py --prompt "a brave little prince in magical forest"

    # 조합 직접 지정
    python compare_models.py --combos sdxl:storybook_sdxl sd15:storybook_sd15
"""

import argparse
import json
import os
import time
from pathlib import Path

from fairytale_lora.generation.config import (
    FAIRYTALE_PROMPTS,
    OUTPUT_CONFIG,
    RECOMMENDED_COMBOS,
    PROJECT_ROOT,
)
from fairytale_lora.generation.generator import FairytaleImageGenerator, create_comparison_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="모델/LoRA 조합 비교 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, default=None, help="테스트 프롬프트")
    parser.add_argument(
        "--preset", type=str, default="enchanted_forest",
        choices=list(FAIRYTALE_PROMPTS.keys()),
        help="동화 프리셋 선택 (기본: enchanted_forest)"
    )
    parser.add_argument(
        "--combos", nargs="+", default=None,
        metavar="MODEL:LORA",
        help="비교할 조합 (예: sdxl:storybook_sdxl sd15:storybook_sd15)"
    )
    parser.add_argument("--seed", type=int, default=42, help="공통 시드 (기본: 42)")
    parser.add_argument(
        "--output", type=str, default=str(PROJECT_ROOT / "comparison_outputs"),
        help="결과 저장 폴더"
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="저메모리 모드"
    )
    return parser.parse_args()


def run_comparison(
    combos: list[tuple[str, str]],
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    output_dir: str = str(PROJECT_ROOT / "comparison_outputs"),
    low_memory: bool = False,
) -> dict:
    """
    여러 모델/LoRA 조합으로 동일 프롬프트 이미지 생성 및 비교

    Returns:
        결과 딕셔너리 {combo_key: {image, time, path}}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    all_images = []
    all_labels = []

    print("\n" + "=" * 60)
    print(f"  비교 실험 시작 ({len(combos)}개 조합)")
    print("=" * 60)
    print(f"  프롬프트: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
    print(f"  시드: {seed}")
    print()

    for model_key, lora_key in combos:
        combo_key = f"{model_key}_{lora_key}"
        print(f"\n[{combo_key}] 실행 중...")
        print("-" * 40)

        try:
            gen = FairytaleImageGenerator(
                base_model_key=model_key,
                lora_key=lora_key if lora_key != "none" else None,
                low_memory_mode=low_memory,
            )
            gen.load()

            images, elapsed = gen.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                num_images=1,
            )

            # 개별 이미지 저장
            saved = gen.save_images(images, output_dir=output_dir, prefix="compare")
            gen.unload()

            results[combo_key] = {
                "model": model_key,
                "lora": lora_key,
                "time_seconds": round(elapsed, 2),
                "path": saved[0] if saved else None,
                "success": True,
            }
            all_images.append(images[0])
            all_labels.append(f"{model_key} + {lora_key}\n({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [오류] {combo_key} 실패: {e}")
            results[combo_key] = {
                "model": model_key,
                "lora": lora_key,
                "error": str(e),
                "success": False,
            }

    # 비교 격자 이미지 생성
    if len(all_images) > 1:
        print("\n비교 격자 이미지 생성 중...")
        grid = create_comparison_grid(all_images, all_labels, cols=min(2, len(all_images)))
        timestamp = int(time.time())
        grid_path = os.path.join(output_dir, f"comparison_grid_{timestamp}.png")
        grid.save(grid_path)
        print(f"  비교 이미지 저장: {grid_path}")
        results["_comparison_grid"] = grid_path

    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("  결과 요약 (속도 기준 정렬)")
    print("=" * 60)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if not k.startswith("_") and v["success"]],
        key=lambda x: x[1]["time_seconds"],
    )
    for rank, (key, res) in enumerate(sorted_results, 1):
        print(f"  {rank}위: {key}")
        print(f"       생성 시간: {res['time_seconds']}초")

    # JSON 결과 저장
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": prompt,
                "seed": seed,
                "timestamp": int(time.time()),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n결과 리포트 저장: {report_path}")

    return results


def main():
    args = parse_args()

    # 프롬프트 결정
    if args.prompt:
        prompt = args.prompt
        negative = ""
    else:
        preset = FAIRYTALE_PROMPTS[args.preset]
        prompt = preset["positive"]
        negative = preset["negative"]
        print(f"프리셋 사용: {args.preset}")

    # 비교 조합 결정
    if args.combos:
        combos = []
        for combo_str in args.combos:
            parts = combo_str.split(":")
            if len(parts) == 2:
                combos.append((parts[0], parts[1]))
            else:
                print(f"[경고] 잘못된 조합 형식 무시: {combo_str} (예: sdxl:storybook_sdxl)")
    else:
        combos = RECOMMENDED_COMBOS
        print(f"추천 조합 {len(combos)}개로 비교합니다.")

    run_comparison(
        combos=combos,
        prompt=prompt,
        negative_prompt=negative,
        seed=args.seed,
        output_dir=args.output,
        low_memory=args.low_memory,
    )


if __name__ == "__main__":
    main()
