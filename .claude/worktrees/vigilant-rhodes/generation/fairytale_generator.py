"""
동화 삽화 이미지 생성기 - 메인 실행 스크립트
사용법:
    # 기본 실행 (추천 설정)
    python fairytale_generator.py

    # 프롬프트 직접 지정
    python fairytale_generator.py --prompt "a little mermaid in magical ocean"

    # 모델과 LoRA 지정
    python fairytale_generator.py --model sdxl --lora storybook_sdxl

    # 프리셋 사용
    python fairytale_generator.py --preset enchanted_forest

    # 여러 장 생성
    python fairytale_generator.py --num 4 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

from fairytale_lora.generation.config import (
    BASE_MODELS,
    LORA_CONFIGS,
    FAIRYTALE_PROMPTS,
    OUTPUT_CONFIG,
    RECOMMENDED_COMBOS,
)
from fairytale_lora.generation.generator import FairytaleImageGenerator, create_comparison_grid


# ─────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="동화 삽화 이미지 생성기 (LoRA 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python fairytale_generator.py
  python fairytale_generator.py --prompt "a brave princess in enchanted forest"
  python fairytale_generator.py --model sd15 --lora storybook_sd15 --num 2
  python fairytale_generator.py --preset dragon_adventure --seed 1234
  python fairytale_generator.py --list-models
        """,
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="생성할 이미지 설명 (영문 권장)"
    )
    parser.add_argument(
        "--negative", type=str, default=None,
        help="제외할 요소 설명"
    )
    parser.add_argument(
        "--model", type=str, default="sdxl",
        choices=list(BASE_MODELS.keys()),
        help="베이스 모델 선택 (기본: sdxl)"
    )
    parser.add_argument(
        "--lora", type=str, default="storybook_sdxl",
        choices=list(LORA_CONFIGS.keys()) + ["none"],
        help="LoRA 설정 선택 (기본: storybook_sdxl)"
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(FAIRYTALE_PROMPTS.keys()),
        help="동화 프리셋 사용"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="이미지 너비 (기본: 모델 기본값)"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="이미지 높이 (기본: 모델 기본값)"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="추론 스텝 수 (기본: 모델 기본값)"
    )
    parser.add_argument(
        "--cfg", type=float, default=None,
        help="CFG 스케일 (기본: 7.5)"
    )
    parser.add_argument(
        "--lora-scale", type=float, default=None,
        help="LoRA 적용 강도 override (예: 0.7, 0.85, 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="랜덤 시드 (재현성)"
    )
    parser.add_argument(
        "--num", type=int, default=1,
        help="생성할 이미지 수 (기본: 1)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_CONFIG["output_dir"],
        help=f"출력 폴더 (기본: {OUTPUT_CONFIG['output_dir']})"
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="저메모리 모드 (VRAM 부족 시 사용)"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="사용 가능한 모델/LoRA 목록 출력"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# 모델 목록 출력
# ─────────────────────────────────────────────
def print_available_options():
    print("\n" + "=" * 60)
    print("  베이스 모델 목록")
    print("=" * 60)
    for key, cfg in BASE_MODELS.items():
        size = "x".join(map(str, cfg["default_size"]))
        print(f"  [{key}] {cfg['description']}")
        print(f"         모델: {cfg['model_id']}")
        print(f"         기본 크기: {size}, 스텝: {cfg['num_inference_steps']}\n")

    print("=" * 60)
    print("  LoRA 목록")
    print("=" * 60)
    for key, cfg in LORA_CONFIGS.items():
        source = cfg["repo_id"] if cfg["repo_id"] else "로컬 파일"
        print(f"  [{key}] {cfg['description']}")
        print(f"         소스: {source}")
        print(f"         베이스: {cfg['base_model']} | 스케일: {cfg['lora_scale']}\n")

    print("=" * 60)
    print("  프리셋 프롬프트 목록")
    print("=" * 60)
    for key, cfg in FAIRYTALE_PROMPTS.items():
        print(f"  [{key}]")
        print(f"         {cfg['positive'][:70]}...\n")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # 목록 출력만 하는 경우
    if args.list_models:
        print_available_options()
        return

    print("\n" + "=" * 60)
    print("  동화 삽화 이미지 생성기")
    print("=" * 60)

    # 프롬프트 결정
    if args.preset:
        preset = FAIRYTALE_PROMPTS[args.preset]
        prompt = preset["positive"]
        negative = args.negative or preset["negative"]
        print(f"프리셋 사용: {args.preset}")
    elif args.prompt:
        prompt = args.prompt
        negative = args.negative or ""
    else:
        # 기본 프롬프트
        prompt = (
            "a magical enchanted forest with glowing fireflies, "
            "a small fairy cottage, watercolor illustration, "
            "children's book art, soft pastel colors, whimsical"
        )
        negative = ""
        print("기본 프롬프트 사용 (--prompt로 변경 가능)")

    # LoRA 키 처리
    lora_key = None if args.lora == "none" else args.lora

    # LoRA와 베이스 모델 호환성 확인
    if lora_key and lora_key in LORA_CONFIGS:
        lora_base = LORA_CONFIGS[lora_key]["base_model"]
        if lora_base != args.model:
            print(f"\n[주의] 선택한 LoRA({lora_key})는 {lora_base} 모델용입니다.")
            print(f"       현재 선택된 모델: {args.model}")
            answer = input("계속 진행할까요? (y/N): ").strip().lower()
            if answer != "y":
                print("모델을 맞춰 재실행하세요:")
                print(f"  python fairytale_generator.py --model {lora_base} --lora {lora_key}")
                return

    # 생성기 초기화 및 로드
    generator = FairytaleImageGenerator(
        base_model_key=args.model,
        lora_key=lora_key,
        low_memory_mode=args.low_memory,
        lora_scale_override=args.lora_scale,
    )
    generator.load()

    # 이미지 생성
    images, elapsed = generator.generate(
        prompt=prompt,
        negative_prompt=negative,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        num_images=args.num,
    )

    # 이미지 저장
    print("\n이미지 저장 중...")
    saved = generator.save_images(images, output_dir=args.output)

    print(f"\n생성 완료!")
    print(f"  총 소요 시간: {elapsed:.1f}초")
    print(f"  저장 위치: {os.path.abspath(args.output)}")
    for p in saved:
        print(f"    - {os.path.basename(p)}")

    # 메모리 정리
    generator.unload()


if __name__ == "__main__":
    main()
