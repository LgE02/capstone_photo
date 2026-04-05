"""
체크포인트별 비교 테스트 — 모델을 바꿔가며 같은 프롬프트로 생성해 비교한다.
ModelManager 덕분에 같은 조합이면 재로드 없이 재사용한다.

실행:
    cd fairytale_lora
    python compare_checkpoints.py

    # 특정 체크포인트만 테스트:
    python compare_checkpoints.py --loras retrain_200 retrain_400

    # 저메모리 모드:
    python compare_checkpoints.py --low-memory
"""

import argparse
from test_pipeline import run_test, DEFAULT_STORY, print_divider

# 비교할 LoRA 키 목록 (권장 테스트 순서)
DEFAULT_LORAS = [
    "raw_200",
    "raw_300",
    "raw_400",
    "raw_100",
    "raw_500",
]

TEST_STORY = """옛날에 작은 토끼가 산속 초가집에서 살았습니다.
어느 날 토끼는 꽃밭에서 나비를 쫓으며 뛰어다녔습니다.
토끼는 장터에서 떡을 사서 임금님께 가져갔습니다.
임금님은 기와집 궁궐에서 토끼를 반갑게 맞아주었습니다."""


def main(loras: list[str], story: str, seed: int, low_memory: bool) -> None:
    print_divider("체크포인트 비교 테스트")
    print(f"  테스트할 LoRA: {loras}")
    print(f"  seed: {seed}")
    print_divider()

    for i, lora_key in enumerate(loras):
        is_last = (i == len(loras) - 1)
        output_dir = f"outputs/compare/{lora_key}"
        print(f"\n[{i+1}/{len(loras)}] {lora_key}")

        run_test(
            story_text=story,
            lora_key=lora_key,
            seed=seed,
            low_memory=low_memory,
            output_dir=output_dir,
            keep_loaded=not is_last,  # 마지막 것만 언로드
        )

    print()
    print_divider("비교 완료")
    print("  결과 저장 위치: outputs/compare/<lora_key>/")
    print("  각 폴더의 이미지를 열어서 구도와 캐릭터 크기를 비교하세요.")
    print_divider()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="체크포인트별 비교 테스트")
    parser.add_argument("--loras", nargs="+", default=DEFAULT_LORAS)
    parser.add_argument("--story", type=str, default=TEST_STORY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-memory", action="store_true")
    args = parser.parse_args()
    main(args.loras, args.story, args.seed, args.low_memory)
