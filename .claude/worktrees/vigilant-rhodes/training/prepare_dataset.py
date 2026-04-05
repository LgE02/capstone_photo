"""
LoRA 학습용 데이터셋 준비 스크립트

수행 작업:
  1. dataset/raw/ 의 이미지를 1024x1024로 리사이즈 → dataset/train/ 저장
  2. BLIP 모델로 각 이미지에 대한 캡션 자동 생성 (.txt 파일)
  3. 스타일 트리거 단어를 캡션 앞에 자동 삽입

사용법:
  # 기본 실행 (자동 캡션 생성)
  python prepare_dataset.py

  # 캡션 생성 없이 리사이즈만
  python prepare_dataset.py --no-caption

  # 직접 캡션 입력 모드
  python prepare_dataset.py --manual-caption
"""

import argparse
import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

from fairytale_lora.training.train_config import PATHS, DATASET


STYLE_PREFIX = (
    "preschool picture book illustration for children ages 3 to 4, "
    "cute simple children's book art, soft pastel colors, plain cream background, "
    "minimal background, character-focused composition, single large main character, "
    "chibi proportions, deformed cute style, clear shapes, gentle expression, "
    "clean composition, uncluttered scene"
)


# ─────────────────────────────────────────────
# 이미지 리사이즈 및 정사각형 패딩
# ─────────────────────────────────────────────
def resize_and_crop(img: Image.Image, target_size: int) -> Image.Image:
    """
    이미지를 target_size × target_size 로 변환.
    - center_crop=True: 중앙 크롭
    - center_crop=False: 비율 유지 + 흰 배경 패딩
    """
    w, h = img.size

    if DATASET["center_crop"]:
        # 짧은 쪽 기준으로 리사이즈 후 중앙 크롭
        scale = target_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))
    else:
        # 비율 유지 리사이즈 후 흰 배경 패딩
        img.thumbnail((target_size, target_size), Image.LANCZOS)
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
        new_img.paste(img, offset)
        img = new_img

    return img


# ─────────────────────────────────────────────
# BLIP 자동 캡션 생성
# ─────────────────────────────────────────────
class BLIPCaptioner:
    def __init__(self):
        self.processor = None
        self.model = None

    def load(self):
        print("BLIP 캡션 모델 로딩 중...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16,
        )
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.device = device
        print("BLIP 모델 로딩 완료!\n")

    def caption(self, img: Image.Image) -> str:
        import torch
        inputs = self.processor(img, return_tensors="pt").to(
            self.device, torch.float16
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=5,
            early_stopping=True,
        )
        return self.processor.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# 메인 데이터 준비 함수
# ─────────────────────────────────────────────
def prepare_dataset(use_caption: bool = True, manual_caption: bool = False):
    raw_dir = Path(PATHS["raw_images_dir"])
    train_dir = Path(PATHS["train_data_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)
    target_size = DATASET["resolution"]
    trigger = DATASET["instance_prompt"]

    # 지원 이미지 포맷
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        f for f in raw_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"[오류] {raw_dir} 에 이미지 파일이 없습니다.")
        print(f"  동화책 삽화 이미지 20~50장을 {raw_dir} 폴더에 넣어주세요.")
        return

    print(f"총 {len(image_files)}장 이미지 발견\n")

    # BLIP 모델 로드
    captioner = None
    if use_caption and not manual_caption:
        captioner = BLIPCaptioner()
        captioner.load()

    # ─── 이미지 처리 루프 ────────────────────────
    for idx, img_path in enumerate(tqdm(image_files, desc="데이터셋 준비")):
        img = Image.open(img_path).convert("RGB")

        # 리사이즈
        img_resized = resize_and_crop(img, target_size)

        # 저장 파일명
        out_name = f"{idx+1:04d}"
        out_img_path = train_dir / f"{out_name}.png"
        out_txt_path = train_dir / f"{out_name}.txt"

        # 이미지 저장
        img_resized.save(out_img_path, format="PNG")

        # 캡션 생성 또는 입력
        if manual_caption:
            print(f"\n[{idx+1}/{len(image_files)}] {img_path.name}")
            img.show()
            caption_body = input("캡션 입력 (Enter = 기본 캡션): ").strip()
            if not caption_body:
                caption_body = DATASET["class_prompt"]
        elif captioner:
            caption_body = captioner.caption(img_resized)
        else:
            caption_body = DATASET["class_prompt"]

        # 트리거 단어 + 캡션 결합
        full_caption = f"{trigger}, {STYLE_PREFIX}, {caption_body}"
        out_txt_path.write_text(full_caption, encoding="utf-8")

    print(f"\n완료! {len(image_files)}장 처리됨")
    print(f"저장 위치: {train_dir.resolve()}")
    print("\n캡션 예시 (첫 번째 파일):")
    first_txt = next(train_dir.glob("*.txt"), None)
    if first_txt:
        print(f"  {first_txt.read_text(encoding='utf-8')}")

    print(f"\n다음 단계: python train_lora.py")


# ─────────────────────────────────────────────
# 데이터셋 통계 출력
# ─────────────────────────────────────────────
def show_stats():
    train_dir = Path(PATHS["train_data_dir"])
    images = list(train_dir.glob("*.png"))
    captions = list(train_dir.glob("*.txt"))
    print(f"\n데이터셋 현황 ({train_dir})")
    print(f"  이미지: {len(images)}장")
    print(f"  캡션:   {len(captions)}개")
    if len(images) < 20:
        print(f"  [권장] 최소 20장 이상 준비하세요 (현재 {len(images)}장)")
    elif len(images) >= 50:
        print(f"  충분한 데이터셋입니다!")
    else:
        print(f"  적절한 데이터셋 크기입니다.")

    # 이미지 크기 확인
    if images:
        sample = Image.open(images[0])
        print(f"  이미지 크기: {sample.size[0]}x{sample.size[1]}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 학습 데이터셋 준비")
    parser.add_argument(
        "--no-caption", action="store_true",
        help="캡션 생성 없이 이미지 리사이즈만 수행"
    )
    parser.add_argument(
        "--manual-caption", action="store_true",
        help="이미지 하나씩 보면서 캡션 직접 입력"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="현재 데이터셋 통계만 출력"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 원본 이미지 폴더 자동 생성
    Path(PATHS["raw_images_dir"]).mkdir(parents=True, exist_ok=True)

    if args.stats:
        show_stats()
    else:
        prepare_dataset(
            use_caption=not args.no_caption,
            manual_caption=args.manual_caption,
        )
        show_stats()
