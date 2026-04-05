"""
이미지 자동 캡션 생성 스크립트

dataset/raw/ 폴더의 이미지를 분석해서
동화 삽화 학습에 최적화된 캡션을 자동으로 생성합니다.

사용법:
  python add_captions.py              # BLIP으로 자동 캡션 생성
  python add_captions.py --preview    # 캡션 미리보기만 (저장 안 함)
  python add_captions.py --overwrite  # 기존 캡션 덮어쓰기
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from fairytale_lora.training.train_config import PATHS, DATASET

RAW_DIR = Path(PATHS["raw_images_dir"])
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TRIGGER = (
    "ftbookstyle, preschool picture book illustration for children ages 3 to 4, "
    "cute simple children's book art, soft pastel colors, minimal background, "
    "clear shapes, gentle expression"
)


def load_blip():
    print("BLIP 모델 로딩 중 (첫 실행 시 약 1~2분 소요)...")
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=dtype,
    ).to(device)
    print(f"BLIP 로드 완료 ({device})\n")
    return processor, model, device


def generate_caption(img: Image.Image, processor, model, device) -> str:
    """BLIP으로 이미지 캡션 생성"""
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(img.convert("RGB"), return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            early_stopping=True,
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def classify_style(filename: str) -> str:
    """
    파일명으로 카테고리 힌트 감지
    (korean/traditional/hanbok → 한국 전통, animal/동물 → 동물)
    """
    name = filename.lower()
    korean_keywords = ["korean", "hanbok", "traditional", "전래", "한복", "심청", "콩쥐", "goryeo"]
    animal_keywords  = ["animal", "fox", "rabbit", "bear", "turtle", "tiger", "owl", "duck", "동물"]

    if any(k in name for k in korean_keywords):
        return "korean"
    elif any(k in name for k in animal_keywords):
        return "animal"
    else:
        return "western"


def enrich_caption(base_caption: str, style: str) -> str:
    """
    BLIP 캡션에 스타일 힌트 추가
    """
    style_hints = {
        "korean": "Korean hanbok character, simple shapes, warm and friendly picture book mood",
        "animal": "friendly animal character, simple shapes, child-friendly picture book style",
        "western": "friendly character, simple clean background, warm and gentle storybook mood",
    }
    hint = style_hints.get(style, "")
    return f"{TRIGGER}, {base_caption}, {hint}"


def run(preview: bool = False, overwrite: bool = False):
    images = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in IMG_EXTS]

    if not images:
        print(f"⚠️  {RAW_DIR} 에 이미지가 없습니다.")
        print(f"   이미지를 {RAW_DIR.resolve()} 폴더에 넣고 다시 실행하세요.")
        return

    # 캡션이 없는 이미지만 처리 (--overwrite 없으면)
    to_process = []
    for img_path in sorted(images):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists() or overwrite:
            to_process.append(img_path)
        else:
            print(f"  SKIP (캡션 있음): {img_path.name}")

    if not to_process:
        print("\n모든 이미지에 캡션이 이미 있습니다.")
        print("덮어쓰려면: python add_captions.py --overwrite")
        return

    print(f"\n총 {len(images)}장 중 {len(to_process)}장 캡션 생성 예정\n")

    processor, model, device = load_blip()

    results = []
    for img_path in tqdm(to_process, desc="캡션 생성"):
        img = Image.open(img_path).convert("RGB")
        base = generate_caption(img, processor, model, device)
        style = classify_style(img_path.name)
        full_caption = enrich_caption(base, style)
        results.append((img_path, full_caption))

        if preview:
            print(f"\n[{img_path.name}]")
            print(f"  스타일: {style}")
            print(f"  캡션: {full_caption}")

    if preview:
        print("\n⚠️  미리보기 모드 - 저장하지 않았습니다.")
        print("저장하려면: python add_captions.py")
        return

    # 캡션 저장
    print("\n캡션 저장 중...")
    for img_path, caption in results:
        txt_path = img_path.with_suffix(".txt")
        txt_path.write_text(caption, encoding="utf-8")

    print(f"\n완료! {len(results)}개 캡션 저장됨 → {RAW_DIR.resolve()}")
    print("\n캡션 예시:")
    for img_path, caption in results[:3]:
        print(f"  [{img_path.name}]")
        print(f"  {caption[:100]}...")
    print(f"\n다음 단계: python prepare_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="동화 삽화 이미지 자동 캡션 생성")
    parser.add_argument("--preview", action="store_true", help="캡션 미리보기만 (저장 안 함)")
    parser.add_argument("--overwrite", action="store_true", help="기존 캡션 덮어쓰기")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    run(preview=args.preview, overwrite=args.overwrite)
