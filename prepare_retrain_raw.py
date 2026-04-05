"""raw 원본 이미지 46장으로 LoRA 재학습 데이터셋을 준비한다.

변경 사항:
  - raw/*.jpg|*.png → retrain_raw_only/*.png (PNG 통일)
  - 캡션: ftbookstyle 트리거 + 통일된 스타일 키워드 + 이미지별 내용 묘사
  - 스타일 관련 문구 제거 (LoRA가 학습할 영역이므로 캡션에서 배제)
"""

import re
import shutil
from pathlib import Path
from PIL import Image

RAW_DIR = Path(__file__).parent / "dataset" / "raw"
OUT_DIR = Path(__file__).parent / "dataset" / "retrain_raw_only"

# LoRA가 학습할 스타일 — 캡션에는 최소한의 트리거만 남기고
# 나머지 스타일은 이미지 자체에서 학습하도록 함
STYLE_PREFIX = "ftbookstyle, child picture book illustration"

# 캡션에서 제거할 스타일 관련 문구 (LoRA가 이미지에서 직접 학습해야 할 부분)
REMOVE_PATTERNS = [
    r"children'?s book illustration,?\s*",
    r"storybook art,?\s*",
    r"soft colors,?\s*",
    r"anime-influenced style[^,]*,?\s*",
    r"watercolor texture,?\s*",
    r"cinematic atmospheric lighting,?\s*",
    r"soft painterly brushstroke[^,]*,?\s*",
    r"Korean traditional style,?\s*",
    r"soft watercolor atmosphere,?\s*",
]


def clean_caption(raw_caption: str) -> str:
    """원본 캡션에서 스타일 문구를 제거하고 내용 묘사만 남긴다."""
    text = raw_caption.strip()
    # BOM 제거
    text = text.lstrip("\ufeff")

    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 앞뒤 쉼표/공백 정리
    text = re.sub(r"^[\s,]+", "", text)
    text = re.sub(r"[\s,]+$", "", text)
    text = re.sub(r",\s*,", ",", text)  # 이중 쉼표 제거
    text = re.sub(r"\s+", " ", text).strip()

    return f"{STYLE_PREFIX}, {text}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg"}
    raw_images = sorted(
        f for f in RAW_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    )

    print(f"원본 이미지: {len(raw_images)}장")
    print(f"출력 디렉토리: {OUT_DIR}")

    for img_path in raw_images:
        stem = img_path.stem
        out_img = OUT_DIR / f"{stem}.png"
        out_txt = OUT_DIR / f"{stem}.txt"

        # 이미지 변환 (JPG → PNG 통일)
        if img_path.suffix.lower() == ".png":
            shutil.copy2(img_path, out_img)
        else:
            img = Image.open(img_path).convert("RGB")
            img.save(out_img, "PNG")

        # 캡션 변환
        raw_txt = img_path.with_suffix(".txt")
        if raw_txt.exists():
            raw_caption = raw_txt.read_text(encoding="utf-8").strip()
            new_caption = clean_caption(raw_caption)
        else:
            new_caption = f"{STYLE_PREFIX}, fairytale illustration"

        out_txt.write_text(new_caption, encoding="utf-8")
        print(f"  {stem}: {new_caption[:80]}...")

    print(f"\n완료: {len(raw_images)}장 준비됨")


if __name__ == "__main__":
    main()
