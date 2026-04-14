"""우리 동화 삽화 데이터셋의 텍스트 임베딩을 사전 계산한다.

HuggingFace datasets의 imagefolder 형식과 동일한 방식으로
이미지를 로드하고 해시를 계산하여 학습 스크립트와 호환되게 한다.

실행:
    cd fairytale_lora
    python model_training/compute_embeddings.py

출력: model_training/embeddings.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm
from transformers import T5EncoderModel

from diffusers import FluxPipeline

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_SEQ_LENGTH = 77
DATA_DIR = str(Path(__file__).resolve().parent.parent / "dataset" / "flux_train")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "embeddings.parquet")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def generate_image_hash(image):
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_pipeline():
    """텍스트 인코더만 로드 (transformer/vae 제외, VRAM 절약)."""
    from transformers import BitsAndBytesConfig as TransformersBnbConfig
    quantization_config = TransformersBnbConfig(load_in_8bit=True)
    text_encoder = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", quantization_config=quantization_config, device_map="auto"
    )
    pipeline = FluxPipeline.from_pretrained(
        MODEL_ID, text_encoder_2=text_encoder, transformer=None, vae=None, device_map="balanced"
    )
    return pipeline


@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    for prompt in tqdm(prompts, desc="임베딩 계산 중"):
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=max_sequence_length)
        all_prompt_embeds.append(prompt_embeds)
        all_pooled_prompt_embeds.append(pooled_prompt_embeds)
        all_text_ids.append(text_ids)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"최대 VRAM 사용: {max_memory:.3f} GB")
    return all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids


def run(args):
    print(f"데이터 디렉토리: {args.data_dir}")

    # imagefolder 형식으로 로드 — 학습 스크립트와 동일한 방식
    dataset = load_dataset("imagefolder", data_dir=args.data_dir, split="train")
    print(f"이미지 수: {len(dataset)}")

    # 캡션 파일에서 프롬프트 읽기
    image_prompts = {}
    data_path = Path(args.data_dir)
    for sample in dataset:
        image = sample["image"]
        image_hash = generate_image_hash(image)

        # imagefolder는 파일명 기반으로 캡션 매칭
        # sample에 text 필드가 없으면 파일명에서 캡션 파일 찾기
        if "text" in sample and sample["text"]:
            caption = sample["text"]
        else:
            # 이미지 파일명으로 캡션 파일 찾기
            img_filename = sample.get("image_file_path", "") or ""
            if not img_filename:
                # dataset의 파일명 추출 시도
                caption = "A children's picture book illustration."
            else:
                caption_path = Path(img_filename).with_suffix(".txt")
                if caption_path.exists():
                    caption = caption_path.read_text(encoding="utf-8").strip()
                else:
                    caption = "A children's picture book illustration."

        image_prompts[image_hash] = caption

    # 캡션 파일 직접 매칭 (imagefolder가 text 필드를 안 줄 경우 대비)
    png_files = sorted(data_path.glob("*.png"))
    if len(image_prompts) > 0 and all(v == "A children's picture book illustration." for v in image_prompts.values()):
        print("  imagefolder에서 캡션을 못 읽음 — 직접 매칭...")
        image_prompts = {}
        for img_file in png_files:
            image = dataset.features  # dummy
            # 다시 이미지 로드해서 해시 매칭
        # 간단한 방식: 순서대로 매칭
        image_prompts = {}
        for i, sample in enumerate(dataset):
            image = sample["image"]
            image_hash = generate_image_hash(image)
            # 파일 순서가 동일하다고 가정
            if i < len(png_files):
                caption_path = png_files[i].with_suffix(".txt")
                if caption_path.exists():
                    caption = caption_path.read_text(encoding="utf-8").strip()
                else:
                    caption = "A children's picture book illustration."
            else:
                caption = "A children's picture book illustration."
            image_prompts[image_hash] = caption

    all_prompts = list(image_prompts.values())
    print(f"캡션 매칭 완료: {len(all_prompts)}개")
    # 샘플 확인
    for i, (h, p) in enumerate(image_prompts.items()):
        if i < 2:
            print(f"  [{i}] hash={h[:16]}... caption={p[:80]}...")

    print(f"\n파이프라인 로드 중 (텍스트 인코더만)...")
    pipeline = load_pipeline()

    all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )

    data = []
    for i, (image_hash, _) in enumerate(image_prompts.items()):
        data.append((image_hash, all_prompt_embeds[i], all_pooled_prompt_embeds[i], all_text_ids[i]))

    # DataFrame 생성
    embedding_cols = ["prompt_embeds", "pooled_prompt_embeds", "text_ids"]
    df = pd.DataFrame(data, columns=["image_hash"] + embedding_cols)

    # 텐서 → 리스트 변환 (bfloat16 → float32 → numpy)
    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: x.cpu().to(torch.float32).numpy().flatten().tolist())

    df.to_parquet(args.output_path)
    print(f"\n임베딩 저장 완료: {args.output_path}")
    print(f"  행: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX 학습용 텍스트 임베딩 사전 계산")
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    run(args)
