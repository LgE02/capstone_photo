"""
LoRA 체크포인트별 이미지 생성 비교 스크립트
각 체크포인트(100~500)로 동일 프롬프트 생성 → 품질 비교
"""

import torch
import time
from pathlib import Path
from diffusers import FluxPipeline
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

OUTPUT_DIR = Path("outputs/lora_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_BASE = Path("lora_output/flux_lora")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

CHECKPOINTS = [
    ("no_lora",      None),
    ("checkpoint-100", LORA_BASE / "checkpoint-100" / "pytorch_lora_weights.safetensors"),
    ("checkpoint-200", LORA_BASE / "checkpoint-200" / "pytorch_lora_weights.safetensors"),
    ("checkpoint-300", LORA_BASE / "checkpoint-300" / "pytorch_lora_weights.safetensors"),
    ("checkpoint-400", LORA_BASE / "checkpoint-400" / "pytorch_lora_weights.safetensors"),
    ("checkpoint-500", LORA_BASE / "checkpoint-500" / "pytorch_lora_weights.safetensors"),
    ("final",          LORA_BASE / "pytorch_lora_weights.safetensors"),
]

TEST_PROMPTS = [
    {
        "name": "hanok_scene",
        "prompt": (
            "A children's book illustration in soft painterly style. "
            "A young boy in colorful Korean hanbok stands in front of a hanok house, "
            "looking up at the sky with a joyful expression, "
            "warm golden afternoon light, glowing particles in the air."
        ),
    },
    {
        "name": "forest_scene",
        "prompt": (
            "A children's book illustration in soft painterly style. "
            "A small fluffy rabbit sitting under a large tree in an enchanted forest, "
            "holding a glowing lantern, looking curious and excited, "
            "fireflies floating around, warm teal and golden tones."
        ),
    },
]


def load_pipeline():
    print("Loading FLUX.1-schnell with NF4...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", quantization_config=nf4_config
    )
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("Pipeline ready.\n")
    return pipe


def generate(pipe, prompt, lora_path, name, seed=42):
    if lora_path and Path(lora_path).exists():
        pipe.load_lora_weights(str(lora_path))

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        image = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
    elapsed = time.time() - start

    out_path = OUTPUT_DIR / f"{name}.png"
    image.save(out_path)
    print(f"  [{name}] {elapsed:.1f}초 → {out_path}")

    if lora_path:
        pipe.unload_lora_weights()

    return image


def main():
    pipe = load_pipeline()

    for prompt_info in TEST_PROMPTS:
        print(f"\n=== {prompt_info['name']} ===")
        for ckpt_name, lora_path in CHECKPOINTS:
            filename = f"{prompt_info['name']}_{ckpt_name}"
            generate(pipe, prompt_info["prompt"], lora_path, filename)

    print(f"\n완료. 결과: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
