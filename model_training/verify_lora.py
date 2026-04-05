"""Verify a trained SDXL LoRA by comparing base vs LoRA generations."""

import argparse
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont

try:
    from fairytale_lora.model_training.train_config import (
        DATASET,
        MODEL,
        PATHS,
        PROMPTS,
        VALIDATION_PROMPTS,
    )
except ModuleNotFoundError:
    from model_training.train_config import (
        DATASET,
        MODEL,
        PATHS,
        PROMPTS,
        VALIDATION_PROMPTS,
    )


def resolve_lora_source(lora_path: str | None) -> Path | None:
    """Prefer the diffusers adapter directory saved alongside the safetensors file."""
    if not lora_path:
        return None

    path = Path(lora_path)
    if path.is_dir():
        return path

    sibling_dir = path.parent / "final_unet_lora"
    if sibling_dir.exists():
        print(f"LoRA safetensors 대신 diffusers 어댑터를 사용합니다: {sibling_dir}")
        return sibling_dir

    return path


def load_pipeline(lora_path: str | None = None, lora_scale: float = 1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"베이스 모델 로딩: {MODEL['base_model_id']}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL["base_model_id"],
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.vae.enable_slicing()

    resolved_path = resolve_lora_source(lora_path)
    if resolved_path:
        if resolved_path.exists():
            print(f"LoRA 로딩: {resolved_path}")
            if resolved_path.is_dir():
                peft_unet = PeftModel.from_pretrained(pipe.unet, str(resolved_path))
                for module in peft_unet.modules():
                    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                        for adapter_key in list(module.scaling.keys()):
                            module.scaling[adapter_key] = lora_scale
                pipe.unet = peft_unet.merge_and_unload()
            else:
                pipe.load_lora_weights(str(resolved_path.parent), weight_name=resolved_path.name)
            print("LoRA 로딩 완료!")
        else:
            print(f"[경고] LoRA 경로가 없습니다: {resolved_path}")
            print("  베이스 모델만으로 검증을 진행합니다.")

    return pipe, device


def generate_image(pipe, prompt: str, seed: int, device: str) -> Image.Image:
    generator = torch.Generator(device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=PROMPTS["default_negative"],
        width=1024,
        height=1024,
        num_inference_steps=35,
        guidance_scale=8.0,
        generator=generator,
    )
    return result.images[0]


def make_comparison(before: Image.Image, after: Image.Image) -> Image.Image:
    w, h = before.size
    label_h = 50
    padding = 10

    canvas = Image.new("RGB", (w * 2 + padding * 3, h + label_h + padding * 2), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    canvas.paste(before, (padding, label_h))
    canvas.paste(after, (w + padding * 2, label_h))

    draw.rectangle([0, 0, w * 2 + padding * 3, label_h], fill=(20, 20, 20))
    draw.text((padding + w // 2 - 45, 12), "Base", fill=(220, 220, 220), font=font)
    draw.text((w + padding * 2 + w // 2 - 45, 12), "LoRA", fill=(120, 230, 120), font=font)
    draw.line(
        [(w + padding + padding // 2, 0), (w + padding + padding // 2, h + label_h)],
        fill=(80, 80, 80),
        width=2,
    )
    return canvas


def find_latest_lora() -> str | None:
    output_dir = Path(PATHS["output_dir"])

    final_dir = output_dir / "final_unet_lora"
    if final_dir.exists():
        return str(final_dir)

    safetensors = sorted(output_dir.glob("*.safetensors"))
    if safetensors:
        return str(safetensors[-1])

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if checkpoints:
        return str(checkpoints[-1])

    return None


def verify(args):
    output_dir = Path(PATHS["output_dir"]) / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [args.prompt] if args.prompt else VALIDATION_PROMPTS
    base_seeds = [42, 123, 777]
    seeds = [base_seeds[i % len(base_seeds)] + (i // len(base_seeds)) * 100 for i in range(len(prompts))]

    print("\n" + "=" * 60)
    print("  LoRA 학습 결과 검증")
    print("=" * 60)

    print("\n[1/2] 베이스 모델로 이미지 생성 중...")
    pipe_base, device = load_pipeline(lora_path=None)
    before_images = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
        print(f"  프롬프트 {i}: {prompt[:60]}...")
        img = generate_image(pipe_base, prompt, seed, device)
        before_images.append(img)
        img.save(output_dir / f"before_{i:02d}.png")

    del pipe_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_path = args.lora or find_latest_lora()
    if not lora_path:
        print("\n[오류] 학습된 LoRA를 찾지 못했습니다.")
        return

    print("\n[2/2] LoRA 적용 후 이미지 생성 중...")
    print(f"  LoRA: {lora_path}")

    trigger = DATASET["instance_prompt"]
    lora_prompts = [f"{trigger}, {p}" if trigger not in p else p for p in prompts]

    pipe_lora, device = load_pipeline(lora_path=lora_path, lora_scale=args.lora_scale)
    after_images = []
    for i, (prompt, seed) in enumerate(zip(lora_prompts, seeds), start=1):
        print(f"  프롬프트 {i}: {prompt[:60]}...")
        img = generate_image(pipe_lora, prompt, seed, device)
        after_images.append(img)
        img.save(output_dir / f"after_{i:02d}.png")

    del pipe_lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n비교 이미지 생성 중...")
    for i, (before, after) in enumerate(zip(before_images, after_images), start=1):
        comparison = make_comparison(before, after)
        save_path = output_dir / f"comparison_{i:02d}.png"
        comparison.save(save_path)
        print(f"  저장: {save_path}")

    print(f"\n검증 완료! 결과 폴더: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 학습 결과 검증")
    parser.add_argument("--lora", type=str, default=None, help="LoRA 파일 또는 디렉터리 경로")
    parser.add_argument("--prompt", type=str, default=None, help="테스트 프롬프트")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength for verification")
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
