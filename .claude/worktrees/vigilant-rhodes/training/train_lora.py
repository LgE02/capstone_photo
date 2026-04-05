"""Train an SDXL LoRA for the Fairytale illustration dataset."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from fairytale_lora.training.train_config import (
    DATASET,
    LORA_NETWORK,
    MODEL,
    OPTIMIZATION,
    PATHS,
    STYLE_HINTS,
    TRAINING,
    VALIDATION_PROMPTS,
)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def save_training_metadata(output_dir: Path, max_steps: int):
    metadata = {
        "model": MODEL,
        "dataset": DATASET,
        "training": {**TRAINING, "effective_max_train_steps": max_steps},
        "lora_network": LORA_NETWORK,
        "optimization": OPTIMIZATION,
        "validation_prompts": VALIDATION_PROMPTS,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class FairytaleDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer_1, tokenizer_2, resolution: int):
        self.data_dir = Path(data_dir)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution

        self.samples = []
        for img_path in sorted(self.data_dir.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else DATASET["instance_prompt"]
            self.samples.append((img_path, self._enhance_caption(img_path, caption)))

        if not self.samples:
            raise ValueError(f"데이터셋이 비어 있습니다: {data_dir}\n먼저 python prepare_dataset.py 를 실행하세요.")

        print(f"데이터셋 로드: {len(self.samples)}장")

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip() if DATASET["random_flip"] else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _enhance_caption(self, img_path: Path, caption: str) -> str:
        if self.data_dir.name == "style_core":
            return caption

        name = img_path.stem.lower()
        hints = [STYLE_HINTS["global"], STYLE_HINTS["storybook_character"]]

        if name.startswith("traditional"):
            if name in {"traditional1", "traditional2", "traditional4", "traditional5", "traditional7", "traditional8"}:
                hints.append(STYLE_HINTS["korean_traditional"])
        elif name.startswith("western"):
            hints.append(STYLE_HINTS["western_fairytale"])
        elif name.startswith("animal"):
            hints.append(STYLE_HINTS["animal_fairytale"])

        merged = ", ".join([caption, *hints])
        return merged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        tokens_1 = self.tokenizer_1(
            caption,
            max_length=self.tokenizer_1.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = self.tokenizer_2(
            caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids_1": tokens_1.input_ids.squeeze(0),
            "input_ids_2": tokens_2.input_ids.squeeze(0),
            "caption": caption,
        }


@torch.no_grad()
def generate_validation_samples(
    unet,
    vae,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    scheduler_config,
    output_dir: str,
    step: int,
    device: str,
):
    print(f"\n[검증 샘플 이미지 생성 중] step {step}")
    sample_dir = Path(output_dir) / "samples" / f"step_{step:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=DDPMScheduler.from_config(scheduler_config),
    ).to(device)

    for i, prompt in enumerate(VALIDATION_PROMPTS[:2], start=1):
        image = pipe(
            prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]
        out_path = sample_dir / f"sample_{i:02d}.png"
        image.save(out_path)
        print(f"  저장: {out_path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if TRAINING["mixed_precision"] == "bf16" else torch.float16
    max_steps = args.max_steps or TRAINING["max_train_steps"]
    warmup_steps = min(TRAINING["lr_warmup_steps"], max_steps)

    seed_everything(TRAINING["seed"])

    print("\n" + "=" * 60)
    print("  SDXL LoRA 파인튜닝 시작")
    print("=" * 60)
    print(f"  디바이스: {device} | 정밀도: {TRAINING['mixed_precision']}")
    print(f"  베이스 모델: {MODEL['base_model_id']}")
    print(f"  LoRA rank: {LORA_NETWORK['rank']} | 배치: {TRAINING['train_batch_size']}")
    print()

    print("모델 로딩 중...")
    model_id = MODEL["base_model_id"]

    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(MODEL["vae_model_id"], torch_dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    print("LoRA 네트워크 구성 중...")
    lora_config = LoraConfig(
        r=LORA_NETWORK["rank"],
        lora_alpha=LORA_NETWORK["alpha"],
        target_modules=LORA_NETWORK["target_modules"],
        lora_dropout=LORA_NETWORK["dropout"],
        bias=LORA_NETWORK["bias"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if TRAINING["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    if OPTIMIZATION["enable_xformers"]:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("  xformers 활성화")
        except Exception:
            print("  xformers 비활성화 (설치 필요)")

    if OPTIMIZATION["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = FairytaleDataset(
        data_dir=PATHS["train_data_dir"],
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        resolution=DATASET["resolution"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING["train_batch_size"],
        shuffle=True,
        num_workers=OPTIMIZATION["dataloader_num_workers"],
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=TRAINING["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / TRAINING["gradient_accumulation_steps"])
    num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        TRAINING["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        num_cycles=1,
    )

    output_dir = Path(args.output or PATHS["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_metadata(output_dir, max_steps)

    print(f"\n학습 시작: 총 {max_steps} 스텝 ({num_epochs} 에포크)")
    print(f"저장 위치: {output_dir.resolve()}\n")

    global_step = 0
    start_time = time.time()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and TRAINING["mixed_precision"] == "fp16"))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        unet.train()
        accum_steps = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids_1 = batch["input_ids_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda" and TRAINING["mixed_precision"] != "no"),
                dtype=dtype,
            ):
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    enc1_out = text_encoder_1(input_ids_1, output_hidden_states=True)
                    enc2_out = text_encoder_2(input_ids_2, output_hidden_states=True)
                    prompt_embeds = torch.concat(
                        [enc1_out.hidden_states[-2], enc2_out.hidden_states[-2]],
                        dim=-1,
                    )
                    pooled_prompt_embeds = enc2_out[0]

                add_time_ids = torch.tensor(
                    [[DATASET["resolution"], DATASET["resolution"], 0, 0, DATASET["resolution"], DATASET["resolution"]]],
                    dtype=dtype,
                    device=device,
                ).repeat(bsz, 1)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                target = (
                    noise
                    if noise_scheduler.config.prediction_type == "epsilon"
                    else noise_scheduler.get_velocity(latents, noise, timesteps)
                )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / TRAINING["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps % TRAINING["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                eta = (max_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * TRAINING['gradient_accumulation_steps']:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "ETA": f"{eta/60:.1f}min",
                    }
                )

                if global_step % TRAINING["checkpointing_steps"] == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    unet.save_pretrained(ckpt_dir)
                    print(f"\n  체크포인트 저장: {ckpt_dir}")

                if global_step >= max_steps:
                    break

        if (epoch + 1) % TRAINING["validation_epochs"] == 0:
            generate_validation_samples(
                unet=unet.base_model.model,
                vae=vae,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                scheduler_config=noise_scheduler.config,
                output_dir=str(output_dir),
                step=global_step,
                device=device,
            )

        if global_step >= max_steps:
            break

    final_path = output_dir / "fairytale_lora_final.safetensors"
    unet.save_pretrained(output_dir / "final_unet_lora")

    try:
        from safetensors.torch import save_file

        lora_state_dict = {k: v for k, v in unet.state_dict().items() if "lora" in k}
        save_file(lora_state_dict, final_path)
        print(f"\n최종 LoRA 저장: {final_path}")
    except Exception as exc:
        print(f"\n[주의] safetensors 저장 실패: {exc}")
        print(f"  diffusers 형식으로 저장됨: {output_dir / 'final_unet_lora'}")

    total_time = time.time() - start_time
    print(f"\n학습 완료! 총 소요 시간: {total_time / 60:.1f}분")
    print("다음 단계: python verify_lora.py")


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA 파인튜닝")
    parser.add_argument("--max-steps", type=int, default=None, help="최대 학습 스텝 수")
    parser.add_argument("--output", type=str, default=None, help="출력 디렉터리")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
