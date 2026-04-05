"""
동화 삽화 이미지 생성기 - 핵심 클래스
LoRA 가중치를 적용한 Stable Diffusion 기반 이미지 생성
"""

import os
import time
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, Union
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from peft import PeftModel
from fairytale_lora.generation.config import BASE_MODELS, LORA_CONFIGS, OUTPUT_CONFIG


class FairytaleImageGenerator:
    """
    LoRA 기반 동화 삽화 이미지 생성기

    사용 예시:
        gen = FairytaleImageGenerator("sdxl", "storybook_sdxl")
        images = gen.generate("a magical forest with fairies")
    """

    def __init__(
        self,
        base_model_key: str = "sdxl",
        lora_key: Optional[str] = None,
        device: Optional[str] = None,
        use_xformers: bool = False,
        low_memory_mode: bool = False,
        lora_scale_override: Optional[float] = None,
    ):
        """
        Args:
            base_model_key: config.py의 BASE_MODELS 키 (sdxl / sd15 / sd21)
            lora_key: config.py의 LORA_CONFIGS 키 (None이면 LoRA 미사용)
            device: cuda / cpu / mps (None이면 자동 감지)
            use_xformers: xformers 메모리 최적화 활성화 (xformers 설치 필요)
            low_memory_mode: VRAM 부족 시 CPU 오프로딩 활성화
        """
        self.base_model_key = base_model_key
        self.lora_key = lora_key
        self.model_config = BASE_MODELS[base_model_key]
        self.lora_config = LORA_CONFIGS.get(lora_key) if lora_key else None
        self.pipeline = None
        self.use_xformers = use_xformers
        self.low_memory_mode = low_memory_mode
        self.lora_scale_override = lora_scale_override
        self.adapter_name = "fairytale_style"
        self.uses_merged_peft_lora = False
        self.adapter_name = "fairytale_style"

        # 디바이스 자동 감지
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[FairytaleGenerator] 디바이스: {self.device}")
        print(f"[FairytaleGenerator] 베이스 모델: {self.model_config['model_id']}")
        if self.lora_config:
            print(f"[FairytaleGenerator] LoRA: {self.lora_config['description']}")
            if self.lora_scale_override is not None:
                print(f"[FairytaleGenerator] LoRA scale override: {self.lora_scale_override}")

    def load(self) -> "FairytaleImageGenerator":
        """모델 및 LoRA 가중치 로드"""
        print(f"\n모델 로딩 중... (첫 실행 시 다운로드 발생)")

        model_id = self.model_config["model_id"]
        model_type = self.model_config["type"]

        # dtype 결정 (CPU는 float32)
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        # ─── 파이프라인 로드 ───────────────────────────────
        if model_type == "sdxl":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None,
            )
        else:
            # SD 1.5, SD 2.1
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None,  # 동화 컨텐츠 필터 비활성화
            )

        # ─── 스케줄러 설정 (DPM++ 2M - 고품질/빠름) ────────
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=True,
        )

        # ─── LoRA 로드 ────────────────────────────────────
        if self.lora_config:
            self._load_lora()

        # ─── 디바이스 이동 ─────────────────────────────────
        if self.low_memory_mode:
            # VRAM 부족 시: CPU와 GPU 간 자동 오프로딩
            self.pipeline.enable_model_cpu_offload()
            print("  → 저메모리 모드 활성화 (CPU 오프로딩)")
        else:
            self.pipeline = self.pipeline.to(self.device)

        # ─── xformers 최적화 ─────────────────────────────
        if self.use_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("  → xformers 최적화 활성화")
            except Exception:
                print("  → xformers 사용 불가 (설치 후 재시도)")

        # SDXL VAE 슬라이싱 (메모리 절약)
        if self.model_config["type"] == "sdxl":
            self.pipeline.enable_vae_slicing()

        print("모델 로딩 완료!\n")
        return self

    def _load_lora(self):
        """LoRA 가중치 로드 (HuggingFace Hub 또는 로컬 파일)"""
        lora = self.lora_config
        print(f"  LoRA 로딩 중: {lora['description']}")

        try:
            if lora["repo_id"] is not None:
                # HuggingFace Hub에서 로드
                load_kwargs = {
                    "pretrained_model_name_or_path_or_dict": lora["repo_id"],
                    "adapter_name": self.adapter_name,
                }
                if lora["weight_name"]:
                    load_kwargs["weight_name"] = lora["weight_name"]
                self.pipeline.load_lora_weights(**load_kwargs)
            else:
                # 로컬 파일에서 로드
                local_path = lora["weight_name"]
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"LoRA 파일을 찾을 수 없습니다: {local_path}")
                if os.path.isdir(local_path):
                    adapter_file = os.path.join(local_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_file):
                        scaled_lora = (
                            self.lora_scale_override
                            if self.lora_scale_override is not None
                            else lora["lora_scale"]
                        )
                        peft_unet = PeftModel.from_pretrained(self.pipeline.unet, local_path)
                        for module in peft_unet.modules():
                            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                                for adapter_key in list(module.scaling.keys()):
                                    module.scaling[adapter_key] = scaled_lora
                        self.pipeline.unet = peft_unet.merge_and_unload()
                        self.uses_merged_peft_lora = True
                    else:
                        self.pipeline.load_lora_weights(
                            local_path,
                            adapter_name=self.adapter_name,
                        )
                else:
                    self.pipeline.load_lora_weights(local_path, adapter_name=self.adapter_name)

            print(f"  LoRA 로딩 완료 (scale: {lora['lora_scale']})")

        except Exception as e:
            print(f"  [경고] LoRA 로딩 실패: {e}")
            print("  LoRA 없이 진행합니다.")
            self.lora_config = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        auto_add_trigger: bool = True,
    ) -> list[Image.Image]:
        """
        이미지 생성

        Args:
            prompt: 생성할 이미지 설명 (영문 권장)
            negative_prompt: 제외할 요소
            width/height: 이미지 크기 (None이면 모델 기본값)
            num_inference_steps: 추론 스텝 수 (많을수록 품질↑, 속도↓)
            guidance_scale: 프롬프트 충실도 (7~9 권장)
            seed: 재현 가능한 결과를 위한 시드값
            num_images: 생성할 이미지 수
            auto_add_trigger: LoRA 트리거 단어 자동 추가
        Returns:
            PIL Image 리스트
        """
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        # 기본값 적용
        w, h = self.model_config["default_size"]
        width = width or w
        height = height or h
        steps = num_inference_steps or self.model_config["num_inference_steps"]
        scale = guidance_scale or self.model_config["guidance_scale"]

        # LoRA 트리거 단어 자동 추가
        if auto_add_trigger and self.lora_config and self.lora_config.get("trigger_word"):
            trigger = self.lora_config["trigger_word"]
            if trigger.lower() not in prompt.lower():
                prompt = f"{trigger}, {prompt}"
                print(f"  트리거 단어 추가: '{trigger}'")

        # 시드 설정
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # LoRA 스케일 적용
        lora_kwargs = {}
        if self.lora_config:
            lora_scale = (
                self.lora_scale_override
                if self.lora_scale_override is not None
                else self.lora_config["lora_scale"]
            )
            if not self.uses_merged_peft_lora:
                try:
                    self.pipeline.set_adapters(self.adapter_name, adapter_weights=lora_scale)
                except Exception:
                    lora_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        print(f"이미지 생성 중...")
        print(f"  프롬프트: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  크기: {width}x{height} | 스텝: {steps} | CFG: {scale}")

        start_time = time.time()

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or self._default_negative(),
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
            num_images_per_prompt=num_images,
            **lora_kwargs,
        )

        elapsed = time.time() - start_time
        images = output.images
        print(f"  완료! ({elapsed:.1f}초, {len(images)}장)")

        return images, elapsed

    def _default_negative(self) -> str:
        """기본 네거티브 프롬프트"""
        return (
            "ugly, blurry, low quality, deformed, distorted face, bad anatomy, "
            "extra fingers, extra limbs, watermark, text, signature, logo, "
            "realistic photo, 3d render, dark, scary, horror, violence, "
            "busy background, detailed background, cluttered scenery, "
            "Chinese clothing, hanfu, qipao, cheongsam, Chinese palace"
        )

    def save_images(
        self,
        images: list[Image.Image],
        output_dir: str = None,
        prefix: str = "fairytale",
    ) -> list[str]:
        """
        생성된 이미지 저장

        Returns:
            저장된 파일 경로 리스트
        """
        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        timestamp = int(time.time())

        # 출력 크기 설정 (config.py OUTPUT_CONFIG["output_size"] 에서 읽음)
        output_size = OUTPUT_CONFIG.get("output_size")  # (width, height) 또는 None

        for i, img in enumerate(images):
            # 최종 출력 크기로 리사이즈 (None이면 원본 크기 유지)
            if output_size is not None:
                img = img.resize(output_size, Image.LANCZOS)

            model_tag = f"{self.base_model_key}"
            lora_tag = f"_{self.lora_key}" if self.lora_key else ""
            filename = f"{prefix}{lora_tag}_{model_tag}_{timestamp}_{i+1:02d}.png"
            path = os.path.join(output_dir, filename)
            img.save(path, format="PNG")
            saved_paths.append(path)
            size_info = f"{output_size[0]}x{output_size[1]}" if output_size else f"{img.width}x{img.height}"
            print(f"  저장: {path} ({size_info})")

        return saved_paths

    def unload(self):
        """모델 언로드 (메모리 해제)"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("모델 언로드 완료.")


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    cols: int = 2,
) -> Image.Image:
    """
    여러 이미지를 격자 형태로 합쳐 비교 이미지 생성

    Args:
        images: PIL Image 리스트
        labels: 각 이미지 레이블
        cols: 열 수
    Returns:
        합쳐진 PIL Image
    """
    from PIL import ImageDraw, ImageFont

    if not images:
        return None

    img_w, img_h = images[0].size
    label_h = 40
    rows = (len(images) + cols - 1) // cols

    grid = Image.new(
        "RGB",
        (img_w * cols, (img_h + label_h) * rows),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, cols)
        x = col * img_w
        y = row * (img_h + label_h)

        grid.paste(img, (x, y + label_h))
        draw.rectangle([x, y, x + img_w, y + label_h], fill=(50, 50, 80))
        draw.text((x + 8, y + 10), label, fill="white", font=font)

    return grid
