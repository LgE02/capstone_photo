"""
동화 삽화 이미지 생성기 - 핵심 클래스
LoRA 가중치를 적용한 Stable Diffusion 기반 이미지 생성
IP-Adapter를 통한 참고 이미지 반영 지원
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
try:
    from fairytale_lora.runtime_pipeline.config import BASE_MODELS, LORA_CONFIGS, OUTPUT_CONFIG
except ModuleNotFoundError:
    from runtime_pipeline.config import BASE_MODELS, LORA_CONFIGS, OUTPUT_CONFIG


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
        self._ip_adapter_loaded = False
        self._ip_adapter_scale = 0.6

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
        """LoRA 가중치 로드 (HuggingFace Hub 또는 로컬 파일)

        IP-Adapter와의 호환성을 위해 diffusers 네이티브 방식으로 로드한다.
        PeftModel.merge_and_unload()는 UNet 가중치 차원을 변경하므로
        IP-Adapter의 Cross-Attention 레이어와 차원 불일치를 일으킨다.

        PEFT(accelerate) 학습 체크포인트는 키가 "base_model.model.*" 형식이므로
        diffusers가 기대하는 "unet.*" 형식으로 변환해서 로드한다.
        """
        lora = self.lora_config
        print(f"  LoRA 로딩 중: {lora['description']}")

        try:
            if lora["repo_id"] is not None:
                load_kwargs = {
                    "pretrained_model_name_or_path_or_dict": lora["repo_id"],
                    "adapter_name": self.adapter_name,
                }
                if lora["weight_name"]:
                    load_kwargs["weight_name"] = lora["weight_name"]
                self.pipeline.load_lora_weights(**load_kwargs)
            else:
                local_path = lora["weight_name"]
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"LoRA 파일을 찾을 수 없습니다: {local_path}")

                # PEFT 체크포인트 키를 diffusers 형식으로 변환
                state_dict = self._convert_peft_to_diffusers(local_path)
                self.pipeline.load_lora_weights(
                    state_dict,
                    adapter_name=self.adapter_name,
                )

            print(f"  LoRA 로딩 완료 (scale: {lora['lora_scale']})")

        except Exception as e:
            print(f"  [경고] LoRA 로딩 실패: {e}")
            print("  LoRA 없이 진행합니다.")
            self.lora_config = None

    @staticmethod
    def _convert_peft_to_diffusers(checkpoint_dir: str) -> dict:
        """PEFT 체크포인트의 키를 diffusers 형식으로 변환한다.

        PEFT: "base_model.model.down_blocks.1.attentions.0.to_q.lora_A.weight"
        diffusers: "unet.down_blocks.1.attentions.0.to_q.lora_A.weight"
        """
        from safetensors.torch import load_file

        safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"adapter_model.safetensors 를 찾을 수 없습니다: {safetensors_path}")

        peft_sd = load_file(safetensors_path)
        converted = {}
        for key, value in peft_sd.items():
            new_key = key.replace("base_model.model.", "unet.")
            converted[new_key] = value

        print(f"  PEFT→diffusers 키 변환: {len(converted)}개 가중치")
        return converted

    # ── IP-Adapter ────────────────────────────────────────────────────

    def load_ip_adapter(
        self,
        scale: float = 0.6,
    ) -> "FairytaleImageGenerator":
        """IP-Adapter를 로드한다.

        Args:
            scale: IP-Adapter 영향력 (0.0~1.0)
                   0.3~0.5: 참고 이미지 분위기만 반영
                   0.6~0.7: 캐릭터/스타일 강하게 반영 (권장)
                   0.8~1.0: 참고 이미지에 거의 종속
        """
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        if self._ip_adapter_loaded:
            self.pipeline.set_ip_adapter_scale(scale)
            print(f"[IP-Adapter] 스케일 변경: {scale}")
            return self

        print(f"[IP-Adapter] 로딩 중...")
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
        )
        self.pipeline.set_ip_adapter_scale(scale)
        self._ip_adapter_loaded = True
        self._ip_adapter_scale = scale

        # IP-Adapter 추가 시 VRAM이 부족해지므로 CPU 오프로딩 활성화
        # (이미 GPU에 있는 파이프라인을 CPU 오프로드 모드로 전환)
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
            print(f"[IP-Adapter] CPU 오프로딩 활성화 (VRAM 절약)")

        print(f"[IP-Adapter] 로드 완료 (scale={scale})")
        return self

    def unload_ip_adapter(self) -> None:
        """IP-Adapter를 언로드한다."""
        if self._ip_adapter_loaded and self.pipeline is not None:
            self.pipeline.unload_ip_adapter()
            self._ip_adapter_loaded = False
            print("[IP-Adapter] 언로드 완료")

    # ── 이미지 생성 ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        prompt_2: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        auto_add_trigger: bool = True,
        ip_adapter_image: Optional[Image.Image] = None,
    ) -> list[Image.Image]:
        """
        이미지 생성

        Args:
            prompt: 생성할 이미지 설명 — SDXL CLIP-L 인코더용 (캐릭터 묘사)
            negative_prompt: 제외할 요소
            prompt_2: SDXL CLIP-G 인코더용 (장면/배경 묘사, None이면 prompt와 동일)
            width/height: 이미지 크기 (None이면 모델 기본값)
            num_inference_steps: 추론 스텝 수 (많을수록 품질↑, 속도↓)
            guidance_scale: 프롬프트 충실도 (7~9 권장)
            seed: 재현 가능한 결과를 위한 시드값
            num_images: 생성할 이미지 수
            auto_add_trigger: LoRA 트리거 단어 자동 추가
            ip_adapter_image: IP-Adapter 참고 이미지 (None이면 미사용)
        Returns:
            (PIL Image 리스트, 소요시간)
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

        # IP-Adapter 참고 이미지
        # IP-Adapter 로드 후에는 UNet이 항상 image_embeds를 요구하므로
        # 참고 이미지가 없을 때는 더미 이미지 + scale=0으로 영향을 제거한다.
        _ip_scale_restored = False
        if self._ip_adapter_loaded:
            if ip_adapter_image is not None:
                lora_kwargs["ip_adapter_image"] = ip_adapter_image
            else:
                # 더미 검정 이미지 + scale=0 → 실질적으로 비활성화
                dummy = Image.new("RGB", (224, 224), color=(0, 0, 0))
                lora_kwargs["ip_adapter_image"] = dummy
                self.pipeline.set_ip_adapter_scale(0.0)
                _ip_scale_restored = True

        print(f"이미지 생성 중...")
        print(f"  프롬프트: {prompt}")
        if prompt_2:
            print(f"  프롬프트2: {prompt_2}")
        print(f"  네거티브: {(negative_prompt or '')[:120]}{'...' if len(negative_prompt or '') > 120 else ''}")
        ref_tag = " | IP-Adapter: ON" if ip_adapter_image is not None else ""
        dual_tag = " | dual-prompt" if prompt_2 else ""
        print(f"  크기: {width}x{height} | 스텝: {steps} | CFG: {scale}{ref_tag}{dual_tag}")

        start_time = time.time()

        # SDXL prompt_2 지원: 두 번째 텍스트 인코더에 장면 묘사 전달
        pipeline_kwargs = dict(
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
        if prompt_2 and self.model_config["type"] == "sdxl":
            pipeline_kwargs["prompt_2"] = prompt_2

        output = self.pipeline(**pipeline_kwargs)

        elapsed = time.time() - start_time
        images = output.images
        print(f"  완료! ({elapsed:.1f}초, {len(images)}장)")

        # IP-Adapter 스케일 복원
        if _ip_scale_restored:
            self.pipeline.set_ip_adapter_scale(self._ip_adapter_scale)

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
