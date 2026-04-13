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
from api.pipeline.config import BASE_MODELS, LORA_CONFIGS, OUTPUT_CONFIG


class FairytaleImageGenerator:
    """LoRA 기반 동화 삽화 이미지 생성기"""

    def __init__(
        self,
        base_model_key: str = "sdxl",
        lora_key: Optional[str] = None,
        device: Optional[str] = None,
        use_xformers: bool = False,
        low_memory_mode: bool = False,
        lora_scale_override: Optional[float] = None,
    ):
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

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        if model_type == "sdxl":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None,
            )
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None,
            )

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=True,
        )

        if self.lora_config:
            self._load_lora()

        if self.low_memory_mode:
            self.pipeline.enable_model_cpu_offload()
            print("  → 저메모리 모드 활성화 (CPU 오프로딩)")
        else:
            self.pipeline = self.pipeline.to(self.device)

        if self.use_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("  → xformers 최적화 활성화")
            except Exception:
                print("  → xformers 사용 불가 (설치 후 재시도)")

        if self.model_config["type"] == "sdxl":
            self.pipeline.enable_vae_slicing()

        print("모델 로딩 완료!\n")
        return self

    def _load_lora(self):
        """LoRA 가중치 로드 (diffusers 네이티브 방식)."""
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
        """PEFT 체크포인트의 키를 diffusers 형식으로 변환한다."""
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
        scale: float = 0.2,
    ) -> "FairytaleImageGenerator":
        """IP-Adapter를 로드한다."""
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
        lora_scale: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
    ) -> list[Image.Image]:
        """이미지 생성. Returns: (PIL Image 리스트, 소요시간)"""
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        w, h = self.model_config["default_size"]
        width = width or w
        height = height or h
        steps = num_inference_steps or self.model_config["num_inference_steps"]
        scale = guidance_scale or self.model_config["guidance_scale"]

        if auto_add_trigger and self.lora_config and self.lora_config.get("trigger_word"):
            trigger = self.lora_config["trigger_word"]
            if trigger.lower() not in prompt.lower():
                prompt = f"{trigger}, {prompt}"
                print(f"  트리거 단어 추가: '{trigger}'")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        lora_kwargs = {}
        if self.lora_config:
            effective_lora_scale = (
                lora_scale
                if lora_scale is not None
                else self.lora_scale_override
                if self.lora_scale_override is not None
                else self.lora_config["lora_scale"]
            )
            if not self.uses_merged_peft_lora:
                try:
                    self.pipeline.set_adapters(self.adapter_name, adapter_weights=effective_lora_scale)
                except Exception:
                    lora_kwargs["cross_attention_kwargs"] = {"scale": effective_lora_scale}

        _ip_scale_restored = False
        if self._ip_adapter_loaded:
            if ip_adapter_image is not None:
                if ip_adapter_scale is not None:
                    self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
                    _ip_scale_restored = True
                lora_kwargs["ip_adapter_image"] = ip_adapter_image
            else:
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

        if _ip_scale_restored:
            self.pipeline.set_ip_adapter_scale(self._ip_adapter_scale)

        return images, elapsed

    def _default_negative(self) -> str:
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
        """생성된 이미지 저장"""
        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        timestamp = int(time.time())

        output_size = OUTPUT_CONFIG.get("output_size")

        for i, img in enumerate(images):
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
