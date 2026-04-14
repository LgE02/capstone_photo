"""
FLUX.1 Schnell 기반 동화 삽화 이미지 생성기
- NF4 양자화 (bitsandbytes)로 GPU 12GB에서 구동
- 단일 T5 자연어 프롬프트
- guidance_scale=0.0, num_inference_steps=4
"""

import os
import time
import torch
from PIL import Image
from typing import Optional

from api.pipeline.config import BASE_MODELS, OUTPUT_CONFIG


class FairytaleImageGenerator:
    """FLUX.1 Schnell 기반 동화 삽화 이미지 생성기."""

    def __init__(
        self,
        base_model_key: str = "flux_schnell",
        device: Optional[str] = None,
        low_memory_mode: bool = True,
    ):
        self.base_model_key = base_model_key
        self.model_config = BASE_MODELS[base_model_key]
        self.pipeline = None
        self.low_memory_mode = low_memory_mode

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[FluxGenerator] 디바이스: {self.device}")
        print(f"[FluxGenerator] 베이스 모델: {self.model_config['model_id']}")

    def load(self) -> "FairytaleImageGenerator":
        """FLUX 모델 로드 (NF4 양자화 적용)."""
        from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig

        print(f"\nFLUX 모델 로딩 중 (NF4 양자화)...")

        model_id = self.model_config["model_id"]
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        # NF4 양자화 — DiT(transformer) 부분만 4비트로 로드
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=dtype,
        )

        self.pipeline = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype,
        )
        # transformer(NF4, ~3GB)는 GPU 상주, T5/CLIP/VAE는 필요할 때만 GPU 올림
        self.pipeline.enable_model_cpu_offload()
        print(f"  → NF4 양자화 적용 완료 (transformer ~3GB)")
        print(f"  → CPU offload: T5/CLIP은 필요 시에만 GPU 사용")

        print("FLUX 모델 로딩 완료!\n")
        return self

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images: int = 1,
    ) -> tuple[list[Image.Image], float]:
        """이미지 생성. Returns: (PIL Image 리스트, 소요시간)"""
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        w, h = self.model_config["default_size"]
        width = width or w
        height = height or h
        steps = num_inference_steps or self.model_config["num_inference_steps"]

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"이미지 생성 중... (FLUX Schnell)")
        print(f"  프롬프트: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"  크기: {width}x{height} | 스텝: {steps} | CFG: 0.0 (고정)")

        start_time = time.time()

        output = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
            num_images_per_prompt=num_images,
        )

        elapsed = time.time() - start_time
        images = output.images
        print(f"  완료! ({elapsed:.1f}초, {len(images)}장)")

        return images, elapsed

    def save_images(
        self,
        images: list[Image.Image],
        output_dir: Optional[str] = None,
        prefix: str = "fairytale",
    ) -> list[str]:
        """생성된 이미지 저장."""
        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        timestamp = int(time.time())
        output_size = OUTPUT_CONFIG.get("output_size")

        for i, img in enumerate(images):
            if output_size is not None:
                img = img.resize(output_size, Image.LANCZOS)

            filename = f"{prefix}_flux_{timestamp}_{i+1:02d}.png"
            path = os.path.join(output_dir, filename)
            img.save(path, format="PNG")
            saved_paths.append(path)
            size_info = f"{output_size[0]}x{output_size[1]}" if output_size else f"{img.width}x{img.height}"
            print(f"  저장: {path} ({size_info})")

        return saved_paths

    def unload(self) -> None:
        """모델 언로드 (메모리 해제)."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("FLUX 모델 언로드 완료.")
