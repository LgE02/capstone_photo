"""
FLUX.2-klein-4B 기반 동화 삽화 이미지 생성기 (api_klein 독립 복사본)
- Photoroom FP8 변환 버전 사용 (bf16 fallback 포함)
- 참조 이미지 기반 캐릭터 일관성 유지
- guidance_scale=1.0, num_inference_steps=4
- Qwen3 텍스트 인코더 (최대 40,960 토큰)
"""

import os
import time
import torch
from PIL import Image
from typing import Optional

from api_klein.pipeline.config import OUTPUT_CONFIG

TRANSFORMER_REPO = "Photoroom/FLUX.2-klein-4b-fp8-diffusers"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"


class KleinImageGenerator:
    """FLUX.2-klein-4B 기반 동화 삽화 이미지 생성기."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline = None
        self._character_reference: Optional[Image.Image] = None

        print(f"[KleinGenerator] 디바이스: {self.device}")
        print(f"[KleinGenerator] 베이스 모델: {BASE_REPO}")

    def load(self) -> "KleinImageGenerator":
        """FLUX.2-klein-4B 모델 로드.

        BF16으로 로드 후 torchao로 FP8 변환 시도.
        torchao 변환 실패 시 BF16 그대로 사용 (fallback).
        """
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        print("\nFLUX.2-klein-4B 로딩 중 (BF16 → FP8 변환 시도)...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            TRANSFORMER_REPO,
            subfolder="transformer_bf16",
            torch_dtype=torch.bfloat16,
        )
        print("  → Transformer (BF16, ~7.7GB) 로드 완료")

        # torchao FP8 변환 시도 (성공 시 ~3.9GB로 줄어듦)
        try:
            from torchao.quantization import quantize_, float8_weight_only
            quantize_(transformer, float8_weight_only())
            print("  → torchao FP8 변환 완료 (~3.9GB)")
        except Exception as e:
            print(f"  → FP8 변환 실패, BF16 유지: {e}")

        self.pipeline = Flux2KleinPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()
        print("  → CPU offload 활성화 (T5/VAE/CLIP은 필요 시만 GPU)")
        print("FLUX.2-klein-4B 로딩 완료!\n")
        return self

    # ── 캐릭터 레퍼런스 관리 ────────────────────────────────────────────────

    def generate_character_reference(
        self,
        character_prompt: str,
        seed: int = 42,
    ) -> Image.Image:
        """
        캐릭터 레퍼런스 이미지 생성.
        동화 시작 전 한 번 호출해서 이후 모든 장면에 재사용.
        """
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        style_prefix = (
            "Children's picture book illustration, "
            "semi-painterly digital art with soft cel shading, "
            "chibi-style proportions: large round head, big glossy expressive eyes, rosy cheeks, "
            "warm color palette, soft storybook aesthetic, NOT realistic, NOT anime. "
        )
        full_prompt = (
            style_prefix + character_prompt
            + " Full body portrait, character centered on white background, clear details."
        )

        print(f"[KleinGenerator] 캐릭터 레퍼런스 생성 중...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipeline(
            prompt=full_prompt,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=generator,
        ).images[0]

        self._character_reference = result
        print(f"  → 캐릭터 레퍼런스 생성 완료")
        return result

    def set_character_reference(self, image: Image.Image) -> None:
        """외부에서 레퍼런스 이미지 직접 설정."""
        self._character_reference = image

    # ── 메인 생성 ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        use_reference: bool = True,
        num_images: int = 1,
    ) -> tuple[list[Image.Image], float]:
        """
        장면 삽화 생성.
        use_reference=True이고 캐릭터 레퍼런스가 있으면 참조 이미지 적용.
        Returns: (PIL Image 리스트, 소요시간)
        """
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=generator,
            num_images_per_prompt=num_images,
        )

        if use_reference and self._character_reference is not None:
            kwargs["image"] = [self._character_reference]

        print(f"[KleinGenerator] 이미지 생성 중...")
        print(f"  프롬프트: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"  크기: {width}x{height} | 레퍼런스: {'✅' if 'image' in kwargs else '❌'}")

        start_time = time.time()
        output = self.pipeline(**kwargs)
        elapsed = time.time() - start_time

        images = output.images
        print(f"  완료! ({elapsed:.1f}초, {len(images)}장)")
        return images, elapsed

    # ── 저장 ────────────────────────────────────────────────────────────────

    def save_images(
        self,
        images: list[Image.Image],
        output_dir: Optional[str] = None,
        prefix: str = "fairytale_klein",
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

            filename = f"{prefix}_{timestamp}_{i+1:02d}.png"
            path = os.path.join(output_dir, filename)
            img.save(path, format="PNG")
            saved_paths.append(path)
            size_info = f"{output_size[0]}x{output_size[1]}" if output_size else f"{img.width}x{img.height}"
            print(f"  저장: {path} ({size_info})")

        return saved_paths

    def save_character_reference(self, output_dir: Optional[str] = None) -> Optional[str]:
        """캐릭터 레퍼런스 이미지 저장."""
        if self._character_reference is None:
            return None

        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, f"character_reference_{int(time.time())}.png")
        self._character_reference.save(path, format="PNG")
        print(f"  캐릭터 레퍼런스 저장: {path}")
        return path

    def unload(self) -> None:
        """모델 언로드 (메모리 해제)."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[KleinGenerator] 모델 언로드 완료.")
