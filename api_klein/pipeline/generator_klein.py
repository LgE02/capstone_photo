"""
FLUX.2-klein-4B 기반 동화 삽화 이미지 생성기 (api_klein 독립 복사본)
- Photoroom FP8 변환 버전 사용 (bf16 fallback 포함)
- 참조 이미지 기반 캐릭터 일관성 유지
- guidance_scale=1.0, num_inference_steps=4
- Qwen3 텍스트 인코더 (최대 40,960 토큰)
"""

import time
import torch
from PIL import Image
from typing import Optional


TRANSFORMER_REPO = "Photoroom/FLUX.2-klein-4b-fp8-diffusers"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"

# 생성 파라미터 — 시간/품질 트레이드오프. 페이지 1장 생성 시간이 이 두 상수에 가장 민감.
OUTPUT_RESOLUTION = 768      # 1024 대비 면적 0.56배 → 약 40% 시간 단축
NUM_INFERENCE_STEPS = 4      # Klein 모델 학습 시 4-step 최적화. 3 이하로 내리면 이미지 깨짐 확인됨.


class KleinImageGenerator:
    """FLUX.2-klein-4B 기반 동화 삽화 이미지 생성기."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline = None

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

    # ── 캐릭터 레퍼런스 ─────────────────────────────────────────────────────

    def generate_character_reference(
        self,
        character_prompt: str,
        seed: int = 42,
    ) -> Image.Image:
        """
        캐릭터 레퍼런스 이미지 생성.
        동화 시작 전 역할별 1회씩 호출해서 이후 모든 페이지 생성 시 reference_images 로 재사용.
        """
        if self.pipeline is None:
            raise RuntimeError("먼저 .load()를 호출하세요.")

        style_prefix = (
            "Children's picture book illustration, "
            "semi-painterly digital art with soft cel shading, "
            "gentle storybook character proportions, "
            "warm color palette, soft picture-book aesthetic, NOT photorealistic. "
        )
        full_prompt = (
            style_prefix + character_prompt
            + " Full body portrait, character centered on white background, clear details. "
            + "The face is fully visible with eyes clearly readable. Fur or skin tone is "
            + "even across the entire face — NO dark patches around the eyes, NO mask-like "
            + "markings, NO eye-area shadow that could look like a worn mask. "
            + "No rope wrappings, no leather bands, no warrior costume on limbs."
        )

        print(f"[KleinGenerator] 캐릭터 레퍼런스 생성 중...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipeline(
            prompt=full_prompt,
            height=OUTPUT_RESOLUTION,
            width=OUTPUT_RESOLUTION,
            guidance_scale=1.0,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        ).images[0]

        print(f"  → 캐릭터 레퍼런스 생성 완료")
        return result

    # ── 메인 생성 ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: int = OUTPUT_RESOLUTION,
        height: int = OUTPUT_RESOLUTION,
        num_images: int = 1,
        reference_images: Optional[list[Image.Image]] = None,
    ) -> tuple[list[Image.Image], float]:
        """장면 삽화 생성. reference_images 가 주어지면 multi-image 가이던스로 주입."""
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
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            num_images_per_prompt=num_images,
        )

        if reference_images:
            kwargs["image"] = list(reference_images)

        ref_count = len(kwargs["image"]) if "image" in kwargs else 0
        print(f"[KleinGenerator] 이미지 생성 중...")
        print(f"  프롬프트 (full):")
        print(f"    {prompt}")
        print(f"  크기: {width}x{height} | 레퍼런스: {ref_count}장")

        start_time = time.time()
        output = self.pipeline(**kwargs)
        elapsed = time.time() - start_time

        images = output.images
        print(f"  완료! ({elapsed:.1f}초, {len(images)}장)")
        return images, elapsed

    # ── 라이프사이클 ────────────────────────────────────────────────────────

    def unload(self) -> None:
        """모델 언로드 (메모리 해제)."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[KleinGenerator] 모델 언로드 완료.")
