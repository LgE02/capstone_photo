"""원본 이미지만 사용하는 LoRA 재학습 설정.

핵심 변경:
  - train_data_dir → dataset/retrain_raw_only (SD 생성 이미지 제외, 원본 46장만)
  - 캡션: ftbookstyle 트리거 + 이미지별 내용 묘사 (스타일 키워드는 이미지에서 학습)
  - max_train_steps: 550 (46장/batch2 = 23 step/epoch → 약 24에포크)
  - learning_rate: 5e-5 (원본 데이터라 학습 효과가 크므로 적당히)
  - checkpointing_steps: 100 (100~500 비교 가능)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "project_root": str(PROJECT_ROOT),
    "raw_images_dir": str(PROJECT_ROOT / "dataset" / "raw"),
    "train_data_dir": str(PROJECT_ROOT / "dataset" / "retrain_raw_only"),
    "output_dir": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only"),
    "logging_dir": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "logs"),
    "sample_output_dir": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "samples"),
}

MODEL = {
    "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "vae_model_id": "madebyollin/sdxl-vae-fp16-fix",
}

DATASET = {
    "resolution": 768,
    "center_crop": True,
    "random_flip": True,
    "caption_extension": ".txt",
    "instance_prompt": "ftbookstyle",
    "class_prompt": "a fairytale illustration",
    "captions_are_complete": True,
}

STYLE_HINTS = {
    "global": "",
    "storybook_character": "",
    "korean_traditional": "",
    "western_fairytale": "",
    "animal_fairytale": "",
}

PROMPTS = {
    "default_negative": (
        "ugly, blurry, low quality, deformed, distorted face, bad anatomy, "
        "extra fingers, extra limbs, watermark, signature, text, logo, "
        "busy background, detailed background, realistic photo, 3d render, "
        "Chinese clothing, hanfu, qipao, cheongsam, Chinese palace"
    ),
}

TRAINING = {
    "num_train_epochs": 100,
    "max_train_steps": 550,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 55,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "use_8bit_adam": False,
    "seed": 42,
    "checkpointing_steps": 100,
    "validation_epochs": 999,
}

LORA_NETWORK = {
    "rank": 32,
    "alpha": 32,
    "target_modules": [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ],
    "dropout": 0.0,
    "bias": "none",
}

VALIDATION_PROMPTS = [
    # 한국 전래 - 사람
    (
        "ftbookstyle, child picture book illustration, "
        "cute Korean girl in colorful hanbok, round face, big eyes, "
        "large character in foreground, hanok village background"
    ),
    # 동물
    (
        "ftbookstyle, child picture book illustration, "
        "cute tiger cub sitting in a meadow, big expressive eyes, "
        "large character in foreground, soft nature background"
    ),
    # 서양 동화
    (
        "ftbookstyle, child picture book illustration, "
        "cute girl in a red hood walking through a forest, "
        "large character in foreground, magical forest background"
    ),
    # 동물 + 한국
    (
        "ftbookstyle, child picture book illustration, "
        "cute rabbit in traditional Korean hanbok, round face, "
        "large character in foreground, Korean countryside background"
    ),
]

OPTIMIZATION = {
    "enable_xformers": True,
    "allow_tf32": True,
    "dataloader_num_workers": 0,
}
