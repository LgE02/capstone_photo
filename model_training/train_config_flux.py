"""FLUX.1 Schnell용 LoRA 학습 설정.

GPU 12GB (NF4 양자화) 환경에 최적화.
- 42장 동화 삽화 데이터셋
- NF4 양자화 트랜스포머 + 8bit Adam
- 해상도 512x512 (VRAM 절약)
- rank=4, lr=1e-4, ~700 steps
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "project_root": str(PROJECT_ROOT),
    "train_data_dir": str(PROJECT_ROOT / "dataset" / "flux_train"),
    "output_dir": str(PROJECT_ROOT / "lora_output" / "flux_lora"),
    "logging_dir": str(PROJECT_ROOT / "lora_output" / "flux_lora" / "logs"),
}

MODEL = {
    "base_model_id": "black-forest-labs/FLUX.1-schnell",
}

DATASET = {
    "resolution": 512,
    "caption_extension": ".txt",
    "center_crop": True,
    "random_flip": True,
}

TRAINING = {
    "max_train_steps": 700,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 50,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "use_8bit_adam": True,
    "seed": 42,
    "checkpointing_steps": 100,
}

LORA = {
    "rank": 4,
    "alpha": 4,
    "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
}

QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
}
