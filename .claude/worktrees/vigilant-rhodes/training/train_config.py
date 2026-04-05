"""Training configuration for the Fairytale LoRA project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "project_root": str(PROJECT_ROOT),
    "raw_images_dir": str(PROJECT_ROOT / "dataset" / "raw"),
    "train_data_dir": str(PROJECT_ROOT / "dataset" / "style_core"),
    "output_dir": str(PROJECT_ROOT / "lora_output"),
    "logging_dir": str(PROJECT_ROOT / "lora_output" / "logs"),
    "sample_output_dir": str(PROJECT_ROOT / "lora_output" / "samples"),
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
}

STYLE_HINTS = {
    "global": (
        "round face, oversized eyes, rosy cheeks, small mouth, soft watercolor shading, "
        "thin clean line art, cute doll-like proportions, readable silhouette, "
        "simple uncluttered background, polished character illustration"
    ),
    "storybook_character": (
        "character-focused composition, large main character, deformed cute style, "
        "soft pastel palette, clean composition"
    ),
    "korean_traditional": (
        "traditional Korean hanbok, Joseon-era Korean clothing, Korean folktale mood, "
        "hanok-inspired details"
    ),
    "western_fairytale": (
        "storybook fantasy costume, charming fairytale clothing, warm and elegant mood"
    ),
    "animal_fairytale": (
        "friendly animal protagonist, simple expressive pose, child-friendly fairytale mood"
    ),
}

PROMPTS = {
    "default_negative": (
        "ugly, blurry, low quality, deformed, distorted face, bad anatomy, "
        "extra fingers, extra limbs, cropped, cut off, watermark, signature, text, logo, "
        "busy background, detailed background, realistic photo, 3d render, "
        "Chinese clothing, hanfu, qipao, cheongsam, Chinese palace"
    ),
    "korean_folktale_positive": (
        "Korean folktale illustration, traditional Korean hanbok, Joseon-era Korean clothing, "
        "jeogori and chima, durumagi, Korean hanok details, Korean folk story mood"
    ),
}

TRAINING = {
    "num_train_epochs": 30,
    "max_train_steps": 140,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 8e-5,
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 40,
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
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    "dropout": 0.0,
    "bias": "none",
}

VALIDATION_PROMPTS = [
    "ftbookstyle, Korean folktale illustration like Kongjwi and Patjwi, very cute chibi Korean girl in hanbok, round face, button nose, soft smile, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, warm storybook mood, simplified background",
    "ftbookstyle, Korean folktale illustration, very cute chibi girl and kind elderly grandmother in hanbok, grandmother with gray hair in a neat bun and warm grandmotherly smile, round faces, button noses, soft smiles, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, warm storybook mood, simplified cottage background",
    "ftbookstyle, Korean folktale illustration, very cute chibi child near a traditional Korean house and pumpkin carriage, round face, button nose, soft smile, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, gentle storybook mood",
    "ftbookstyle, fantasy fairytale illustration like Little Red Riding Hood, very cute chibi girl in a red hood, round face, button nose, soft smile, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, warm forest storybook mood",
    "ftbookstyle, fantasy fairytale illustration, very cute chibi child in a cozy magical winter alley, soft lantern light, round face, button nose, soft smile, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, dreamy storybook mood",
    "ftbookstyle, fantasy fairytale illustration, very cute chibi child with kind grandfather storyteller, grandfather drawn in the same soft storybook style, round faces, button noses, soft smiles, big gentle eyes, rosy cheeks, simple cute proportions, thin clean line art, soft pastel colors, magical storybook mood, simplified background, not realistic",
    "ftbookstyle, animal protagonist fairytale illustration like The Tortoise and the Hare, very cute rabbit and very cute tortoise standing side by side as separate characters, rabbit with long ears and no shell, tortoise with shell and short legs, round simple shapes, soft smile, gentle expression, thin clean line art, soft pastel colors, friendly storybook mood, simplified background",
    "ftbookstyle, animal protagonist fairytale illustration, very cute bear cub with kind grandmother bear, round simple shapes, soft smiles, gentle expression, thin clean line art, soft pastel colors, warm storybook mood, simplified forest background",
    "ftbookstyle, animal protagonist fairytale illustration like a rabbit folk tale, very cute little rabbit main character, round simple shapes, soft smile, gentle expression, thin clean line art, soft pastel colors, friendly storybook mood, plain warm background",
]

OPTIMIZATION = {
    "enable_xformers": True,
    "allow_tf32": True,
    "dataloader_num_workers": 0,
}
