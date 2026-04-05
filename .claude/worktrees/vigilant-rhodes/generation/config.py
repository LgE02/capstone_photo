"""Generation configuration for the Fairytale LoRA project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODELS = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "type": "sdxl",
        "description": "High-quality fairytale illustrations",
        "default_size": (1024, 1024),
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    },
    "sd15": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "type": "sd15",
        "description": "Faster generation with broad LoRA compatibility",
        "default_size": (512, 512),
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    },
    "sd21": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "type": "sd21",
        "description": "Balanced option between SD 1.5 and SDXL",
        "default_size": (768, 768),
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    },
}

LORA_CONFIGS = {
    "storybook_sdxl": {
        "repo_id": "artificialguybr/StoryBookRedmond-V2",
        "weight_name": "StoryBookRedmondV2-StoryBook-StoryBookAF.safetensors",
        "base_model": "sdxl",
        "lora_scale": 0.9,
        "trigger_word": "StoryBook",
        "description": "Storybook style for SDXL",
    },
    "watercolor_sdxl": {
        "repo_id": "pcuenq/lora-yarn-art-style",
        "weight_name": None,
        "base_model": "sdxl",
        "lora_scale": 0.85,
        "trigger_word": "yarn art style",
        "description": "Soft handcrafted watercolor-like SDXL style",
    },
    "littletinies_sdxl": {
        "repo_id": "alvdansen/littletinies",
        "weight_name": None,
        "base_model": "sdxl",
        "lora_scale": 0.9,
        "trigger_word": "littletinies",
        "description": "Cute miniature fairytale style for SDXL",
    },
    "storybook_sd15": {
        "repo_id": "artificialguybr/StoryBookRedmond-V2",
        "weight_name": "StoryBookRedmondV2-StoryBook-StoryBookAF.safetensors",
        "base_model": "sd15",
        "lora_scale": 0.85,
        "trigger_word": "StoryBook",
        "description": "Storybook style for SD 1.5",
    },
    "anime_illustration_sd15": {
        "repo_id": "Pclanglais/TintinIA",
        "weight_name": None,
        "base_model": "sd15",
        "lora_scale": 0.8,
        "trigger_word": "tintin style",
        "description": "Comic-like illustrated SD 1.5 style",
    },
    "local_lora": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "final_unet_lora"),
        "base_model": "sdxl",
        "lora_scale": 0.87,
        "trigger_word": "ftbookstyle",
        "description": "Current locally trained fairytale LoRA",
    },
}

FAIRYTALE_PROMPTS = {
    "enchanted_forest": {
        "positive": (
            "a magical enchanted forest with glowing fireflies, "
            "a small fairy cottage, watercolor illustration, "
            "children's book art, soft pastel colors, whimsical, "
            "dreamlike atmosphere, detailed foliage"
        ),
        "negative": (
            "dark, scary, horror, realistic photo, 3d render, "
            "ugly, deformed, blurry, text, watermark"
        ),
    },
    "princess_castle": {
        "positive": (
            "a beautiful princess in a magical castle, "
            "fairytale storybook illustration, soft warm lighting, "
            "intricate details, vibrant colors, fantasy art, "
            "children's book style"
        ),
        "negative": (
            "dark, horror, realistic, 3d render, ugly, deformed, "
            "blurry, text, watermark, modern"
        ),
    },
    "dragon_adventure": {
        "positive": (
            "a cute friendly dragon and a brave child adventurer, "
            "colorful storybook illustration, magical landscape, "
            "warm sunlight, watercolor style, charming, whimsical"
        ),
        "negative": (
            "scary, dark, horror, realistic, 3d, ugly, blurry, "
            "text, watermark"
        ),
    },
    "ocean_mermaid": {
        "positive": (
            "a beautiful mermaid underwater kingdom, "
            "glowing coral reef, colorful fish, magical bubbles, "
            "children's book illustration, soft blue tones, "
            "dreamy, whimsical"
        ),
        "negative": (
            "dark, horror, realistic photo, 3d render, ugly, "
            "blurry, text, watermark"
        ),
    },
}

OUTPUT_CONFIG = {
    "output_dir": str(PROJECT_ROOT / "outputs"),
    "save_format": "PNG",
    "save_comparison": True,
    "num_images_per_prompt": 1,
    "output_size": (387, 409),
}

RECOMMENDED_COMBOS = [
    ("sdxl", "storybook_sdxl"),
    ("sdxl", "littletinies_sdxl"),
    ("sd15", "storybook_sd15"),
]
