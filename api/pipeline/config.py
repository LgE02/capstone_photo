"""동화 삽화 생성 파이프라인 공통 설정."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODELS = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "type": "sdxl",
        "description": "Primary base model for fairytale illustration generation",
        "default_size": (1024, 1024),
        "num_inference_steps": 30,
        "guidance_scale": 8.0,
    },
}

LORA_CONFIGS = {
    "raw_100": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "checkpoint-100"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only checkpoint-100",
    },
    "raw_200": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "checkpoint-200"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only checkpoint-200",
    },
    "raw_300": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "checkpoint-300"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only checkpoint-300",
    },
    "raw_400": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "checkpoint-400"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only checkpoint-400",
    },
    "raw_500": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "checkpoint-500"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only checkpoint-500",
    },
    "raw_final": {
        "repo_id": None,
        "weight_name": str(PROJECT_ROOT / "lora_output" / "retrain_raw_only" / "final_unet_lora"),
        "base_model": "sdxl",
        "lora_scale": 0.80,
        "trigger_word": "ftbookstyle",
        "description": "raw_only 최종 550-step",
    },
}

OUTPUT_CONFIG = {
    "output_dir": str(PROJECT_ROOT / "outputs"),
    "save_format": "PNG",
    "save_comparison": True,
    "num_images_per_prompt": 1,
    "output_size": (387, 409),
}

STYLE_PRESETS = {
    "fairytale_pastel": {
        "positive": (
            "ftbookstyle, flat cartoon illustration, bold line art, pastel colors"
        ),
        "negative": (
            "multiple characters, crowd scene, many figures, repeating pattern, "
            "wallpaper pattern, group of characters, "
            "plain white background, empty background, "
            "photorealistic background, dark gloomy background, "
            "realistic photo, 3d render, dark mood, harsh shadows, "
            "tiny character, character too small, "
            "overly detailed background, cluttered background, "
            "text, watermark, signature, "
            "thin line art, sketch, rough sketch, watercolor, oil painting"
        ),
    }
}

THEME_EXPANSIONS = {
    "KOREAN_TRADITIONAL": {
        "aliases": ["korean_traditional", "korean traditional", "joseon", "korean"],
        "positive": "Korean hanok house, curved tiled roof, dancheong painted eaves",
        "negative": (
            "Chinese architecture, Chinese pagoda, Chinese lantern, red Chinese temple, "
            "Chinese hanfu, Chinese dragon robe, Chinese palace, "
            "Japanese kimono, Japanese torii, Japanese castle, "
            "modern city, Western medieval castle, European building"
        ),
    },
    "FOREST_NATURE": {
        "aliases": ["forest_nature", "forest", "nature", "woods", "woodland"],
        "positive": "lush green forest, tall trees, dappled sunlight, woodland path",
        "negative": "modern buildings, city street, industrial background, East Asian architecture",
    },
    "MIXED": {
        "aliases": ["mixed", "hybrid", "varied"],
        "positive": "colorful storybook background, warm lighting",
        "negative": "photorealistic collage, chaotic clutter, dark mood",
    },
    "FANTASY_WORLD": {
        "aliases": ["fantasy_world", "fantasy", "magic world"],
        "positive": "magical fantasy landscape, enchanted forest, glowing particles",
        "negative": "modern office, realistic suburb, military setting, East Asian architecture",
    },
    "EUROPEAN_MEDIEVAL": {
        "aliases": ["european_medieval", "western medieval", "medieval", "european"],
        "positive": "European medieval village, cobblestone street, half-timbered houses, stone castle",
        "negative": "modern city, East Asian architecture, hanok, hanbok, Korean clothing, Chinese clothing, Japanese clothing",
    },
    "MODERN_FANTASY": {
        "aliases": ["modern_fantasy", "modern fantasy"],
        "positive": "modern city with magical elements, glowing lights, enchanted garden",
        "negative": "strict historical costume, military gear, East Asian architecture",
    },
    "UNDERWATER": {
        "aliases": ["underwater", "ocean", "sea", "sea kingdom"],
        "positive": "underwater ocean scene, colorful coral reef, bubbles, light rays through water",
        "negative": "dry desert, city street, land vegetation, sky, clouds",
    },
    "SKY_HEAVEN": {
        "aliases": ["sky_heaven", "sky", "heaven", "cloud kingdom"],
        "positive": "celestial sky kingdom, fluffy clouds, golden sunlight, starry atmosphere",
        "negative": "underground cave, dark industrial background, ocean, water",
    },
}

CHARACTER_TYPE_HINTS = {
    "human": "cute human character, round face, big expressive eyes, child-friendly proportions",
    "animal": "cute anthropomorphic animal character, round face, big eyes, soft rounded body",
    "other": "storybook character, simple rounded silhouette, cute details",
}

JOB_HINTS = {
    "king": "wearing red gonryongpo dragon robe, royal golden crown, magnificent Joseon king",
    "queen": "wearing elaborate royal hanbok, ornate golden hair ornament, Korean queen",
    "prince": "wearing royal blue hanbok, ornate headpiece, Joseon prince",
    "princess": "wearing vibrant royal hanbok, elegant hair ornament, Korean princess",
    "magistrate": "wearing dark navy official hanbok, tall black gat horsehair hat, Joseon official",
    "scholar": "wearing white jeogori and black baji, black gat horsehair hat, Joseon scholar",
    "farmer": "wearing plain beige jeogori and baji, straw hat, Korean farmer",
    "monk": "wearing grey monk robe, simple prayer beads, Korean Buddhist monk",
    "soldier": "wearing Korean traditional armor, red tassel, Joseon soldier",
    "merchant": "wearing travel hanbok, carrying large cloth bundle, Korean merchant",
    "villager": "wearing simple plain hanbok, everyday Korean village clothing",
    "witch": "wearing dark magical outfit, pointed hat, storybook witch",
    "wizard": "wearing long magical robe, holding glowing staff, storybook wizard",
}

SPECIES_HINTS = {
    "frog":     "small cute frog, chubby face, big bright eyes, expressive",
    "rabbit":   "small fluffy rabbit, long ears, big bright eyes, expressive",
    "tiger":    "friendly chubby tiger, round face, soft stripes, expressive",
    "fox":      "small cute fox, fluffy tail, pointy ears, expressive",
    "bear":     "round chubby bear, gentle eyes, small snout, expressive",
    "cat":      "small cute cat, pointy ears, big round eyes, expressive",
    "dog":      "friendly small dog, floppy ears, bright eyes, expressive",
    "deer":     "gentle small deer, tiny antlers, big soft eyes, expressive",
    "bird":     "small round bird, big bright eyes, fluffy feathers",
    "turtle":   "small round turtle, friendly smile, smooth shell",
    "dragon":   "small friendly dragon, chubby body, big bright eyes",
    "lion":     "friendly chubby lion, round face, soft fluffy mane",
    "monkey":   "small playful monkey, round face, big expressive eyes",
    "pig":      "small round pig, chubby cheeks, bright eyes, expressive",
    "snake":    "small cute snake, friendly face, shiny scales, expressive",
    "wolf":     "friendly fluffy wolf, round face, bright eyes, expressive",
    "mouse":    "tiny cute mouse, round ears, big bright eyes",
    "squirrel": "small fluffy squirrel, bushy tail, big bright eyes",
    "horse":    "cute small pony, flowing mane, big bright eyes",
    "elephant": "small friendly elephant, big soft ears, bright eyes",
}

SPECIES_FALLBACK_TEMPLATE = (
    "cute {species} character, round chubby face, big bright eyes, soft rounded body, picture book"
)

THEME_CHARACTER_COSTUME_HINTS = {
    "KOREAN_TRADITIONAL": {
        "human": "vibrant colorful hanbok, jeogori ribbon bow",
        "animal": "small colorful hanbok, ribbon bow tie",
        "other": "Korean traditional ornamental details",
    },
    "EUROPEAN_MEDIEVAL": {
        "human": "medieval tunic and cloak, leather belt",
        "animal": "small medieval cape, storybook costume",
        "other": "European medieval decorative details",
    },
    "FANTASY_WORLD": {
        "human": "colorful fantasy adventurer outfit",
        "animal": "small fantasy outfit, magical accessory",
        "other": "magical glowing details",
    },
    "FOREST_NATURE": {
        "human": "simple folk costume, earth tones",
        "animal": "",
        "other": "natural leaf and flower decorations",
    },
    "UNDERWATER": {
        "human": "ocean-colored garment, seashell accessories",
        "animal": "",
        "other": "coral and seashell decorations",
    },
    "SKY_HEAVEN": {
        "human": "flowing white celestial robe",
        "animal": "small cloud-white outfit",
        "other": "cloud and star decorations",
    },
}

RECOMMENDED_COMBOS = [
    ("sdxl", "raw_200"),
    ("sdxl", "raw_300"),
    ("sdxl", "raw_400"),
]

DEFAULT_LORA = "raw_300"
