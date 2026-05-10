"""Klein 파이프라인 공통 설정."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_CONFIG = {
    "output_dir": str(PROJECT_ROOT / "outputs" / "klein_jobs"),
    "save_format": "PNG",
    "save_comparison": False,
    "num_images_per_prompt": 1,
    "output_size": (1024, 1024),
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
        "negative": "modern city, East Asian architecture, hanok, hanbok, Korean clothing",
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
