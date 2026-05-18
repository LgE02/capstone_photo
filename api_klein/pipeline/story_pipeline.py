"""Klein 스토리 파이프라인 — setting → 배경 힌트 변환."""

from __future__ import annotations

from dataclasses import dataclass

from api_klein.pipeline.config import THEME_EXPANSIONS


@dataclass
class WorldProfile:
    theme: str
    expansion_key: str
    positive_hint: str
    negative_hint: str


def build_world_profile(theme: str) -> WorldProfile:
    """테마 키를 실제 프롬프트 힌트로 변환."""
    normalized = theme.strip().lower()
    matched_key = "FOREST_NATURE"
    for key, value in THEME_EXPANSIONS.items():
        aliases = [key.lower()] + [alias.lower() for alias in value.get("aliases", [])]
        if normalized in aliases:
            matched_key = key
            break

    config = THEME_EXPANSIONS[matched_key]
    return WorldProfile(
        theme=theme,
        expansion_key=matched_key,
        positive_hint=config["positive"],
        negative_hint=config["negative"],
    )
