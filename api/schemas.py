"""Pydantic 요청/응답 모델."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ProtagonistType(str, Enum):
    human = "human"
    animal = "animal"
    other = "other"


class ThemeType(str, Enum):
    KOREAN_TRADITIONAL = "KOREAN_TRADITIONAL"
    FOREST_NATURE = "FOREST_NATURE"
    MIXED = "MIXED"
    FANTASY_WORLD = "FANTASY_WORLD"
    EUROPEAN_MEDIEVAL = "EUROPEAN_MEDIEVAL"
    MODERN_FANTASY = "MODERN_FANTASY"
    UNDERWATER = "UNDERWATER"
    SKY_HEAVEN = "SKY_HEAVEN"


class GenerateRequest(BaseModel):
    story_text: str = Field(..., description="\\n으로 구분된 동화 전문")
    protagonist_type: ProtagonistType = ProtagonistType.other
    theme: ThemeType = ThemeType.FOREST_NATURE
    seed: int | None = 42
    lora_key: str = "raw_200"
