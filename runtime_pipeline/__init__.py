"""Runtime-only pipeline package for fairytale illustration generation."""

from __future__ import annotations

from importlib import import_module

from .story_pipeline import build_story_plan

__all__ = ["FairytaleImageGenerator", "build_story_plan"]


def __getattr__(name: str):
    if name == "FairytaleImageGenerator":
        return import_module(".generator", __name__).FairytaleImageGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
