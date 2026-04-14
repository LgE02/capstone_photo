"""KleinModelManager — FLUX.2-klein-4B 모델 싱글톤 매니저."""

from __future__ import annotations

import threading
from typing import Optional

from api_klein.pipeline.generator_klein import KleinImageGenerator


class KleinModelManager:
    """프로세스 전역 싱글톤 — Klein 모델 전용."""

    _instance: Optional["KleinModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._generator: Optional[KleinImageGenerator] = None

    @classmethod
    def get(cls) -> "KleinModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None and cls._instance._generator is not None:
                cls._instance._generator.unload()
            cls._instance = None

    def load(self) -> "KleinModelManager":
        if self._generator is not None:
            print("[KleinModelManager] 이미 로드됨 — 재사용")
            return self

        print("[KleinModelManager] FLUX.2-klein-4B 로드 시작")
        self._generator = KleinImageGenerator()
        self._generator.load()
        print("[KleinModelManager] 로드 완료")
        return self

    def unload(self) -> None:
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            print("[KleinModelManager] 언로드 완료")

    @property
    def generator(self) -> KleinImageGenerator:
        if self._generator is None:
            raise RuntimeError("먼저 KleinModelManager.get().load()를 호출하세요.")
        return self._generator

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None
