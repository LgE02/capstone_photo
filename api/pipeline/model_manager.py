"""ModelManager — FLUX 모델을 한 번만 로드하고 재사용한다.

사용 예시:
    from api.pipeline.model_manager import ModelManager

    mgr = ModelManager.get()
    mgr.load()
    generator = mgr.generator
"""

from __future__ import annotations

import threading
from typing import Optional

from api.pipeline.generator import FairytaleImageGenerator


class ModelManager:
    """프로세스 전역 싱글톤 모델 매니저."""

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._generator: Optional[FairytaleImageGenerator] = None

    @classmethod
    def get(cls) -> "ModelManager":
        """전역 ModelManager 인스턴스를 반환한다 (없으면 생성)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """싱글톤을 초기화한다 (테스트 또는 프로세스 종료 시 사용)."""
        with cls._lock:
            if cls._instance is not None and cls._instance._generator is not None:
                cls._instance._generator.unload()
            cls._instance = None

    def load(
        self,
        base_model_key: str = "flux_schnell",
        low_memory_mode: bool = True,
    ) -> "ModelManager":
        """모델을 로드한다. 이미 로드됐으면 재사용."""
        if self._generator is not None:
            print("[ModelManager] 이미 로드됨 — 재사용")
            return self

        print(f"[ModelManager] 모델 로드 시작 (base={base_model_key})")
        self._generator = FairytaleImageGenerator(
            base_model_key=base_model_key,
            low_memory_mode=low_memory_mode,
        )
        self._generator.load()
        print(f"[ModelManager] 로드 완료")
        return self

    def unload(self) -> None:
        """현재 모델을 수동으로 언로드한다."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            print("[ModelManager] 언로드 완료")

    @property
    def generator(self) -> FairytaleImageGenerator:
        if self._generator is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. 먼저 ModelManager.get().load()를 호출하세요."
            )
        return self._generator

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None
