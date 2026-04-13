"""ModelManager — SDXL + LoRA 모델을 한 번만 로드하고 재사용한다.

사용 예시:
    from api.pipeline.model_manager import ModelManager

    mgr = ModelManager.get()
    mgr.load(lora_key="raw_200")
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
        self._loaded_lora_key: Optional[str] = None
        self._loaded_base_key: Optional[str] = None

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
        base_model_key: str = "sdxl",
        lora_key: Optional[str] = "raw_200",
        low_memory_mode: bool = False,
        lora_scale_override: Optional[float] = None,
    ) -> "ModelManager":
        """모델을 로드한다. 동일 조합이면 재사용."""
        same_combo = (
            self._generator is not None
            and self._loaded_base_key == base_model_key
            and self._loaded_lora_key == lora_key
        )

        if same_combo:
            print(
                f"[ModelManager] 이미 로드됨 — 재사용 "
                f"(base={base_model_key}, lora={lora_key})"
            )
            return self

        if self._generator is not None:
            print(
                f"[ModelManager] 기존 모델 언로드 "
                f"(base={self._loaded_base_key}, lora={self._loaded_lora_key})"
            )
            self._generator.unload()
            self._generator = None

        print(f"[ModelManager] 모델 로드 시작 (base={base_model_key}, lora={lora_key})")
        self._generator = FairytaleImageGenerator(
            base_model_key=base_model_key,
            lora_key=lora_key,
            low_memory_mode=low_memory_mode,
            lora_scale_override=lora_scale_override,
        )
        self._generator.load()
        self._loaded_base_key = base_model_key
        self._loaded_lora_key = lora_key
        print(f"[ModelManager] 로드 완료")
        return self

    def unload(self) -> None:
        """현재 모델을 수동으로 언로드한다."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            self._loaded_base_key = None
            self._loaded_lora_key = None
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

    @property
    def loaded_combo(self) -> tuple[Optional[str], Optional[str]]:
        return self._loaded_base_key, self._loaded_lora_key
