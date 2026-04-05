"""ModelManager — SDXL + LoRA 모델을 한 번만 로드하고 재사용한다.

동일 모델/LoRA 조합이면 이미 로드된 인스턴스를 그대로 반환한다.
다른 조합으로 전환할 때만 언로드 후 재로드한다.

사용 예시:
    from runtime_pipeline.model_manager import ModelManager

    mgr = ModelManager.get()
    mgr.load(lora_key="raw_200")

    # 이후 어디서든 동일 인스턴스 재사용
    mgr = ModelManager.get()
    generator = mgr.generator   # 이미 로드된 FairytaleImageGenerator
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    from fairytale_lora.runtime_pipeline.generator import FairytaleImageGenerator
except ModuleNotFoundError:
    from runtime_pipeline.generator import FairytaleImageGenerator


class ModelManager:
    """프로세스 전역 싱글톤 모델 매니저.

    멀티스레드 환경에서도 안전하게 단 하나의 인스턴스만 유지한다.
    """

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._generator: Optional[FairytaleImageGenerator] = None
        self._loaded_lora_key: Optional[str] = None
        self._loaded_base_key: Optional[str] = None

    # ── 싱글톤 접근 ─────────────────────────────────────────────────────────

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

    # ── 모델 로드 / 언로드 ──────────────────────────────────────────────────

    def load(
        self,
        base_model_key: str = "sdxl",
        lora_key: Optional[str] = "raw_200",
        low_memory_mode: bool = False,
        lora_scale_override: Optional[float] = None,
    ) -> "ModelManager":
        """모델을 로드한다.

        이미 동일한 조합이 로드되어 있으면 즉시 반환 (재로드 없음).
        다른 조합이면 기존 모델을 언로드한 후 새로 로드한다.

        Args:
            base_model_key: BASE_MODELS 키 (기본 "sdxl")
            lora_key: LORA_CONFIGS 키 (None이면 LoRA 미사용)
            low_memory_mode: VRAM 부족 시 CPU 오프로딩 활성화
            lora_scale_override: LoRA 스케일 강제 지정 (None이면 config 기본값 사용)

        Returns:
            self (메서드 체이닝 지원)
        """
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

        # 기존 모델 언로드
        if self._generator is not None:
            print(
                f"[ModelManager] 기존 모델 언로드 "
                f"(base={self._loaded_base_key}, lora={self._loaded_lora_key})"
            )
            self._generator.unload()
            self._generator = None

        # 새 모델 로드
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
        """현재 모델을 수동으로 언로드한다 (VRAM 확보 필요 시)."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            self._loaded_base_key = None
            self._loaded_lora_key = None
            print("[ModelManager] 언로드 완료")

    # ── 속성 접근 ───────────────────────────────────────────────────────────

    @property
    def generator(self) -> FairytaleImageGenerator:
        """로드된 FairytaleImageGenerator를 반환한다.

        Raises:
            RuntimeError: 모델이 로드되지 않은 상태에서 접근 시
        """
        if self._generator is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. 먼저 ModelManager.get().load()를 호출하세요."
            )
        return self._generator

    @property
    def is_loaded(self) -> bool:
        """모델이 현재 로드된 상태인지 반환한다."""
        return self._generator is not None

    @property
    def loaded_combo(self) -> tuple[Optional[str], Optional[str]]:
        """현재 로드된 (base_model_key, lora_key) 조합을 반환한다."""
        return self._loaded_base_key, self._loaded_lora_key
