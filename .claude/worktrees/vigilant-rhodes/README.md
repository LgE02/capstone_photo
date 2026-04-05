# 동화 삽화 이미지 생성기 (LoRA 기반)

LoRA 가중치를 적용한 Stable Diffusion으로 동화 스타일 삽화를 생성합니다.

---

## 📁 파일 구조

```
fairytale_lora/
├── config.py               # 모델/LoRA/프롬프트 설정
├── generator.py            # 핵심 이미지 생성 클래스
├── fairytale_generator.py  # 단일 이미지 생성 실행 스크립트
├── compare_models.py       # 여러 모델 비교 실험 스크립트
├── requirements.txt        # 패키지 의존성
└── outputs/                # 생성된 이미지 저장 폴더
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

GPU 환경(CUDA)에서 VRAM이 부족한 경우:
```bash
pip install xformers  # 메모리 최적화
```

---

## 🚀 빠른 시작

### 기본 실행 (추천 설정 자동 적용)
```bash
python fairytale_generator.py
```

### 프롬프트 직접 지정
```bash
python fairytale_generator.py --prompt "a little mermaid exploring underwater palace"
```

### 프리셋 사용
```bash
python fairytale_generator.py --preset enchanted_forest
python fairytale_generator.py --preset princess_castle
python fairytale_generator.py --preset dragon_adventure
python fairytale_generator.py --preset ocean_mermaid
```

### 모델 및 LoRA 지정
```bash
# 고품질 (느림)
python fairytale_generator.py --model sdxl --lora storybook_sdxl

# 빠른 생성 (가벼운 환경)
python fairytale_generator.py --model sd15 --lora storybook_sd15
```

---

## 🔍 모델/LoRA 비교 실험

동일한 프롬프트로 여러 조합을 테스트하고 비교합니다:

```bash
# 추천 조합 3개 자동 비교
python compare_models.py

# 특정 프리셋으로 비교
python compare_models.py --preset dragon_adventure

# 조합 직접 지정
python compare_models.py --combos sdxl:storybook_sdxl sd15:storybook_sd15
```

결과는 `comparison_outputs/` 폴더에 저장되며, 격자 비교 이미지와 JSON 리포트가 생성됩니다.

---

## 🎨 추천 모델/LoRA 조합

| 우선순위 | 베이스 모델 | LoRA | 특징 |
|---------|-----------|------|------|
| ⭐⭐⭐ | SDXL | StoryBook Redmond V2 | 고품질 동화책 스타일 |
| ⭐⭐⭐ | SDXL | littletinies | 귀여운 미니어처 스타일 |
| ⭐⭐ | SD 1.5 | StoryBook Redmond V2 | 빠른 속도 |

---

## 📌 LoRA 추가하기

### HuggingFace Hub에서
`config.py`의 `LORA_CONFIGS`에 추가:
```python
"my_lora": {
    "repo_id": "username/my-fairytale-lora",
    "weight_name": "my_lora.safetensors",  # None이면 자동 감지
    "base_model": "sdxl",
    "lora_scale": 0.9,
    "trigger_word": "my style",
    "description": "나만의 동화 스타일",
},
```

### 로컬 파일에서 (.safetensors)
```python
"local_custom": {
    "repo_id": None,
    "weight_name": "./lora_weights/custom.safetensors",
    "base_model": "sdxl",
    "lora_scale": 0.85,
    "trigger_word": "",
    "description": "로컬 커스텀 LoRA",
},
```

---

## 💡 팁

- **seed 고정**: `--seed 42` 로 재현 가능한 결과를 얻을 수 있습니다.
- **VRAM 부족**: `--low-memory` 플래그를 사용하면 CPU 오프로딩이 활성화됩니다.
- **품질 vs 속도**: `--steps 50`으로 높이면 품질 향상, `--steps 15`로 낮추면 속도 향상.
- **프롬프트**: 영문으로 작성하면 더 좋은 결과가 나옵니다.
