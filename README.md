# 동화 삽화 생성 프로젝트

4~6세 아동용 동화 텍스트를 입력하면, 장면별 삽화를 자동 생성하는 파이프라인입니다.

- **모델**: SDXL + LoRA (ftbookstyle) + IP-Adapter
- **LLM**: GPT-4o (장면 분석 및 프롬프트 추출)
- **출력**: 387×409px PNG (생성 해상도 832×896)
- **지원 범위**: 주인공 3종(동물/사람/기타) × 배경 8종

---

## 프로젝트 구조

```text
fairytale_lora/
├─ runtime_pipeline/          # 삽화 생성 파이프라인 (핵심)
│  ├─ config.py               # 모델/스타일/테마/캐릭터 설정
│  ├─ llm_prompt_extractor.py # GPT-4o 기반 스토리 분석
│  ├─ story_pipeline.py       # 스토리 → 장면 프롬프트 변환
│  ├─ story_analyzer.py       # 키워드 매핑 (LLM fallback)
│  ├─ generator.py            # SDXL 이미지 생성기
│  ├─ async_generator.py      # 비동기/동기 생성 + IP-Adapter
│  └─ model_manager.py        # 모델 로드/언로드 싱글턴
├─ model_training/            # LoRA 학습
│  ├─ train_lora.py
│  ├─ train_config_raw_only.py
│  └─ verify_lora.py
├─ dataset/                   # 학습 데이터
├─ lora_output/               # 학습된 LoRA 체크포인트
├─ outputs/                   # 생성 결과 저장
├─ test_pipeline.py           # 파이프라인 테스트 스크립트
├─ compare_checkpoints.py     # LoRA 체크포인트 비교
└─ prepare_retrain_raw.py     # 재학습 데이터 준비
```

---

## 파이프라인 흐름

```
동화 텍스트 (한국어)
    │
    ▼
┌─────────────────────────────┐
│  1. LLM 분석 (GPT-4o)       │  llm_prompt_extractor.py
│  - 캐릭터 추출 (종/직업/외형) │
│  - 테마 분류                 │
│  - 장면별 영어 프롬프트 생성  │
│  - 등장 캐릭터 판별           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. 스토리 플랜 생성          │  story_pipeline.py
│  - 캐릭터 바이블 (시각 묘사)  │
│  - 세계관 프로필 (테마별 힌트) │
│  - 장면별 듀얼 프롬프트 조립  │
│  - 네거티브 프롬프트 구성     │
│  - 장면별 LoRA scale 결정    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. 이미지 생성 (SDXL)       │  generator.py + async_generator.py
│  - LoRA 적용 (동적 scale)    │
│  - IP-Adapter (단일 캐릭터)  │
│  - 듀얼 프롬프트 (CLIP-L/G)  │
│  - 387×409 리사이즈 후 저장   │
└─────────────────────────────┘
```

---

## SDXL 듀얼 프롬프트 전략

SDXL은 텍스트 인코더가 2개이므로, 각각에 다른 역할을 부여합니다.

| 인코더 | 파라미터 | 역할 | 내용 예시 |
|--------|----------|------|-----------|
| CLIP-L (77토큰) | `prompt` | 캐릭터 외형 상세 | `ftbookstyle, flat cartoon illustration, small white rabbit with big floppy ears wearing red vest` |
| CLIP-G (77토큰) | `prompt_2` | 스타일 + 장면 + 구도 | `child picture book illustration, a white rabbit animal, walking through forest path, solo, large character` |

다중 캐릭터 시 CLIP-L에서 `BREAK` 토큰으로 캐릭터 개념을 분리합니다:
```
small white rabbit, red vest BREAK young woodcutter, brown tunic, human NOT rabbit
```

---

## 캐릭터 일관성 전략

### 캐릭터 바이블
LLM(GPT-4o)이 각 캐릭터의 고정 시각 묘사를 생성합니다:
- 체색/피부색
- 고유 액세서리 (빨간 조끼, 금 왕관 등)
- 고유 신체 특징 (큰 귀, 긴 수염 등)

이 묘사가 모든 장면의 프롬프트에 일관되게 삽입됩니다.

### IP-Adapter (캐릭터 초상화 참조)
- **단일 캐릭터 장면**: IP-Adapter ON (scale 0.2) — 해당 캐릭터 초상화를 참조
- **다중 캐릭터 장면**: IP-Adapter OFF — 한 캐릭터 참조 시 다른 캐릭터도 그 모습으로 변하는 문제 방지

### LoRA Scale 동적 조절
LoRA가 단일 캐릭터 이미지로 학습되어, 다중 캐릭터 시 하나로 합치려는 경향이 있습니다.
- **단일 캐릭터**: LoRA scale 0.8 (스타일 강하게)
- **다중 캐릭터 (2~3명)**: LoRA scale 0.55 (SDXL 기본 모델의 구성 능력 활용)

---

## 캐릭터 혼합 방지 전략

다중 캐릭터 장면에서 캐릭터가 섞이는 문제(나무꾼에 토끼귀 등)를 방지하기 위한 다층 접근:

| 계층 | 방법 | 적용 조건 |
|------|------|-----------|
| 프롬프트 | BREAK 토큰으로 캐릭터 분리 | 2명 이상 |
| 프롬프트 | 동물에 "animal NOT human" 접미 | 동물 캐릭터 |
| 프롬프트 | 캐릭터 좌/우 위치 명시 (LLM) | 2명 이상 |
| 네거티브 | "hybrid creature, merged characters, fused body" | 2명 이상 |
| 네거티브 | "human, person, human face" | 전원 동물 장면 |
| 네거티브 | "anthropomorphic, kemonomimi" | 동물+사람 혼재 |
| 네거티브 | "picture frame, border, page layout" | 모든 장면 |
| 모델 | LoRA scale 0.55로 낮춤 | 2~3명 |
| 모델 | IP-Adapter OFF | 2~3명 |

---

## 지원 테마 (8종)

| 테마 키 | 설명 | 배경 힌트 |
|---------|------|-----------|
| `KOREAN_TRADITIONAL` | 한국 전통 | 한옥, 단청, 기와 |
| `FOREST_NATURE` | 숲/자연 | 자연 배경 |
| `FANTASY_WORLD` | 판타지 | 마법 배경 |
| `EUROPEAN_MEDIEVAL` | 유럽 중세 | 중세 마을 |
| `MODERN_FANTASY` | 현대 판타지 | 마법 배경 |
| `UNDERWATER` | 바닷속 | 산호 배경 |
| `SKY_HEAVEN` | 하늘/천상 | 파스텔 하늘 |
| `MIXED` | 혼합 | 컬러풀 배경 |

한국 전통 테마에서는 SDXL이 한국/중국을 혼동하는 문제를 방지하기 위해
LLM에게 한옥→"Korean hanok", 한복→"hanbok" 등 명시적 번역 규칙을 적용합니다.
이 규칙은 비한국 테마에서는 적용하지 않습니다.

---

## 주요 설정값

| 설정 | 값 | 파일 |
|------|-----|------|
| CFG Scale | 9.0 | `config.py` |
| Inference Steps | 30 | `config.py` |
| 스케줄러 | DPM++ 2M Karras | `generator.py` |
| LoRA Scale (solo) | 0.8 | `story_pipeline.py` |
| LoRA Scale (multi) | 0.55 | `story_pipeline.py` |
| IP-Adapter Scale | 0.2 | `generator.py` |
| 생성 해상도 | 832×896 | `test_pipeline.py` |
| 출력 해상도 | 387×409 | `config.py` |
| 트리거 단어 | `ftbookstyle` | `config.py` |
| LLM 모델 | GPT-4o | `llm_prompt_extractor.py` |
| 최대 장면 수 | 10 (기본, 조정 가능) | `llm_prompt_extractor.py` |

---

## 실행 방법

### 환경 설정
```bash
cd fairytale_lora
.venv/Scripts/activate   # Windows
# 또는
source .venv/bin/activate  # Linux/Mac
```

### 기본 테스트
```bash
python test_pipeline.py
```

### 옵션
```bash
# 커스텀 동화 텍스트
python test_pipeline.py --story "옛날에 토끼가 숲에서 살았습니다..."

# 시드 고정 (재현성)
python test_pipeline.py --seed 42

# LoRA 체크포인트 선택
python test_pipeline.py --lora raw_300

# 저메모리 모드 (VRAM 부족 시)
python test_pipeline.py --low-memory

# 모델 유지 (연속 테스트)
python test_pipeline.py --keep-loaded
```

### LoRA 체크포인트 비교
```bash
python compare_checkpoints.py
```

---

## LLM 프롬프트 추출 (GPT-4o)

`llm_prompt_extractor.py`가 동화 전체를 1회 API 호출로 분석합니다.

**입력**: 한국어 동화 텍스트
**출력**:
```json
{
  "theme": "KOREAN_TRADITIONAL",
  "characters": [
    {
      "id": "rabbit",
      "type": "animal",
      "species": "rabbit",
      "visual_description": "small white rabbit with big floppy ears, wearing a red vest with gold buttons"
    }
  ],
  "scenes": [
    {
      "line": "토끼가 숲길을 걸어갔습니다.",
      "scene_prompt": "white rabbit with animal body walking on forest path, tall trees around",
      "focus_characters": ["rabbit"]
    }
  ]
}
```

**핵심 규칙**:
- `focus_characters`: 물리적으로 **보이는** 캐릭터만 (언급만 되는 캐릭터 제외)
- 동물 캐릭터: `"animal body, NOT human"` 필수 포함
- 다중 캐릭터: `"character A on left, character B on right"` 위치 명시
- 한국 전통 테마: 한옥/한복 등 명시적 번역 규칙 적용
- 비용: 동화 1편당 약 $0.03

---

## LoRA 학습 정보

- **학습 데이터**: 원본 동화 삽화 46장 (raw only)
- **체크포인트**: 100 / 200 / 300 / 400 / 500 step + final (550 step)
- **현재 사용**: `raw_300` (기본)
- **트리거 단어**: `ftbookstyle`
- **PEFT→diffusers 키 변환**: `base_model.model.*` → `unet.*` (자동 처리)

---

## 알려진 제한사항

- **다중 캐릭터 혼합**: BREAK 토큰 + 네거티브로 완화하지만, SDXL 한계로 완전 방지는 어려움
- **캐릭터 일관성**: IP-Adapter 0.2 + 캐릭터 바이블로 개선했으나, 페이지 간 완벽한 일관성은 미달
- **동물 의인화**: LoRA가 인간 형태로 끌려가는 경향이 있어 네거티브로 억제
- **CLIP 토큰 한계**: CLIP-L/G 각 77토큰이므로 캐릭터 묘사가 길면 잘림
- **3명 이상**: 최대 3캐릭터까지 지원, 그 이상은 품질 보장 어려움
