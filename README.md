# 동화 삽화 생성 프로젝트

4~6세 아동용 동화 텍스트를 입력하면, 장면별 삽화를 자동 생성하는 파이프라인입니다.

- **모델**: FLUX.1 Schnell (NF4 양자화)
- **LLM**: GPT-4o (장면 분석 및 프롬프트 추출)
- **프롬프트**: T5 인코더용 자연어 문장
- **출력**: 387×409px PNG (생성 해상도 1024×1024)
- **지원 범위**: 주인공 3종(동물/사람/기타) × 배경 8종

---

## 프로젝트 구조

```text
fairytale_lora/
├─ api/                       # FastAPI 서버 (진입점)
│  ├─ app.py                  # 앱 + lifespan (모델 로드/언로드)
│  ├─ schemas.py              # Pydantic 요청/응답 모델
│  ├─ routes.py               # POST /generate (SSE), GET /jobs/{id}
│  ├─ tasks.py                # Job, JobStore, JobExecutor, ImageStorage
│  └─ pipeline/               # 이미지 생성 파이프라인
│     ├─ config.py            # 모델/테마 설정
│     ├─ generator.py         # FLUX Schnell NF4 이미지 생성기
│     ├─ model_manager.py     # 싱글톤 모델 매니저
│     ├─ story_pipeline.py    # 스토리 → T5 자연어 프롬프트 변환
│     ├─ llm_prompt_extractor.py  # GPT-4o 스토리 분석
│     └─ story_analyzer.py    # 키워드 매핑 (LLM fallback)
├─ model_training/            # LoRA 학습
│  ├─ compute_embeddings.py   # 텍스트 임베딩 사전 계산
│  ├─ train_dreambooth_lora_flux_miniature.py  # FLUX LoRA 학습
│  ├─ train_config_flux.py    # 학습 설정
│  └─ run_flux_train.bat      # 학습 실행 스크립트
├─ dataset/
│  └─ flux_train/             # 학습 데이터 (42장 + 자연어 캡션)
├─ lora_output/               # 학습된 LoRA 체크포인트
├─ outputs/                   # 생성 결과 저장
└─ test_pipeline.py           # CLI 테스트 스크립트
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
│  - 장면별 자연어 프롬프트 생성 │
│  - 감정/표정/행동 묘사 포함   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. 스토리 플랜 생성          │  story_pipeline.py
│  - 캐릭터 바이블 (시각 묘사)  │
│  - 세계관 프로필 (테마별 힌트) │
│  - T5 자연어 프롬프트 조립    │
│  - LLM visual_hint 직접 사용 │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. 이미지 생성 (FLUX)       │  generator.py
│  - NF4 양자화 (GPU 12GB)    │
│  - 4 steps, CFG 0.0         │
│  - 387×409 리사이즈 후 저장   │
└─────────────────────────────┘
```

---

## FLUX T5 프롬프트 전략

FLUX는 T5 인코더(512토큰)를 사용하며, 자연어 문장으로 프롬프트를 구성합니다.

```
A children's picture book illustration in flat cartoon style with bold outlines
and soft pastel colors. There are exactly two separate characters.
First character: small white rabbit with big floppy ears, wearing a red vest.
This character is an animal with an animal body, positioned on the left side.
Second character: a sturdy woodcutter in rough brown work clothes, carrying an axe.
This character is a human person, positioned on the right side.
The scene shows: The rabbit looks up at the woodcutter with a worried expression
in a sunlit forest clearing.
The background is lush green forest, tall trees, dappled sunlight.
Both characters are large and clearly visible in the foreground.
```

- **캐릭터 묘사**: GPT-4o의 `visual_hint`를 직접 사용 (하드코딩 매핑 없음)
- **감정/표정**: LLM이 매 장면마다 캐릭터 감정을 자연어로 기술
- **negative_prompt**: FLUX에서 미지원 → 사용하지 않음

---

## 캐릭터 일관성 전략

### 캐릭터 바이블
LLM(GPT-4o)이 각 캐릭터의 고정 시각 묘사를 생성합니다:
- 체색/피부색
- 고유 액세서리 (빨간 조끼, 금 왕관 등)
- 고유 신체 특징 (큰 귀, 긴 수염 등)

이 묘사가 모든 장면의 프롬프트에 일관되게 삽입됩니다.

### LoRA (스타일 고정)
- QLoRA (NF4 양자화) 방식으로 42장 동화 삽화 학습
- 그림체를 일관되게 유지

---

## 한국 문화 캐릭터 번역 규칙

한국 전래동화 특수 캐릭터의 문화적 오역을 방지합니다:

| 한국어 | 영어 변환 | 이유 |
|--------|-----------|------|
| 선녀 | 천의를 입은 한국 여인 (날개 없음) | 서양 fairy로 오역 방지 |
| 나무꾼 | 도끼를 든 한국 나무꾼 | 판타지 레인저로 오역 방지 |
| 도깨비 | 뿔 달린 한국 도깨비 (방망이) | 일본 오니/서양 악마로 오역 방지 |

---

## 지원 테마 (8종)

| 테마 키 | 설명 |
|---------|------|
| `KOREAN_TRADITIONAL` | 한국 전통 (한옥, 초가집) |
| `FOREST_NATURE` | 숲/자연 |
| `FANTASY_WORLD` | 판타지 세계 |
| `EUROPEAN_MEDIEVAL` | 유럽 중세 |
| `UNDERWATER` | 바닷속 |
| `SKY_HEAVEN` | 하늘/천상 |
| `MODERN_FANTASY` | 현대 판타지 |
| `MIXED` | 혼합 배경 |

---

## 실행 방법

### API 서버
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### CLI 테스트
```bash
python test_pipeline.py              # 기본 동화
python test_pipeline.py --story-a    # 동물 주인공 (토끼+나무꾼+선녀)
python test_pipeline.py --story-b    # 사람 주인공 (릴리아, 유럽 중세)
python test_pipeline.py --seed 42    # 시드 고정
python test_pipeline.py --keep-loaded  # 모델 유지
```

### LoRA 학습
```bash
# 1. 텍스트 임베딩 사전 계산
python model_training/compute_embeddings.py

# 2. LoRA 학습
python -m accelerate.commands.accelerate_cli launch \
  model_training/train_dreambooth_lora_flux_miniature.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
  --instance_data_dir="dataset/flux_train" \
  --data_df_path="model_training/embeddings.parquet" \
  --output_dir="lora_output/flux_lora" \
  --mixed_precision="bf16" --rank=4 --resolution=512 \
  --train_batch_size=1 --learning_rate=1e-4 \
  --max_train_steps=500 --checkpointing_steps=100 \
  --use_8bit_adam --cache_latents --seed=42
```

---

## 주요 설정값

| 설정 | 값 |
|------|-----|
| 모델 | FLUX.1 Schnell (NF4) |
| CFG Scale | 0.0 (고정) |
| Inference Steps | 4 |
| 생성 해상도 | 1024×1024 |
| 출력 해상도 | 387×409 |
| LLM 모델 | GPT-4o |
| 최대 장면 수 | 10 |
| LoRA rank | 4 |
| 학습 해상도 | 512×512 |
