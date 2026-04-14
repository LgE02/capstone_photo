# 시스템 아키텍처 — 동화 삽화 생성 파이프라인

동화 텍스트를 입력받아 장면별 삽화를 자동 생성하는 파이프라인의 **내부 동작 방식**을 기술한다.

- **이미지 생성 모델**: FLUX.1 Schnell (NF4 양자화)
- **스토리 분석**: GPT-4o (OpenAI API)
- **프롬프트 방식**: T5 인코더용 자연어 문장

---

## 1. 전체 흐름

```
클라이언트 (동화 생성 모델)
    │  POST /generate
    │  { story_text, protagonist_type, theme, seed }
    ▼
┌──────────────────────────────────────┐
│  FastAPI 서버 (api/app.py)            │
│  ├─ SSE 스트리밍 응답 연결            │
│  └─ JobExecutor 큐에 작업 등록        │
└──────────────────────────────────────┘
    │  백그라운드 워커 스레드
    ▼
┌──────────────────────────────────────┐
│  1. 스토리 분석                       │
│  build_story_plan(story_text)        │  pipeline/story_pipeline.py
│  ├─ LLM 분석 (GPT-4o)  ← 1순위      │  pipeline/llm_prompt_extractor.py
│  └─ 키워드 매핑         ← fallback   │  pipeline/story_analyzer.py
│  → StoryPlan (캐릭터, 테마, 장면)     │
└──────────────────────────────────────┘
    │  SSE: event: analyzing
    ▼
┌──────────────────────────────────────┐
│  2. 페이지별 이미지 생성              │
│  for scene in story_plan.scenes:     │
│    generator.generate(prompt)        │  pipeline/generator.py
│    └─ FLUX Schnell NF4 추론          │
│       단일 자연어 프롬프트             │
│       4 steps, guidance_scale=0.0    │
│  SSE: event: page_complete (매 페이지)│
└──────────────────────────────────────┘
    │  SSE: event: complete
    ▼
  이미지 URL 반환 (/images/{job_id}/page_01.png)
```

---

## 2. 디렉토리 구조

```
api/                          ← 진입점 (FastAPI 서버)
  app.py                      # FastAPI 앱 + lifespan (모델 로드/언로드)
  schemas.py                  # Pydantic 요청/응답 모델
  routes.py                   # POST /generate (SSE), GET /jobs/{id}
  tasks.py                    # Job, JobStore, JobExecutor, ImageStorage
  pipeline/                   ← 이미지 생성 파이프라인 (내부)
    config.py                 # 모델/테마 설정 (FLUX Schnell)
    generator.py              # FLUX Schnell NF4 이미지 생성기
    model_manager.py          # 싱글톤 모델 매니저
    story_pipeline.py         # 스토리 → T5 자연어 프롬프트 변환
    llm_prompt_extractor.py   # GPT-4o 스토리 분석
    story_analyzer.py         # 키워드 매핑 fallback

model_training/               ← LoRA 학습
  compute_embeddings.py       # 텍스트 임베딩 사전 계산
  train_dreambooth_lora_flux_miniature.py  # 공식 FLUX LoRA 학습 스크립트
  train_config_flux.py        # 학습 설정
  run_flux_train.bat          # 학습 실행 스크립트

dataset/
  flux_train/                 # 학습 데이터 (42장 PNG + 자연어 캡션)

test_pipeline.py              # CLI 테스트 스크립트
```

### 실행

```bash
# API 서버
uvicorn api.app:app --host 0.0.0.0 --port 8000

# CLI 테스트
python test_pipeline.py --story-a   # Story A: 동물 주인공
python test_pipeline.py --story-b   # Story B: 사람 주인공

# LoRA 학습
python model_training/compute_embeddings.py
python -m accelerate.commands.accelerate_cli launch model_training/train_dreambooth_lora_flux_miniature.py ...
```

### 엔드포인트

#### `POST /generate` — SSE 스트리밍 삽화 생성
```
Request (JSON):
  story_text: str          # \n으로 구분된 동화 전문
  protagonist_type: str    # "human" | "animal" | "other"
  theme: str               # 8종 테마 키
  seed: int | None

Response: SSE (text/event-stream)
  event: analyzing      → {"total_pages": 10}
  event: page_complete  → {"page_index": 1, "image_url": "...", "elapsed": 11.2}
  event: page_complete  → {"page_index": 2, ...}
  ...
  event: complete       → {"total_pages": 10, "total_elapsed": 112.0}
  event: error          → {"message": "...", "page_index": 3}
```

#### `GET /jobs/{job_id}` — Job 상태 조회 (디버깅용)

### 서버 라이프사이클
- **시작**: ModelManager → FLUX Schnell NF4 로드 → JobExecutor 워커 시작
- **요청**: SSE 연결 → Job 큐 등록 → 워커가 순차 처리 → 페이지별 이벤트 전송
- **종료**: 모델 언로드

### GPU 독점 워커 (`tasks.py`)
- 단일 백그라운드 스레드 + `queue.Queue(maxsize=3)`
- 한 번에 하나의 Job만 GPU 사용
- 큐 가득 차면 HTTP 503 반환

### 이미지 저장 (추상화)
```
ImageStorage (ABC)
  ├─ LocalStorage   # 현재: 로컬 디스크 + FastAPI static files
  └─ S3Storage      # 추후: AWS S3 업로드
```
- 현재: `outputs/jobs/{job_id}/page_01.png` → `/images/{job_id}/page_01.png`

---

## 3. 코어 파이프라인 모듈

### `llm_prompt_extractor.py` — LLM 스토리 분석
- **모델**: GPT-4o (OpenAI API)
- **입력**: 한국어 동화 전체 텍스트
- **출력**: JSON (theme, characters[], scenes[])
- **핵심 규칙**:
  - `visual_description`: 체색 + 고유 액세서리 + 고유 신체 특징 + 크기 (4가지 필수)
  - `scene_prompt`: 자연어 문장 (최대 30단어), 감정/표정 필수 포함
  - 다중 캐릭터: 좌/우 위치 명시
  - 한국 문화 캐릭터 번역 규칙 (선녀, 나무꾼, 도깨비 등)

### `story_pipeline.py` — 프롬프트 조립 엔진
1. **`normalize_story_input()`**: LLM 결과 → `StoryInput`
2. **`build_world_profile()`**: 테마 → 배경 힌트
3. **`build_character_bible()`**: 캐릭터별 고정 시각 묘사 (LLM visual_hint 직접 사용)
4. **`build_scene_plans()`**: 장면별 자연어 프롬프트 생성
5. **`build_prompt()`**: T5용 자연어 단일 프롬프트 조립

### `generator.py` — FLUX 이미지 생성기
- 베이스 모델: `black-forest-labs/FLUX.1-schnell`
- 양자화: NF4 (bitsandbytes) — transformer만 4비트, ~3GB
- CPU offload: T5/CLIP/VAE는 필요 시에만 GPU
- 추론: `guidance_scale=0.0`, `num_inference_steps=4`

### `model_manager.py` — 싱글톤 모델 매니저
- 프로세스 전역 싱글톤 (thread-safe)
- 서버 시작 시 1회 로드, 요청마다 재사용

---

## 4. FLUX T5 프롬프트 구조

SDXL의 듀얼 프롬프트(CLIP-L + CLIP-G)와 달리, FLUX는 **단일 T5 자연어 프롬프트**를 사용한다.

### 프롬프트 구성 (build_prompt)

```
[스타일] A children's picture book illustration in flat cartoon style
         with bold outlines and soft pastel colors.
[캐릭터] There are exactly two separate characters.
         First character: {LLM visual_hint}. This character is an animal...
         Second character: {LLM visual_hint}. This character is a human...
[장면]   The scene shows: {LLM scene_prompt with emotion/action}.
[배경]   The background is {theme positive hint}.
[구도]   Both characters are large and clearly visible in the foreground.
```

- **캐릭터 묘사**: GPT-4o의 `visual_hint`를 직접 사용 (하드코딩 매핑 없음)
- **negative_prompt**: FLUX에서 미지원 → 사용하지 않음
- **T5 토큰 한도**: 512 (CLIP 77보다 훨씬 여유)

---

## 5. LoRA 학습 (QLoRA)

### 학습 환경
- GPU: 12GB VRAM
- 양자화: NF4 (transformer ~3GB)
- 옵티마이저: 8bit AdamW
- 데이터셋: 42장 동화 삽화 + 자연어 캡션

### 학습 파라미터
| 설정 | 값 |
|------|-----|
| LoRA rank | 4 |
| 학습률 | 1e-4 |
| 스텝 | 500 |
| batch size | 1 |
| 해상도 | 512x512 |
| 스케줄러 | cosine |

### 학습 흐름
1. `compute_embeddings.py` → 텍스트 임베딩 사전 계산 (parquet)
2. `train_dreambooth_lora_flux_miniature.py` → QLoRA 학습
3. 100 step마다 체크포인트 저장 → 최적 체크포인트 선별

---

## 6. 테마 시스템 (8종)

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

## 7. 주요 설정값

| 설정 | 값 | 위치 |
|------|-----|------|
| 모델 | FLUX.1 Schnell | `config.py` |
| 양자화 | NF4 (bitsandbytes) | `generator.py` |
| CFG Scale | 0.0 (고정) | `config.py` |
| Inference Steps | 4 | `config.py` |
| 생성 해상도 | 1024x1024 | `tasks.py` |
| 출력 해상도 | 387x409 | `config.py` |
| LLM 모델 | GPT-4o | `llm_prompt_extractor.py` |
| 최대 장면 수 | 10 | `llm_prompt_extractor.py` |

---

## 8. 테스트 스토리

| 이름 | 주인공 | 테마 | 캐릭터 |
|------|--------|------|--------|
| Story A | 동물 (토끼) | KOREAN_TRADITIONAL | 토끼 + 나무꾼 + 선녀 |
| Story B | 사람 (릴리아) | EUROPEAN_MEDIEVAL | 릴리아 + 요정들 |

`test_pipeline.py`에서 `--story-a` / `--story-b` 플래그로 선택 가능.
