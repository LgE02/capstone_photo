# 시스템 아키텍처 — 동화 삽화 생성 파이프라인

동화 텍스트를 입력받아 장면별 삽화를 자동 생성하는 파이프라인의 **내부 동작 방식**을 기술한다.

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
│    generator.generate()              │  pipeline/generator.py
│    ├─ LoRA 스케일 적용               │
│    ├─ IP-Adapter 스케일 적용         │
│    ├─ SDXL 듀얼 프롬프트 전달        │
│    └─ 이미지 생성 → 로컬 저장        │
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
    config.py                 # 모델/테마/스타일 설정
    generator.py              # SDXL + LoRA + IP-Adapter 이미지 생성
    model_manager.py          # 싱글톤 모델 매니저
    story_pipeline.py         # 스토리 → 프롬프트 변환
    llm_prompt_extractor.py   # GPT-4o 스토리 분석
    story_analyzer.py         # 키워드 매핑 fallback
test_pipeline.py              # CLI 테스트 스크립트
```

### 실행

```bash
# API 서버
uvicorn api.app:app --host 0.0.0.0 --port 8000

# CLI 테스트
python test_pipeline.py --story-a   # Story A: 동물 주인공
python test_pipeline.py --story-b   # Story B: 사람 주인공
```

### 엔드포인트

#### `POST /generate` — SSE 스트리밍 삽화 생성
```
Request (JSON):
  story_text: str          # \n으로 구분된 동화 전문
  protagonist_type: str    # "human" | "animal" | "other"
  theme: str               # 8종 테마 키
  seed: int | None
  lora_key: str

Response: SSE (text/event-stream)
  event: analyzing      → {"total_pages": 20}
  event: page_complete  → {"page_index": 1, "image_url": "...", "elapsed": 14.8}
  event: page_complete  → {"page_index": 2, ...}
  ...
  event: complete       → {"total_pages": 20, "total_elapsed": 295.0}
  event: error          → {"message": "...", "page_index": 3}
```

#### `GET /jobs/{job_id}` — Job 상태 조회 (디버깅용)

### 서버 라이프사이클
- **시작**: ModelManager → SDXL + LoRA 로드 → IP-Adapter 로드 → JobExecutor 워커 시작
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
  - 동물 캐릭터: `"animal body, NOT human"` 강제
  - 다중 캐릭터: `"character A on left, character B on right"` 위치 명시

### `story_pipeline.py` — 프롬프트 조립 엔진
1. **`normalize_story_input()`**: LLM 결과 → `StoryInput`
2. **`build_world_profile()`**: 테마 → 배경/의상 힌트
3. **`build_character_bible()`**: 캐릭터별 고정 시각 묘사
4. **`build_scene_plans()`**: 장면별 듀얼 프롬프트 + 네거티브 + LoRA 스케일

### `generator.py` — SDXL 이미지 생성기
- 베이스 모델: `stabilityai/stable-diffusion-xl-base-1.0`
- 스케줄러: DPM++ 2M Karras
- LoRA: PEFT → diffusers 키 변환 후 로드
- IP-Adapter: `h94/IP-Adapter` (sdxl_models)

### `model_manager.py` — 싱글톤 모델 매니저
- 프로세스 전역 싱글톤 (thread-safe)
- 서버 시작 시 1회 로드, 요청마다 재사용

---

## 4. SDXL 듀얼 프롬프트 구조

| 인코더 | 파라미터 | 역할 | 토큰 한도 |
|--------|----------|------|-----------|
| CLIP-L | `prompt` | 트리거 + 캐릭터 외형 상세 | 77 |
| CLIP-G | `prompt_2` | 스타일 + 테마 배경 + 캐릭터 종류 + 장면 + 구도 | 77 |

---

## 5. 캐릭터 타입 보호

다중 캐릭터 장면의 BREAK 구간에서:
- 동물: `"{desc}, {species} animal NOT human"`
- 사람: `"{desc}, human person NOT animal"`

네거티브 보호:
- 전원 동물: `"human, person, human face"` 추가
- 동물+사람 혼재: `"anthropomorphic, kemonomimi"` 추가
- 다중 캐릭터: `"hybrid creature, merged characters, fused body"` 추가

---

## 6. IP-Adapter / LoRA 동적 제어

| 장면 유형 | IP-Adapter | LoRA Scale |
|-----------|------------|------------|
| 단일 캐릭터 | 0.2 (P1 참조) | 0.8 |
| 다중 캐릭터 | 0.05 (스타일만) | 0.70 |
| 참조 없음 | 0.0 (더미) | — |

P1 결과를 `ref_image`로 저장 → P2~이후 스타일 앵커.

---

## 7. 테마 시스템 (8종)

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

## 8. 주요 설정값

| 설정 | 값 | 위치 |
|------|-----|------|
| CFG Scale | 8.0 | `config.py` |
| Inference Steps | 30 | `config.py` |
| 스케줄러 | DPM++ 2M Karras | `generator.py` |
| LoRA Scale (solo) | 0.8 | `story_pipeline.py` |
| LoRA Scale (multi) | 0.70 | `story_pipeline.py` |
| IP-Adapter Scale (solo) | 0.2 | `generator.py` |
| IP-Adapter Scale (multi) | 0.05 | `tasks.py` |
| 생성 해상도 | 1024x1024 | `tasks.py` |
| 출력 해상도 | 387x409 | `config.py` |
| 트리거 단어 | `ftbookstyle` | `config.py` |
| LLM 모델 | GPT-4o | `llm_prompt_extractor.py` |
| 최대 장면 수 | 10 | `llm_prompt_extractor.py` |

---

## 9. 테스트 스토리

| 이름 | 주인공 | 테마 | 캐릭터 |
|------|--------|------|--------|
| Story A | 동물 (토끼) | FOREST_NATURE | 토끼 + 나무꾼 + 선녀 |
| Story B | 사람 (릴리아) | EUROPEAN_MEDIEVAL | 릴리아 + 요정들 |

`test_pipeline.py`에서 `--story-a` / `--story-b` 플래그로 선택 가능.
