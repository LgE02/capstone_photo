# 시스템 아키텍처 — FLUX.2-klein-4B 동화 삽화 생성 파이프라인

동화 텍스트를 입력받아 **캐릭터 일관성이 유지된** 장면별 삽화를 자동 생성하는 파이프라인.

- **이미지 생성 모델**: FLUX.2-klein-4B (bf16)
- **스토리 분석**: GPT-4o (OpenAI API)
- **프롬프트 방식**: Qwen3 인코더용 자연어 문장 (최대 40,960 토큰)
- **캐릭터 일관성**: 참조 이미지 기반 (레퍼런스 이미지 → 모든 장면에 주입)
- **개발 환경**: RTX 3060 12GB VRAM / RAM 64GB / CUDA 12.x

---

## 1. 모델 개선 이력

### 🔴 1단계: SDXL 기본 적용

**구성**
- 베이스 모델: `stabilityai/stable-diffusion-xl-base-1.0` (SDXL 1.0)
- 프롬프트: CLIP 방식 (positive + negative 분리)
- 설정: 30 steps, guidance_scale=7.5, 1024×1024

**이미지 생성 시간**
| 설정 | 해상도 | 소요시간 (RTX 3060 12GB) |
|------|--------|--------------------------|
| SDXL 기본 | 1024×1024 | **약 20~25초/장** |

**문제점**
| 문제 | 내용 |
|------|------|
| 스타일 제어 불가 | 동화책 삽화 스타일로 유도가 어려움 |
| 캐릭터 일관성 없음 | 장면마다 외형 달라짐 |
| CLIP 토큰 한계 | CLIP-L 77토큰 → 상세 묘사 불가 |
| negative_prompt 의존 | 원치 않는 요소를 negative로 제거해야 함 |
| 한국 전래동화 고증 부족 | 시대·문화 배경 반영 어려움 |

**결론**: 기본 SDXL로는 동화 삽화 스타일 제어 불가. 자체 LoRA 학습 필요.

---

### 🔴 2단계: SDXL + 자체 LoRA 학습 시도

**구성**
- 베이스 모델: SDXL 1.0 + FP16 VAE (`madebyollin/sdxl-vae-fp16-fix`)
- trigger word: `ftbookstyle`
- 데이터셋: `dataset/style_core` (resolution 768)
- 학습 설정: rank=32, alpha=32, lr=8e-5, max_steps=140, batch=2

**문제점**
| 문제 | 내용 |
|------|------|
| 데이터 부족 | 학습 데이터 매우 적음 → 스타일 학습 실패 |
| 140스텝으로 underfitting | 충분히 학습되지 않음 |
| SDXL 자체 VRAM 높음 | 12GB에서 학습 자체가 빠듯 |
| 품질 미달 | ftbookstyle trigger로 일관된 스타일 미생성 |

**결론**: SDXL 학습 인프라 문제 + 데이터 부족. FLUX 계열로 전환.

---

### 🟡 3단계: FLUX.1-schnell + QLoRA 자체 학습

**구성**
- 베이스 모델: `black-forest-labs/FLUX.1-schnell` (12B, T5-XXL 512토큰)
- 양자화: NF4 (bitsandbytes) — transformer ~3GB
- 학습: QLoRA rank=4, lr=1e-4, **500 steps**, batch=1
- 데이터: **42장 동화 삽화** + 자연어 캡션
- 체크포인트: 100, 200, 300, 400, 500 (100 step 간격)

**이미지 생성 시간**
| 설정 | 소요시간 (RTX 3060 12GB) |
|------|--------------------------|
| FLUX.1-schnell NF4, 4 steps | **약 30~45초/장** |

*NF4 양자화 + 12B 파라미터 크기로 인해 SDXL보다 오히려 느림*

**체크포인트 비교 결과**
| 체크포인트 | 품질 |
|-----------|------|
| no_lora (baseline) | ✅ 가장 깨끗한 이미지 |
| checkpoint-100 | 🟡 약간의 스타일 적용 |
| checkpoint-200 | 🔴 이미지 붕괴 시작 (뭉개짐, 색상 번짐) |
| checkpoint-300~500 | 🔴 완전 붕괴 |

**문제점**
| 문제 | 내용 |
|------|------|
| 과학습 (overfitting) | 42장 데이터 → checkpoint-200부터 이미지 붕괴 |
| no_lora가 최고 성능 | LoRA를 쓰는 게 오히려 손해 |
| 캐릭터 일관성 없음 | LoRA가 특정 캐릭터를 기억하지 못함 |
| T5 512토큰 한계 | 상세한 시대 고증 묘사 불가 |
| 느린 속도 | NF4 + 12B → 30~45초/장 |

**결론**: LoRA 방향 포기. no_lora 상태에서 프롬프트 파이프라인으로 전환.

---

### 🟡 4단계: FLUX.1-schnell (no_lora) + GPT-4o 파이프라인

**구성**
- 모델: FLUX.1-schnell NF4 그대로 (LoRA 미적용)
- GPT-4o로 동화 전체를 1회 분석 → 캐릭터 + 장면 프롬프트 자동 생성
- `api/` 폴더에 FastAPI 파이프라인 구축 (SSE 스트리밍)
- 프롬프트 구조: T5 자연어 문장 (512토큰 한도)

**이미지 생성 시간**: 30~45초/장 (동일)

**개선된 점**
- 동화 텍스트 → 자동 장면 분석 (GPT-4o)
- 캐릭터 visual_hint 자동 추출
- FastAPI SSE 스트리밍으로 실시간 진행 상황 전달

**문제점**
| 문제 | 내용 |
|------|------|
| 캐릭터 일관성 없음 | 장면마다 주인공 외형이 완전히 달라짐 |
| 구도 고정 | 모든 장면이 비슷한 정면/와이드샷 반복 |
| 동적 표현 부족 | 캐릭터 표정·행동이 정적, 단조로움 |
| 시대 고증 실패 | 나무꾼이 현대 복장, 조선시대 배경인데 서양 건물 등 |
| 하드코딩 한계 | 직업별 번역 규칙 추가해도 모든 케이스 커버 불가 |
| T5 512토큰 | 상세 묘사 여전히 제한적 |

**결론**: 캐릭터 일관성 문제는 프롬프트 엔지니어링으로 근본 해결 불가. 참조 이미지를 지원하는 모델로 교체 필요.

---

### 🟢 5단계: FLUX.2-klein-4B — 현재 (`api_klein/`)

**구성**
- 베이스 모델: `black-forest-labs/FLUX.2-klein-4B` (4B 파라미터)
- Transformer: `Photoroom/FLUX.2-klein-4b-fp8-diffusers` (bf16 버전, ~7.7GB)
- 텍스트 인코더: Qwen3 (40,960 토큰 — T5 대비 80배)
- 4 steps, guidance_scale=1.0, CPU offload
- `api_klein/` 폴더 — `api/` 완전 독립 (import 없음)

**이미지 생성 시간**
| 설정 | 소요시간 (RTX 3060 12GB) |
|------|--------------------------|
| FLUX.2-klein-4B bf16, 4 steps | **약 10~15초/장** |
| + 캐릭터 레퍼런스 생성 (1회) | +10~15초 (동화당 최초 1회만) |

*FLUX.1-schnell 대비 2~3배 빠름, SDXL 수준 속도에 훨씬 높은 품질*

**개선된 점**
| 항목 | 개선 내용 |
|------|----------|
| 캐릭터 일관성 | 레퍼런스 이미지 1회 생성 → 전 장면 `image=[ref]` 주입 |
| 속도 | 10~15초/장 (이전 30~45초 대비 2~3배 향상) |
| 토큰 한도 | 40,960토큰 → 상세 시대 고증 묘사 가능 |
| 시대 고증 자동화 | 요청의 테마(KOREAN_TRADITIONAL 등)를 기반으로 GPT-4o가 캐릭터별 세부 복식·소품·헤어 묘사 생성 (하드코딩 X) |
| 동적 구도 | LLM이 매 장면 다른 카메라 앵글 지정 (close-up, wide shot, low angle 등) |
| 표정 표현 | 만화적 과장 표현 룰 (눈이 커지기, 말풍선, 물음표 등) |

**현재 미해결 과제**
| 문제 | 원인 | 상태 |
|------|------|------|
| 레퍼런스 있을 때 — 장면 감정/포즈 고정 | `image=[ref]`가 외형과 포즈를 함께 복제 → 장면 프롬프트와 충돌 | 🔴 미해결 |
| 레퍼런스 없을 때 — 캐릭터 외형 매 장면 변함 | 일관성 기준 이미지 없음 → 모델이 매번 다르게 생성 | 🔴 미해결 |
| 한국 전통 복식 편향 (중국 한푸로 출력) | FLUX 학습 데이터가 중국 의상 비중 훨씬 높음 → 프롬프트로 근본 해결 불가 | 🔴 모델 한계 |
| 동물 해부학 오류 (토끼 귀 등) | 프롬프트 묘사 강화로 일부 개선, 완전 해결은 어려움 | 🟡 부분 개선 |

---

## 2. 현재 파이프라인 흐름

```
클라이언트
    │  POST /generate
    │  { story_text, protagonist_type, theme, seed }
    ▼
┌─────────────────────────────────────────────────────────────┐
│  routes.py                                                  │
│  1. JobStore.create() → Job 생성 (job_id, status=queued)    │
│  2. asyncio.Queue 생성 (SSE 이벤트용)                       │
│  3. KleinJobExecutor.submit(job_id, on_event)               │
│     └─ max_queue=3 초과 시 → 503 반환                       │
│  4. EventSourceResponse 반환 (SSE 스트리밍 연결 유지)        │
└─────────────────────────────────────────────────────────────┘
    │  (asyncio.Queue ↔ 백그라운드 스레드 브릿지)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  KleinJobExecutor (단일 워커 스레드 — GPU 독점)              │
│                                                             │
│  [1] 스토리 분석  (job.status = "analyzing")                │
│      build_story_plan(story_text)                           │
│      ├─ GPT-4o  ← 1순위                                    │
│      │   ├─ 캐릭터 시대 고증 visual_description 생성        │
│      │   ├─ 매 장면 카메라 앵글 + 감정/행동 scene_prompt    │
│      │   └─ theme 감지 (LLM 결과)                          │
│      └─ 키워드 매핑  ← GPT-4o 실패 시 fallback             │
│                                                             │
│      ※ LLM 감지 theme ≠ 요청 theme → 요청 theme 우선 적용  │
│         (world/scene_plans 재빌드)                          │
│                                                             │
│      SSE emit → "analyzing" { total_pages }                 │
│                                                             │
│  [2] 캐릭터 레퍼런스 생성                                   │
│      generator.generate_character_reference(char_prompt)    │
│      └─ 저장: outputs/klein_jobs/{job_id}/page_00.png       │
│      SSE emit → "character_reference" { image_url }         │
│                                                             │
│  [3] 페이지별 삽화 생성  (job.status = "generating")        │
│      for scene in story_plan.scenes:                        │
│        generator.generate(prompt, use_reference=True)       │
│        └─ 저장: outputs/klein_jobs/{job_id}/page_NN.png     │
│        SSE emit → "page_complete" { page_index, image_url } │
│                                                             │
│      SSE emit → "complete" { total_pages, total_elapsed }   │
└─────────────────────────────────────────────────────────────┘

클라이언트는 SSE로 실시간 수신:
  analyzing          → { total_pages: 10 }
  character_reference→ { image_url: "/images/{job_id}/page_00.png" }
  page_complete      → { page_index: 1, image_url: "...", elapsed: 12.3 }
  ...
  complete           → { total_pages: 10, total_elapsed: 130.0 }

GET /jobs/{job_id}   → Job 전체 상태 조회 (SSE 연결 끊긴 후 재조회용)
GET /images/{job_id}/page_NN.png  → 생성된 이미지 정적 파일
```

---

## 3. 디렉토리 구조

```
api_klein/                          ← Klein API 진입점 (api/ 완전 독립)
  app.py                            # FastAPI 앱 (포트 8001) + lifespan
  schemas.py                        # Pydantic 요청/응답 모델
  routes.py                         # POST /generate (SSE), GET /jobs/{id}
  tasks.py                          # Job, JobStore, KleinJobExecutor
  pipeline/
    __init__.py
    config.py                       # 모델 설정, 테마 확장, 캐릭터 힌트
    generator_klein.py              # FLUX.2-klein-4B 이미지 생성기
    model_manager.py                # KleinModelManager 싱글톤
    llm_prompt_extractor.py         # GPT-4o 스토리 분석 (시대 고증 자동화)
    story_analyzer.py               # 키워드 매핑 fallback 분석기
    story_pipeline.py               # 완전 독립 파이프라인 + Klein 프롬프트 빌더

test_pipeline_klein.py              # CLI 통합 테스트 (Story A/B, --no-ref 옵션)
outputs/
  test_klein/
    story_a_with_ref/               # 레퍼런스 있음 결과
    story_a_no_ref/                 # 레퍼런스 없음 결과 (--no-ref)
    story_b_with_ref/
    story_b_no_ref/
      reference/                    # 캐릭터 레퍼런스 이미지
  klein_jobs/                       # API 서버 생성 이미지
```

---

## 4. API 엔드포인트

### `POST /generate` — SSE 스트리밍 삽화 생성

```
Request (JSON):
  story_text: str          # \n으로 구분된 동화 전문
  protagonist_type: str    # "human" | "animal" | "other"
  theme: str               # 8종 테마 키
  seed: int | None

Response: SSE (text/event-stream)
  event: analyzing           → {"total_pages": 10}
  event: character_reference → {"image_url": "/images/{job_id}/page_00.png"}
  event: page_complete       → {"page_index": 1, "image_url": "...", "elapsed": 12.3}
  ...
  event: complete            → {"total_pages": 10, "total_elapsed": 130.0}
  event: error               → {"message": "...", "page_index": 3}
```

### `GET /jobs/{job_id}` — Job 상태 조회

```json
{
  "job_id": "...",
  "status": "completed",
  "total_pages": 10,
  "character_reference_url": "/images/{job_id}/page_00.png",
  "pages": [
    {"page_index": 1, "source_text": "...", "image_url": "...", "elapsed": 12.3}
  ],
  "total_elapsed": 130.0
}
```

---

## 5. 핵심 설계 결정

### GPT-4o 기반 캐릭터 시대 고증 자동화 (하드코딩 제거)

요청에는 이미 `theme` (KOREAN_TRADITIONAL 등)이 포함되어 있다.
GPT-4o의 역할은 테마를 감지하는 게 아니라, 그 테마 안에서 **캐릭터별 세부 외형을 정확하게 묘사**하는 것.

**기존 방식 (문제)**
```python
# ❌ 나무꾼만 하드코딩 → 등장인물이 바뀌면 적용 안 됨
"나무꾼 → wearing rough beige hemp jeogori..."
"선녀 → flowing white Korean cheonui..."
```

**현재 방식 (해결)**
```
GPT-4o에게: "요청 테마(KOREAN_TRADITIONAL)를 기반으로,
            이 캐릭터가 실제 그 시대/문화에서 어떤 복식·외형을 가졌는지
            네 지식으로 직접 묘사하라."
```
- 어떤 직업/캐릭터가 나와도 GPT-4o가 시대 고증된 묘사 생성
- 조선시대면 한복, 중세 유럽이면 tunic/chainmail, 판타지면 판타지 의상

### 캐릭터 일관성 (레퍼런스 방식)

```
1회: generate_character_reference(visual_hint)
     → self._character_reference 이미지 저장

이후 모든 장면: generate(prompt, image=[self._character_reference])
     → 모델이 레퍼런스 이미지의 캐릭터 외형을 유지하며 다른 배경/포즈 생성
```

**알려진 트레이드오프**
| 모드 | 장점 | 단점 |
|------|------|------|
| `--no-ref` (레퍼런스 없음) | 장면 감정·포즈 자유롭게 표현 | 매 장면 캐릭터 외형이 달라짐 |
| 기본 (레퍼런스 있음) | 캐릭터 외형 일관성 유지 | `image=[ref]`가 포즈까지 복제 → 표정·구도 제한 |

→ 현재로서는 스토리 표현력과 캐릭터 일관성을 동시에 완전히 해결하는 방법 없음.

### LLM 장면 프롬프트 우선 사용

프롬프트 빌더는 **구조만 잡고**, 실제 내용(구도, 감정, 행동)은 GPT-4o 결과를 그대로 사용:
```
[스타일]  Korean children's book illustration, soft cell shading...
[캐릭터]  {LLM visual_hint — 시대 고증 포함}
[장면]    Scene: {LLM scene_prompt — 카메라 앵글 + 감정 + 행동}
[배경]    Setting: {테마 기반 배경 힌트}
```

---

## 6. 모델 스펙

| 항목 | 값 |
|------|-----|
| 파라미터 | 4B |
| 텍스트 인코더 | Qwen3 (40,960 토큰) |
| 추론 스텝 | 4 (distilled) |
| guidance_scale | 1.0 |
| 참조 이미지 | 멀티 레퍼런스 지원 |
| Transformer 크기 (bf16) | ~7.7GB |
| Transformer 크기 (FP8) | ~3.9GB |
| 필요 VRAM (bf16 + CPU offload) | 8~10GB |
| 생성 속도 | 약 10~15초/장 |
| 라이선스 | Apache 2.0 |

| 가중치 | 출처 |
|--------|------|
| Transformer (bf16/FP8) | `Photoroom/FLUX.2-klein-4b-fp8-diffusers` |
| T5 / VAE / CLIP | `black-forest-labs/FLUX.2-klein-4B` |

---

## 7. 실행

```bash
# API 서버 (포트 8001)
uvicorn api_klein.app:app --host 0.0.0.0 --port 8001

# CLI 통합 테스트
python test_pipeline_klein.py --story-a           # 한국 전통: 토끼+나무꾼+선녀 (레퍼런스 있음)
python test_pipeline_klein.py --story-a --no-ref  # 레퍼런스 없이 생성 (표정 자유, 일관성 낮음)
python test_pipeline_klein.py --story-b           # 유럽 중세: 릴리아+요정
python test_pipeline_klein.py --seed 42 --output outputs/custom_dir
```

---

## 8. 테마 시스템 (8종)

| 테마 키 | 배경 힌트 |
|---------|----------|
| `KOREAN_TRADITIONAL` | Korean hanok, curved tiled roof, dancheong eaves |
| `FOREST_NATURE` | lush green forest, tall trees, dappled sunlight |
| `FANTASY_WORLD` | magical landscape, enchanted forest, glowing particles |
| `EUROPEAN_MEDIEVAL` | cobblestone street, half-timbered houses, stone castle |
| `UNDERWATER` | colorful coral reef, bubbles, light rays through water |
| `SKY_HEAVEN` | celestial sky, fluffy clouds, golden sunlight |
| `MODERN_FANTASY` | modern city with magical elements, glowing lights |
| `MIXED` | colorful storybook background, warm lighting |
