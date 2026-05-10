# 동화 삽화 생성 프로젝트

4~6세 아동용 동화 텍스트를 입력하면 캐릭터 일관성이 유지된 장면별 삽화를 자동 생성하는 파이프라인.

자세한 설계 결정과 모델 개선 이력은 [ARCHITECTURE_KLEIN.md](ARCHITECTURE_KLEIN.md) 참고.

---

## 핵심 스펙

| 항목 | 값 |
|------|-----|
| 이미지 생성 모델 | FLUX.2-klein-4B (bf16 → torchao FP8 변환 시도) |
| 텍스트 인코더 | Qwen3 (40,960 토큰) |
| 추론 스텝 | 4 (distilled), guidance_scale=1.0 |
| 스토리 분석 | GPT-4o (OpenAI API) |
| 캐릭터 일관성 | 레퍼런스 이미지 1회 생성 → 전 장면 주입 |
| 출력 해상도 | 1024×1024 PNG |
| 생성 속도 | 약 10~15초/장 (RTX 3060 12GB) |
| 라이선스 | Apache 2.0 |

---

## 디렉터리 구조

```text
fairytale_lora/
├─ api_klein/                       # FastAPI 서버 (포트 8001)
│  ├─ app.py                        # 앱 + lifespan (모델 로드/언로드)
│  ├─ schemas.py                    # Pydantic 요청/응답 모델
│  ├─ routes.py                     # POST /generate (SSE), GET /jobs/{id}
│  ├─ tasks.py                      # Job, JobStore, KleinJobExecutor
│  └─ pipeline/
│     ├─ config.py                  # 출력/테마 설정
│     ├─ generator_klein.py         # FLUX.2-klein-4B 이미지 생성기
│     ├─ model_manager.py           # KleinModelManager 싱글톤
│     ├─ llm_prompt_extractor.py    # GPT-4o 스토리 분석
│     └─ story_pipeline.py          # 분석 결과 → Klein 자연어 프롬프트
├─ outputs/                         # 생성 결과 저장
└─ test_pipeline_klein.py           # CLI 통합 테스트
```

---

## 파이프라인 흐름

```
동화 텍스트 (한국어)
    │
    ▼
[1] GPT-4o 분석 (llm_prompt_extractor)
    - 캐릭터 추출 + 시대 고증된 visual_description
    - 장면별 자연어 prompt (카메라 앵글 + 감정/행동 포함)
    - 테마 자동 감지 (요청 theme이 우선)
    │
    ▼
[2] 스토리 플랜 생성 (story_pipeline)
    - 캐릭터 Bible (visual_hint 그대로 사용)
    - 세계관 프로필 (테마별 배경 힌트)
    - Qwen3 자연어 프롬프트 조립
    │
    ▼
[3] 캐릭터 레퍼런스 1장 생성 (generator_klein)
    - 동화 시작 시 1회만 생성
    - 모든 장면에 image=[ref]로 주입 → 외형 일관성 유지
    │
    ▼
[4] 페이지별 삽화 생성
    - 4 steps, guidance 1.0, 1024×1024
    - SSE로 실시간 스트리밍
```

---

## 실행

### API 서버 (포트 8001)

```bash
uvicorn api_klein.app:app --host 0.0.0.0 --port 8001
```

엔드포인트:
- `POST /generate` — SSE 스트리밍 삽화 생성
- `GET /jobs/{job_id}` — Job 상태 조회
- `GET /images/{job_id}/page_NN.png` — 생성된 이미지

### CLI 통합 테스트

```bash
python test_pipeline_klein.py --story-a            # 한국 전통: 토끼+나무꾼+선녀
python test_pipeline_klein.py --story-b            # 유럽 중세: 릴리아+요정
python test_pipeline_klein.py --story-a --no-ref   # 레퍼런스 없이 생성
python test_pipeline_klein.py --seed 42 --output outputs/custom_dir
```

### 의존성

```bash
pip install -r requirements.txt
```

`.env` 에 `OPENAI_API_KEY` 설정 필요.

---

## 캐릭터 일관성 트레이드오프

| 모드 | 장점 | 단점 |
|------|------|------|
| 기본 (레퍼런스 있음) | 캐릭터 외형 일관성 유지 | `image=[ref]`가 포즈까지 복제 → 표정·구도 제한 |
| `--no-ref` (레퍼런스 없음) | 장면 감정·포즈 자유 | 매 장면 캐릭터 외형이 달라짐 |

→ 자세한 설계 결정 / 트레이드오프 분석은 [ARCHITECTURE_KLEIN.md](ARCHITECTURE_KLEIN.md) 5절 참고.

---

## 지원 테마 (8종)

| 테마 키 | 배경 |
|---------|------|
| `KOREAN_TRADITIONAL` | 한옥, 단청 처마 |
| `FOREST_NATURE` | 숲/자연 |
| `FANTASY_WORLD` | 판타지 마법 풍경 |
| `EUROPEAN_MEDIEVAL` | 유럽 중세 마을 |
| `UNDERWATER` | 바닷속 |
| `SKY_HEAVEN` | 하늘/천상 |
| `MODERN_FANTASY` | 현대 판타지 |
| `MIXED` | 혼합 배경 |
