# Fairytale LoRA Worker

동화 텍스트를 바탕으로 캐릭터 설정을 만들고, 페이지별 삽화를 생성한 뒤 S3와 Kafka로 결과를 전달하는 워커 프로젝트입니다.

현재 구현은 `worker.py`를 시작점으로 사용하는 Kafka 소비형 파이프라인 기준으로 정리되어 있습니다. 상세 설계 배경은 같은 저장소의 `ARCHITECTURE_KLEIN.md`를 참고하면 됩니다.

## 개요

- 입력: Spring 등 외부 시스템이 Kafka로 보내는 동화 생성 이벤트
- 분석: OpenAI Chat Completion (기본 `gpt-5.5`)로 캐릭터 바이블과 페이지 장면 정보 추출
- 생성: `FLUX.2-klein-4B` 기반 이미지 생성
- 저장: 캐릭터 reference 이미지와 페이지 이미지를 S3에 업로드
- 결과 전달: 생성 완료 이미지를 Kafka 결과 토픽으로 publish

## 처리 흐름

### 1. 초기화 메시지 처리

`KAFKA_TOPIC_INIT` 기본값은 `fairytale_created` 입니다.

이 토픽의 메시지를 받으면 워커는 다음 작업을 수행합니다.

- `fairytaleId`, `setting`, `char_species`, `characters`를 읽음
- GPT-5.5로 역할별 `visual_description`을 생성 (종별 anatomy + acorn 회피 등 가드 강제)
- `bible.json`을 S3에 저장
- 역할별 캐릭터 reference 이미지를 생성해서 S3에 저장
- 이후 페이지 생성에서 재사용할 수 있도록 메모리 캐시에 올림

### 2. 페이지 메시지 처리

`KAFKA_TOPIC_PAGE` 기본값은 `fairytale_paragraph` 입니다.

이 토픽의 메시지를 받으면 워커는 다음 작업을 수행합니다.

- `fairytaleId`, `pageNo`, `sentences`를 읽음
- 이미 생성된 페이지가 있으면 재생성하지 않고 기존 URL을 재전송
- `bible.json`을 불러와 페이지 장면을 분석
- 장면에 실제로 등장하는 역할만 골라 reference 이미지 로드
- 페이지 프롬프트를 조합해 삽화 생성
- 결과 이미지를 S3에 업로드
- 결과 Kafka 토픽으로 `{fairytaleId, pageNo, imageurl}` 전송

### 3. 결과 토픽 publish

`KAFKA_TOPIC_RESULT` 기본값은 `fairytale_image` 입니다.

페이지 생성이 완료되면 아래 형식으로 결과를 보냅니다.

```json
{
  "fairytaleId": 17,
  "pageNo": 3,
  "imageurl": "https://.../fairytales/17/pages/page_03.png"
}
```

## 메시지 형식

### INIT 메시지 예시

```json
{
  "fairytaleId": 17,
  "setting": "KOREAN_TRADITIONAL",
  "char_species": "HUMAN",
  "characters": {
    "HERO": "나무꾼",
    "HELPER": "선녀",
    "VILLAIN": "호랑이"
  }
}
```

### PAGE 메시지 예시

한 페이지의 `sentences`는 1~3문장 가변입니다. 줄바꿈 문자열 또는 문자열 배열 모두 허용합니다.

```json
{
  "fairytaleId": 17,
  "pageNo": 1,
  "sentences": "나무꾼이 산길을 걸어갔어요.\n멀리서 신비한 빛이 보였어요."
}
```

## 저장 구조

S3에는 아래와 같은 구조로 결과가 저장됩니다.

```text
fairytales/{fairytaleId}/
  bible.json
  references/
    HERO.png
    VILLAIN.png
    HELPER.png
    DISPATCHER.png
    FALSE_HERO.png
    DONOR.png
  pages/
    page_01.png
    page_02.png
```

## 주요 구성 요소

```text
fairytale_lora/
  worker.py                           # 워커 진입점
  api_klein/
    consumer.py                       # Kafka consumer 루프
    processor.py                      # INIT/PAGE 메시지 분기와 처리
    publisher.py                      # 결과 Kafka publish
    storage.py                        # S3 업로드/다운로드
    pipeline/
      config.py                       # 테마별 배경/네거티브 힌트 설정
      generator_klein.py              # FLUX 이미지 생성기
      llm_prompt_extractor.py         # OpenAI Chat Completion 기반 캐릭터/장면 분석 (종별 anatomy + SINGLE BEAT + PASSIVE POSITIONING 가드)
      model_manager.py                # 싱글턴 모델 로더
      story_pipeline.py               # 테마 키 → 배경 힌트(positive/negative) 변환
```

## 모델 및 생성 설정

- 이미지 생성 모델: `black-forest-labs/FLUX.2-klein-4B`
- transformer repo: `Photoroom/FLUX.2-klein-4b-fp8-diffusers`
- 텍스트 분석 모델: `gpt-5.5`
- 출력 해상도: `768x768` (속도-품질 트레이드오프 — `OUTPUT_RESOLUTION` 상수)
- 추론 스텝: `4` (`NUM_INFERENCE_STEPS` 상수 — Klein이 4-step에 최적화돼 있어 3 이하로 내리면 깨짐)
- `guidance_scale`: `1.0`
- reference 이미지: 등장 역할 기준 multi-image 입력 지원
- 메모리 최적화: CPU offload, `torchao` FP8 양자화 시도 후 실패 시 BF16 fallback

## 지원 테마

- `KOREAN_TRADITIONAL`
- `FOREST_NATURE`
- `FANTASY_WORLD`
- `EUROPEAN_MEDIEVAL`
- `UNDERWATER`
- `SKY_HEAVEN`
- `MODERN_FANTASY`
- `MIXED`

각 테마는 배경 힌트와 의상 힌트를 가지며, 장면 프롬프트와 캐릭터 설명 조합에 사용됩니다.

## 설치

```bash
pip install -r requirements.txt
```

선택적으로 VRAM 절감을 위해 `torchao`를 추가 설치할 수 있습니다.

```bash
pip install torchao
```

## 환경 변수

실행 전 `.env`에 아래 값을 준비해야 합니다.

### 필수

- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `KAFKA_BOOTSTRAP_SERVERS`

### 선택

- `AWS_REGION` 기본값 `ap-northeast-2`
- `KAFKA_TOPIC_INIT` 기본값 `fairytale_created`
- `KAFKA_TOPIC_PAGE` 기본값 `fairytale_paragraph`
- `KAFKA_TOPIC_RESULT` 기본값 `fairytale_image`
- `KAFKA_GROUP_ID` 기본값 `illustration-worker`
- `KAFKA_SECURITY_PROTOCOL` 기본값 `PLAINTEXT`
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`
- `OPENAI_MODEL` 기본값 `gpt-5.5` — 텍스트 분석에 쓰는 OpenAI 모델 ID. 워커 시작 시 1회 읽으므로 교체 후 재시작 필요.

## 실행

```bash
python worker.py
```

실행 시 순서는 다음과 같습니다.

1. `.env` 로드
2. 필수 환경 변수 확인
3. FLUX 모델 로드
4. S3와 Kafka publisher 초기화
5. Kafka consumer 루프 시작

## 구현 특징

### 메시지 처리
- INIT와 PAGE를 서로 다른 Kafka 토픽으로 분리
- PAGE가 INIT보다 먼저 도착하면 `BibleNotReadyError` → commit 보류 + 재시도
- 동일 페이지가 이미 있으면 재생성 없이 기존 URL 재전송 (idempotent)
- INIT도 idempotent — bible + reference가 S3에 있으면 skip
- Kafka consumer는 수동 commit (성공 시에만 commit)
- 실패 시 exponential backoff + 최대 재시도, 한계 초과 시 DLQ 토픽(`fairytale_dlq` 기본값)으로 격리

### 프롬프트 가드 (GPT 단계)
- **종별 anatomy** — BIRD/REPTILE/FISH/AMPHIBIAN/MAMMAL 별로 단어 선택 강제 (새한테 `fur` 금지, 측면 시 한쪽 눈만 등)
- **acorn 회피** — 새의 prop 식상화 방지
- **SINGLE BEAT RULE** — 한 페이지 = 한 액션 1명만 수행, 나머지는 배경 존재 (페이지가 동작 여러 개로 어색해지는 것 방지)
- **단수 한정사 self-check** — 모든 등장 캐릭터 앞에 `"the lone X"` 자가점검 (캐릭터 복제 완화)
- **PASSIVE CHARACTER POSITIONING** — 액션 안 하는 캐릭터에 단일 anchor + pose verb + 가시 부위 명시 강제
- **NO EXTRA CHARACTERS** — "친구들" 같은 한국어 표현이 들어와도 무명 캐릭터 안 그림

### 코드 단 가드
- 페이지당 reference 이미지 최대 3장 (4명+ 등장 시 자동 컷)
- ANIMAL 캐릭터 중 새 키워드 감지 시 페이지 프롬프트 안전망 자동 추가

### 모델 사용
- 텍스트 분석: `OPENAI_MODEL` (기본 `gpt-5.5`) + `reasoning_effort=low`
- 이미지 생성: FLUX.2-klein-4B, 768×768, 4-step, `guidance_scale=1.0`
- GPT-5.5 호환성 처리: `max_completion_tokens` 사용, `temperature` 1.0 강제

## 테스트

로컬에서 워커에 메시지를 흘려보낼 때는 `test_send.py`를 사용합니다.

```bash
# 단일 페이지 publish
python test_send.py --id 9001 --sentences "..."

# INIT만 publish (캐릭터 지정)
python test_send.py --id 9001 --init-only --char-species ANIMAL \
  --character HERO=까마귀 --character VILLAIN=여우

# 동화 1편 시퀀스 publish (JSON 파일)
python test_send.py --id 9001 --story story.json
```

`--story` 모드의 JSON 형식은 [test_send.py](test_send.py) 상단 docstring 참고.

## 참고

- 구조 및 설계 메모: `ARCHITECTURE_KLEIN.md`
- 현재 README는 워커 중심 동작만 정리했습니다.
