# Fairytale LoRA Worker

동화 텍스트를 바탕으로 캐릭터 설정을 만들고, 페이지별 삽화를 생성한 뒤 S3와 Kafka로 결과를 전달하는 워커 프로젝트입니다.

현재 구현은 `worker.py`를 시작점으로 사용하는 Kafka 소비형 파이프라인 기준으로 정리되어 있습니다. 상세 설계 배경은 같은 저장소의 `ARCHITECTURE_KLEIN.md`를 참고하면 됩니다.

## 개요

- 입력: Spring 등 외부 시스템이 Kafka로 보내는 동화 생성 이벤트
- 분석: GPT-4o로 캐릭터 바이블과 페이지 장면 정보 추출
- 생성: `FLUX.2-klein-4B` 기반 이미지 생성
- 저장: 캐릭터 reference 이미지와 페이지 이미지를 S3에 업로드
- 결과 전달: 생성 완료 이미지를 Kafka 결과 토픽으로 publish

## 처리 흐름

### 1. 초기화 메시지 처리

`KAFKA_TOPIC_INIT` 기본값은 `fairytale_created` 입니다.

이 토픽의 메시지를 받으면 워커는 다음 작업을 수행합니다.

- `fairytaleId`, `setting`, `character_type`, `characters`를 읽음
- GPT-4o로 역할별 `visual_description`을 생성
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
  "character_type": "HUMAN",
  "characters": {
    "HERO": "나무꾼",
    "HELPER": "선녀",
    "VILLAIN": "호랑이"
  }
}
```

### PAGE 메시지 예시

`sentences`는 줄바꿈 문자열 또는 문자열 배열 모두 허용합니다.

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
      config.py                       # 테마, 출력, 프롬프트 설정
      generator_klein.py              # FLUX 이미지 생성기
      llm_prompt_extractor.py         # GPT-4o 기반 캐릭터/장면 분석
      model_manager.py                # 싱글턴 모델 로더
      story_pipeline.py               # 프롬프트 조합 유틸리티
```

## 모델 및 생성 설정

- 이미지 생성 모델: `black-forest-labs/FLUX.2-klein-4B`
- transformer repo: `Photoroom/FLUX.2-klein-4b-fp8-diffusers`
- 텍스트 분석 모델: `gpt-4o`
- 출력 해상도: `1024x1024`
- 추론 스텝: `4`
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

- INIT와 PAGE를 서로 다른 Kafka 토픽으로 분리
- PAGE가 INIT보다 먼저 도착한 경우 `BibleNotReadyError` 기반 재시도 처리
- 동일 페이지가 이미 있으면 재생성 없이 결과만 재전송
- 역할별 reference 이미지와 `bible.json`을 캐시해 중복 비용 감소
- Kafka consumer는 수동 commit 방식으로 동작
- 실패 시 backoff 및 최대 재시도 횟수 제한 적용

## 참고

- 구조 및 설계 메모: `ARCHITECTURE_KLEIN.md`
- 현재 README는 워커 중심 동작만 정리했습니다.
