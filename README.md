# 동화 삽화 생성 프로젝트

이 프로젝트는 동화 텍스트를 바탕으로 삽화를 생성하기 위한 실험용 코드베이스입니다.

현재는 크게 두 영역으로 나뉩니다.

- `runtime_pipeline/`
  동화 한 줄씩 삽화를 만들기 위한 생성 파이프라인
- `model_training/`
  LoRA 학습, 데이터셋 준비, 검증용 코드

기존의 `generation/`, `training/` 폴더는 예전 경로 호환을 위한 얇은 래퍼입니다. 앞으로는 `runtime_pipeline/`, `model_training/` 기준으로 보는 것이 맞습니다.

## 현재 구조

```text
fairytale_lora/
├─ runtime_pipeline/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ story_pipeline.py
│  ├─ generator.py
│  ├─ fairytale_generator.py
│  └─ compare_models.py
├─ model_training/
│  ├─ __init__.py
│  ├─ train_config.py
│  ├─ prepare_dataset.py
│  ├─ add_captions.py
│  ├─ train_lora.py
│  └─ verify_lora.py
├─ generation/   # 호환 래퍼
├─ training/     # 호환 래퍼
├─ dataset/
├─ lora_output/
└─ outputs/
```

## 지금까지 수정한 내용

### 1. 생성 코드와 학습 코드를 분리함

이전에는 생성과 학습 관련 코드가 같은 수준에 섞여 있었습니다.

지금은:

- `runtime_pipeline/` 에 생성 관련 코드만 모음
- `model_training/` 에 학습 관련 코드만 모음

이렇게 분리해 두어서, 나중에 파이프라인만 따로 올리거나 배포하기 쉬운 구조로 바뀌었습니다.

### 2. 스토리 기반 생성용 파이프라인 뼈대를 추가함

단순히 프롬프트 한 줄을 바로 이미지로 넘기는 구조에서, 스토리 입력을 받아 장면별 프롬프트를 만드는 단계가 추가되었습니다.

핵심 파일:

- [runtime_pipeline/story_pipeline.py](/c:/Users/leega/Desktop/Projects/CapStone/picture/fairytale_lora/runtime_pipeline/story_pipeline.py)
- [runtime_pipeline/config.py](/c:/Users/leega/Desktop/Projects/CapStone/picture/fairytale_lora/runtime_pipeline/config.py)

현재 이 파이프라인은 다음 입력을 기준으로 동작합니다.

- `theme`
- `characters`
- `lines`

예시:

```json
{
  "theme": "korean_traditional",
  "characters": [
    {
      "id": "hero",
      "type": "animal",
      "species": "frog",
      "role": "main",
      "traits": ["kind", "curious"]
    }
  ],
  "lines": [
    "개구리는 숲길을 걸어갔다.",
    "개구리는 마을 어귀에 도착했다."
  ]
}
```

스타일은 입력 JSON에 매번 넣지 않고 시스템 기본값으로 사용합니다.
현재 기본 스타일은 `fairytale_pastel` 입니다.

### 3. 범용 규칙 기반 프롬프트 조립 구조를 넣음

특정 예시 하나에 맞춘 구조가 아니라, 아래 정보를 조합해서 프롬프트를 만드는 방식으로 바뀌었습니다.

- 스타일 규칙
- 세계관 확장 규칙
- 캐릭터 타입 규칙
- 역할 규칙
- 직업 규칙
- 동물 종 규칙

즉 `왕이 꼭 나오는 구조`가 아니라, 들어오는 캐릭터 정보에 따라 힌트를 조립하는 구조입니다.

### 4. 기존 실행 경로를 완전히 깨지 않도록 래퍼를 남김

예전처럼 `generation/...`, `training/...` 경로를 참조해도 되도록 호환 래퍼 파일을 남겨두었습니다.

## 현재 구현 상태

### 구현된 것

- 생성 코드와 학습 코드의 물리적 분리
- 스토리 입력 스키마 기반 파이프라인 뼈대
- `theme + characters + lines -> prompt` 조립 로직
- 세계관 확장 규칙 테이블
- 캐릭터 타입/역할/직업/종 기반 힌트 조립
- 장면별 prompt / negative prompt 생성
- 구조 확인용 `--plan-only` 흐름
- 기존 경로 호환 래퍼

### 아직 미구현이거나 초안 단계인 것

- 동화 생성 결과를 이 스키마로 자동 변환하는 부분
- 줄별 장면 해석 고도화
- 등장인물 일관성을 위한 실제 reference image 전략
- 장소가 아니라 세계관 안의 세부 배경 선택 로직
- 비동기 생성
- preload
- 캐시
- 사용자 페이지 넘김 UX와 연결되는 큐 관리
- 최종 서비스용 API 또는 앱 연결

## 지금 파이프라인은 어디까지 됐는가

현재 파이프라인은 다음 단계까지 구현돼 있습니다.

1. 구조화된 스토리 입력을 받음
2. 줄 단위로 정리함
3. 세계관 힌트를 확장함
4. 캐릭터 바이블을 조립함
5. 각 줄마다 최종 이미지 생성용 프롬프트를 만듦

즉, 현재는 `스토리 기반 이미지 생성의 전처리/프롬프트 구성 단계`까지는 들어와 있습니다.

반면, 아직 `서비스에서 부드럽게 돌기 위한 운영 단계`는 거의 남아 있습니다.

## 추천 실행 방식

### 1. 파이프라인 계획만 확인

```bash
python runtime_pipeline/fairytale_generator.py --plan-only --story-json-file sample_story.json
```

또는 예전 경로 호환:

```bash
python generation/fairytale_generator.py --plan-only --story-json-file sample_story.json
```

### 2. 학습 관련 실행

```bash
python model_training/prepare_dataset.py
python model_training/train_lora.py
python model_training/verify_lora.py
```

또는 예전 경로 호환:

```bash
python training/prepare_dataset.py
python training/train_lora.py
python training/verify_lora.py
```

## 모델 설정에 대한 현재 판단

[runtime_pipeline/config.py](/c:/Users/leega/Desktop/Projects/CapStone/picture/fairytale_lora/runtime_pipeline/config.py) 에는 현재 `sdxl`, `sd15`, `sd21` 세 가지 base model 키가 남아 있습니다.

하지만 실제 프로젝트 기준으로는 거의 `sdxl` 중심으로 보면 됩니다.

특히:

- 현재 로컬 LoRA인 `local_lora`는 SDXL용
- 동화 삽화 품질도 SDXL 쪽이 더 적합

그래서 실사용 기준 추천 조합은 아래입니다.

- `base_model_key="sdxl"`
- `lora_key="local_lora"`

`sd15`, `sd21`은 비교용 또는 실험 흔적에 가깝습니다.

## 이제 다음에 해야 할 일

우선순위 기준으로 정리하면 아래 순서가 좋습니다.

### 1. 동화 생성 출력 스키마 확정

가장 먼저 해야 할 일입니다.

동화 생성 쪽에서 이미지 생성 쪽으로 넘겨줄 형식을 확정해야 합니다.

추천 최소 스키마:

```json
{
  "theme": "korean_traditional",
  "characters": [
    {
      "id": "hero",
      "type": "animal",
      "species": "frog",
      "role": "main",
      "traits": ["kind", "curious"]
    }
  ],
  "lines": [
    "개구리는 궁궐로 향했다.",
    "개구리는 왕을 만났다."
  ]
}
```

### 2. 세계관 규칙 더 정리

지금은 몇 가지 예시 세계관만 들어 있습니다.

앞으로는 다음처럼 늘려야 합니다.

- 한국 전통
- 서양 중세
- 숲속 판타지
- 바다 세계
- 현대 도시 판타지

### 3. 캐릭터 바이블 강화

지금은 텍스트 힌트 수준입니다.

다음 단계에서는:

- 반복 등장 캐릭터 외형 고정
- 캐릭터별 대표 시각 특징 정리
- 가능하면 reference image 방식 검토

까지 가야 합니다.

### 4. 줄별 장면 표현 보강

현재는 `scene action: 각 줄 텍스트` 수준입니다.

나중에는 줄에서 다음 정보를 더 뽑도록 발전시켜야 합니다.

- 누가 등장하는지
- 어떤 표정/감정인지
- 배경이 세계관 안에서 어디인지
- 어떤 소품이 필요한지

### 5. UX용 비동기 처리 추가

서비스 관점에서 매우 중요하지만 아직 안 했습니다.

필요한 것:

- 현재 페이지 생성
- 다음 페이지 preload
- 생성 결과 캐시
- 실패 시 fallback

## 한 줄 요약

현재는 `동화 한 줄씩 삽화를 만들기 위한 파이프라인 구조 정리와 프롬프트 조립 뼈대`까지 구현된 상태입니다.

다음 핵심 과제는 `동화 생성 출력 스키마 확정`, `캐릭터/세계관 규칙 강화`, `비동기 preload/캐시 추가`입니다.
