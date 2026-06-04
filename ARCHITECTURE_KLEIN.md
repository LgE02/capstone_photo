# ARCHITECTURE_KLEIN

`FLUX.2-klein-4B` 기반 동화 삽화 생성 워커의 현재 아키텍처를 정리한 문서입니다.  
현재 구현의 중심은 FastAPI 서버가 아니라 `worker.py` 기반 Kafka 소비형 파이프라인입니다.

## 1. 목표

이 시스템은 동화의 전체 캐릭터 설정을 먼저 고정한 뒤, 페이지별 장면을 일관된 스타일로 생성하는 것을 목표로 합니다.

핵심 요구사항은 아래와 같습니다.

- 캐릭터 외형 일관성 유지
- 페이지별 장면 분리 생성
- 문화권과 시대에 맞는 캐릭터/배경 묘사
- 생성 결과를 외부 시스템이 비동기로 받을 수 있는 구조
- GPU 1장 환경에서도 운영 가능한 단순한 워커 구조

## 2. 현재 아키텍처 요약

현재 파이프라인은 크게 5개 영역으로 나뉩니다.

1. Kafka consumer가 INIT/PAGE 메시지를 수신
2. processor가 메시지 종류에 따라 초기화 또는 페이지 생성 수행
3. GPT-5.5가 캐릭터 바이블과 페이지 장면 정보를 추출
4. FLUX.2-klein-4B가 reference 이미지와 페이지 이미지를 생성
5. S3 저장 후 Kafka result topic으로 완료 이벤트를 publish

## 3. 상위 구조

```text
Spring / upstream service
  -> Kafka topic: fairytale_created
  -> Kafka topic: fairytale_paragraph

worker.py
  -> api_klein.consumer.run_consumer_loop()
  -> api_klein.processor.FairytaleProcessor.handle_message()
     -> GPT-5.5 character/page analysis
     -> FLUX.2-klein-4B image generation
     -> S3 upload
     -> Kafka result publish

Result consumer
  -> Kafka topic: fairytale_image
```

## 4. 메시지 기반 처리 구조

### INIT 토픽

기본 토픽명은 `fairytale_created` 입니다.

역할:

- 동화 단위 초기화
- 캐릭터 바이블 생성
- 역할별 reference 이미지 생성
- 이후 PAGE 생성에 필요한 공통 자원 준비

예시 payload:

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

### PAGE 토픽

기본 토픽명은 `fairytale_paragraph` 입니다.

역할:

- 페이지 단위 삽화 생성
- 장면 분석
- 등장 캐릭터 reference 주입
- 결과 이미지 저장과 완료 이벤트 publish

예시 payload:

```json
{
  "fairytaleId": 17,
  "pageNo": 1,
  "sentences": "나무꾼이 산길을 걸어갔어요.\n멀리서 신비한 빛이 보였어요."
}
```

### RESULT 토픽

기본 토픽명은 `fairytale_image` 입니다.

예시 payload:

```json
{
  "fairytaleId": 17,
  "pageNo": 1,
  "imageurl": "https://.../fairytales/17/pages/page_01.png"
}
```

## 5. 처리 흐름

### 5.1 INIT 처리

INIT 메시지를 받으면 `FairytaleProcessor._ensure_initialized()`가 실행됩니다.

흐름:

1. `fairytaleId`, `setting`, `char_species`, `characters` 파싱
2. S3에 기존 `bible.json`과 reference 이미지가 있는지 확인
3. 이미 있으면 재생성하지 않고 캐시에 적재
4. 없으면 GPT-5.5로 역할별 `visual_description` 생성
5. `bible.json`을 S3에 업로드
6. 각 역할에 대해 reference 이미지를 생성하고 S3에 업로드
7. 메모리 캐시에 `bible`과 reference 이미지를 저장

INIT는 결과 토픽으로 별도 publish 하지 않습니다.  
이 단계는 PAGE 생성 준비 단계입니다.

### 5.2 PAGE 처리

PAGE 메시지를 받으면 `FairytaleProcessor._process_page()`가 실행됩니다.

흐름:

1. `fairytaleId`, `pageNo`, `sentences` 파싱
2. 이미 해당 페이지가 S3에 있으면 생성 생략
3. 기존 결과 URL을 result topic에 다시 publish
4. `bible.json` 로드
5. GPT-5.5로 현재 페이지의 `focus_roles`와 `narrative_hint` 추출
6. 현재 페이지에 실제로 등장하는 역할의 reference 이미지만 로드
7. 최종 프롬프트 조합
8. FLUX.2-klein-4B로 1024x1024 이미지 생성
9. S3에 업로드
10. 결과 Kafka topic으로 완료 메시지 publish

## 6. GPT-5.5의 역할

GPT-5.5는 이미지 생성 자체가 아니라, 이미지 생성에 필요한 구조화된 해석을 담당합니다.
모델 ID는 `.env`의 `OPENAI_MODEL`로 교체 가능 (미설정 시 `gpt-5.5`).

### 6.1 캐릭터 바이블 생성

`extract_character_bible()`의 역할:

- 역할별 캐릭터 타입 추론
- 문화권/시대에 맞는 복식 묘사 생성
- 감정 표현이 아닌 중립적 외형 설명 생성
- reference 이미지 생성용 `visual_description` 작성
- 종별 anatomy 가드(BIRD/REPTILE/AMPHIBIAN/MAMMAL) 적용 — 새에 "귀 뒤" 같은 포유류 표현이 들어가지 않도록 단어 선택을 제약
- VILLAIN 톤 가이드 — 호러로 빠지지 않는 범위 내에서 어둡고 위압감 있는 묘사

출력 예시는 아래와 같은 형태입니다.

```json
{
  "HERO": {
    "name": "나무꾼",
    "type": "HUMAN",
    "visual_description": "..."
  },
  "HELPER": {
    "name": "선녀",
    "type": "ETC",
    "visual_description": "..."
  }
}
```

### 6.2 페이지 장면 분석

`extract_page_scene()`의 역할:

- 현재 페이지(1~3문장 가변)에서 실제로 보일 역할만 선택
- 가장 시각적으로 강한 한 순간을 골라 `narrative_hint`(영어 단일 문장)로 압축
- 동작, 감정, 환경, 감정 심볼('!', '?', ❤ 등)을 한 문장에 포함
- 이전 페이지 분석 결과(`previous_scenes`)를 받아 공간/상황 연속성 유지
- 인접 페이지 간 같은 시각 비트가 반복되지 않도록 통제
- 프롬프트에 불필요한 군중/추가 캐릭터가 생기지 않도록 통제

출력 예시:

```json
{
  "narrative_hint": "Wide shot of the woodcutter walking along a mountain path, looking startled as a mysterious glow appears ahead in the forest.",
  "focus_roles": ["HERO"]
}
```

## 7. FLUX.2-klein-4B 사용 방식

### 7.1 선택 이유

이전 실험에서는 SDXL, SDXL LoRA, FLUX.1-schnell 기반 구성이 있었지만, 현재는 `FLUX.2-klein-4B`가 가장 현실적인 균형을 제공한다고 판단했습니다.

주요 이유:

- Qwen3 텍스트 인코더 기반의 긴 자연어 프롬프트 처리
- 4 step 설정에서도 빠른 생성 속도
- reference image 입력을 통한 캐릭터 일관성 보강
- RTX 3060 12GB 환경에서 CPU offload와 함께 운용 가능

### 7.2 생성 설정

- base model: `black-forest-labs/FLUX.2-klein-4B`
- transformer: `Photoroom/FLUX.2-klein-4b-fp8-diffusers`
- output size: `768x768` (`OUTPUT_RESOLUTION` 상수로 통제, 1024 대비 약 40% 단축)
- inference steps: `4` (`NUM_INFERENCE_STEPS` 상수, Klein 모델이 4-step 학습 최적화라 3 이하는 이미지 깨짐 확인됨)
- guidance scale: `1.0`
- dtype: BF16
- optimization: CPU offload
- optional optimization: `torchao` FP8 weight-only quantization

### 7.3 reference 이미지 전략

캐릭터 일관성 유지를 위해 동화 단위로 역할별 reference 이미지를 먼저 생성합니다.

전략:

1. INIT 단계에서 역할별 reference 이미지 생성
2. PAGE 단계에서 현재 장면에 필요한 역할만 골라 multi-image로 입력
3. 외형은 유지하고, 장면 설명은 페이지 프롬프트로 제어

장점:

- 페이지 간 외형 일관성 향상
- 역할이 여러 개인 장면에서도 필요한 reference만 선택 가능

한계:

- reference가 강하게 작동하면 포즈와 표정 자유도가 줄어듦
- reference를 쓰지 않으면 감정 표현은 자유롭지만 외형이 흔들릴 수 있음

## 8. 저장 구조

S3에는 동화 단위로 아래 구조를 사용합니다.

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

의도:

- `bible.json`은 동화 전체의 공통 메타데이터
- `references/`는 역할별 재사용 자산
- `pages/`는 최종 결과물

## 9. 캐시 전략

`FairytaleProcessor`는 메모리 내 캐시를 사용합니다.

- `_bible_cache`: `fairytale_id -> bible payload`
- `_ref_cache`: `fairytale_id -> { role -> PIL.Image }`
- `_scene_history`: `fairytale_id -> [ { page_no, sentences, narrative_hint, focus_roles } ]`
  — 페이지 N 처리 시 1..N-1의 분석 결과를 GPT-5.5에 함께 전달해 공간/상황 연속성과 장면 중복 회피에 사용

의도:

- 같은 동화의 여러 페이지 처리 시 S3 재다운로드 최소화
- 한 워커 프로세스 내 반복 처리 비용 감소
- 페이지 간 시각적 일관성 향상

현재는 간단한 dict 기반 캐시이며 LRU나 TTL은 적용하지 않았습니다. 워커 재시작 시 캐시는 모두 소실되고 S3에서 다시 로드됩니다.

## 10. 장애 처리와 멱등성

### 10.1 수동 commit

consumer는 `enable_auto_commit=False`로 동작합니다.  
메시지 처리가 성공했을 때만 commit 합니다.

### 10.2 PAGE가 INIT보다 먼저 오는 경우

이 경우 `BibleNotReadyError`를 발생시켜 commit을 보류하고 재시도합니다.

동작:

- 최대 재시도 횟수: `MAX_RETRIES = 5`
- 대기 시간: `BIBLE_WAIT_SECONDS = 5`
- 재시도 시 같은 offset으로 `seek`

### 10.3 일반 예외 재시도

모델 생성, S3, Kafka publish 등에서 일반 예외가 발생하면 exponential backoff로 재시도합니다.

동작:

- 재시도 횟수 추적: `(partition, offset)` 기준
- backoff: `2 ** retry_count`
- 최대 횟수 초과 시 DLQ 토픽(`fairytale_dlq` 기본값)으로 격리 후 commit

### 10.4 DLQ(Dead Letter Queue) 격리

다음 케이스가 DLQ로 격리됩니다.

- JSON 파싱 실패(poison message)
- 일반 예외가 `MAX_RETRIES`(기본 5회) 초과
- `BibleNotReadyError`가 `MAX_RETRIES` 초과 — INIT가 영원히 안 올 가능성

DLQ payload는 원본 토픽/파티션/오프셋/페이로드 + `failureReason`/`lastError`/`retryCount`/`failedAt`을 포함합니다. DLQ 전송이 실패해도 메인 consumer 루프는 죽지 않도록 예외를 흡수합니다.

### 10.5 멱등 처리

PAGE 처리 전에 `storage.page_exists(fairytale_id, page_no)`를 확인합니다.

이미 생성된 페이지가 있으면:

- 이미지 재생성 생략
- 기존 URL을 결과 토픽으로 다시 publish

INIT도 idempotent — S3에 `bible.json`과 모든 reference가 이미 있으면 GPT-5.5/FLUX 호출 없이 캐시만 채우고 종료합니다. (단, 시스템 프롬프트 가드가 바뀌어도 기존 bible은 갱신되지 않습니다 — 새 동화 ID로 처리하거나 해당 S3 자료를 지워야 새 가드 효과가 적용됩니다.)

이 방식으로 중복 메시지나 재시도 상황에서 생성 비용을 줄입니다.

## 11. 주요 모듈 역할

```text
worker.py
  워커 시작점. 환경 변수 확인, 모델 로드, publisher 시작, consumer 루프 실행.

api_klein/consumer.py
  Kafka consumer 구성, topic subscribe, 수동 commit, 재시도 제어.

api_klein/processor.py
  INIT/PAGE 메시지 분기, 캐릭터 바이블 생성, 페이지 생성, 캐시 관리.

api_klein/publisher.py
  Kafka 결과 토픽 publish 전담. 내부적으로 background event loop에서 producer 운용.

api_klein/storage.py
  S3 업로드/다운로드, object 존재 확인, public URL / s3 URL 생성.

api_klein/pipeline/llm_prompt_extractor.py
  GPT-5.5 기반 캐릭터 바이블 생성과 페이지 장면 추출.
  종별 anatomy 가드, VILLAIN 톤 가이드, 장면 연속성/중복 회피 규칙 포함.

api_klein/pipeline/generator_klein.py
  FLUX.2-klein-4B 파이프라인 로드 및 reference/page 이미지 생성.
  reference 이미지는 흰 배경 풀바디, 페이지는 multi-image 가이던스로 전달.

api_klein/pipeline/model_manager.py
  단일 프로세스에서 모델을 공유하는 싱글턴 로더.

api_klein/pipeline/story_pipeline.py
  테마 키(KOREAN_TRADITIONAL 등)를 positive/negative 배경 힌트로 변환.

api_klein/pipeline/config.py
  테마별 배경/네거티브 힌트 정의(`THEME_EXPANSIONS`).
```

## 12. 현재 트레이드오프

### 장점

- 구조가 단순하고 운영 흐름이 명확함
- INIT와 PAGE를 분리해 캐릭터 설정 재사용 가능
- S3와 Kafka를 중심으로 외부 시스템과 느슨하게 연결됨
- 긴 자연어 프롬프트를 적극 활용할 수 있음

### 한계

- reference 이미지가 강할수록 장면 연출 자유도가 줄어듦
- 단일 워커 기준이라 처리량 확장 전략이 아직 단순함
- DLQ, 모니터링, 메트릭, job 추적 레이어가 아직 약함
- 문화권 표현 편향은 프롬프트만으로 완전히 해결되지 않음

## 13. 향후 개선 후보

- 워커 다중화와 partition 전략 정교화(`fairytaleId` 파티션 키 + 다중 워커)
- presigned URL 기반 비공개 S3 배포
- 캐시 eviction 정책 도입(LRU/TTL)
- DLQ 자동 재처리 도구
- OpenAI 쿼터/헬스 체크 + 알람 채널
- 페이지 생성 결과 품질 검수 단계 추가
- reference 강도를 제어할 수 있는 generation 옵션 실험

## 14. 실행

```bash
pip install -r requirements.txt
python worker.py
```

필수 환경 변수:

- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `KAFKA_BOOTSTRAP_SERVERS`

주요 선택 환경 변수:

- `AWS_REGION`
- `KAFKA_TOPIC_INIT`
- `KAFKA_TOPIC_PAGE`
- `KAFKA_TOPIC_RESULT`
- `KAFKA_GROUP_ID`
- `KAFKA_SECURITY_PROTOCOL`
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`

## 15. 모델 변경 히스토리

프로젝트는 초기부터 현재 구조까지 여러 모델과 방식을 거치며 개선되었습니다.

### 15.1 SDXL 기반 초기 시도

초기에는 SDXL 계열 모델을 사용해 동화 삽화를 생성했습니다.

한계:

- 동화책 특유의 일러스트 스타일이 약했음
- 장면마다 캐릭터 외형이 달라졌음
- 프롬프트 길이와 표현력이 제한적이었음
- 생성 속도 대비 품질 만족도가 높지 않았음

### 15.2 SDXL + LoRA 학습 시도

이후 동화풍 스타일 보정을 위해 SDXL 기반 LoRA 학습을 시도했습니다.

한계:

- 학습 데이터 규모가 작아 충분한 일반화가 어려웠음
- 12GB VRAM 환경에서 학습 효율이 낮았음
- 원하는 수준의 스타일 고정 효과가 크지 않았음

### 15.3 FLUX.1-schnell + QLoRA 시도

그다음 단계에서는 FLUX.1-schnell과 QLoRA 기반 실험을 진행했습니다.

개선점:

- SDXL보다 자연어 프롬프트 반영력이 좋아졌음
- 장면 묘사 자체는 더 유연해졌음

한계:

- 생성 속도가 느렸음
- 소규모 데이터에서 overfitting이 발생했음
- LoRA를 붙였을 때 오히려 결과가 불안정해지는 구간이 있었음
- 캐릭터 일관성 문제는 여전히 충분히 해결되지 않았음

### 15.4 GPT-4o 기반 프롬프트 파이프라인 도입

이후에는 모델 자체를 계속 미세조정하기보다, GPT-4o를 이용해 캐릭터와 장면 정보를 구조화하는 방향으로 전환했습니다.

개선점:

- 캐릭터 설명을 더 정교하게 만들 수 있었음
- 장면별 감정, 동작, 카메라 앵글을 프롬프트에 반영할 수 있었음
- 문화권과 시대를 고려한 묘사 보정이 가능해졌음

한계:

- 프롬프트만으로 캐릭터 외형 일관성을 완전히 고정하기는 어려웠음

### 15.5 FLUX.2-klein-4B + reference 이미지 방식

이 단계에서 `FLUX.2-klein-4B` + GPT-4o 분석 + 역할별 reference 이미지 결합 구조를 도입했습니다.

개선점:

- 생성 속도가 이전 FLUX.1-schnell 대비 개선됨
- 긴 자연어 프롬프트 활용이 쉬워짐
- reference 이미지로 캐릭터 일관성이 향상됨
- GPT-4o와 결합해 장면별 연출 제어력이 높아짐

남은 한계:

- reference를 강하게 쓰면 포즈와 표정 변화가 제한됨
- 문화권 표현 왜곡 가능성이 완전히 사라지지는 않음
- 다중 캐릭터 장면 혼합 문제는 일부 남아 있음

### 15.6 GPT-5.5 업그레이드 + 가드 시스템 강화 (현재)

현재 구조는 `FLUX.2-klein-4B` + OpenAI Chat Completion(`gpt-5.5` 기본) + 역할별 reference 이미지 + 다층 프롬프트 가드입니다.

변경점:

- 모델 ID를 `.env`의 `OPENAI_MODEL`로 추출 (기본 `gpt-5.5`, 운영 중 교체 가능)
- GPT-5.5 호환성: `max_completion_tokens=2048`, `temperature` 자유 지정 불가, `reasoning_effort="low"` 적용
- 종별 anatomy 가드 추가 (BIRD/REPTILE/FISH/AMPHIBIAN/MAMMAL)
- 페이지 분석 가드: SINGLE BEAT RULE, 단수 한정사 self-check, PASSIVE CHARACTER POSITIONING
- 페이지당 reference 이미지 최대 3장 컷
- 출력 해상도 768×768 (1024 대비 40% 단축)

개선점:

- 종 위반(까마귀 입꼬리, 양서류에 fur 등) 사전 차단
- 캐릭터 복제 패턴 감소 (단수 한정사 self-check + PASSIVE POSITIONING)
- 복합 액션 페이지 안정화 (SINGLE BEAT RULE)
- 동화 1편 생성 시간 약 30~40% 단축 (해상도 + reasoning_effort)

남은 한계:

- FLUX 자체의 다중 캐릭터 attention 분산 한계는 본질적
- 추상적 결말/감정 페이지에서 산발적 미니어처 복제 가능
- 레퍼런스 이미지가 정면 풀바디 1포즈만 있어 자세 변형 자유도 제한
- bible 단계 호출 시간 GPT-4o 대비 증가 (reasoning 토큰 비용)

## 16. 성능 및 품질 개선 요약

모델과 구조를 변경하면서 다음과 같은 개선이 있었습니다.

### 속도 측면

- SDXL 기반 대비 현재 구조가 더 안정적인 속도로 동작
- FLUX.1-schnell 대비 FLUX.2-klein-4B에서 생성 시간이 단축됨
- CPU offload와 FP8 시도로 제한된 VRAM 환경 대응력이 좋아짐

### 품질 측면

- 단순 프롬프트 입력 방식보다 GPT-4o 기반 장면 분석으로 스토리 반영력이 향상됨
- 역할별 reference 이미지 도입으로 캐릭터 일관성이 개선됨
- 문화권과 시대를 고려한 프롬프트 보정으로 배경 및 복식 품질이 향상됨

### 운영 구조 측면

- 단순 단일 생성 스크립트에서 Kafka 기반 비동기 워커 구조로 발전
- INIT와 PAGE를 분리해 공통 자산을 재사용할 수 있게 됨
- S3 저장과 Kafka 결과 전송을 통해 외부 시스템 연동이 쉬워짐
