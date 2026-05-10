"""End-to-end pipeline smoke test — Spring 응답 예시를 사용해 카프카·Redis 없이
실제 GPT-4o + FLUX + S3 흐름을 검증.

흐름:
  1. Spring 응답 데이터 하드코딩
  2. content를 페이지 단위(3문장씩, 마지막은 짧을 수 있음)로 분할
  3. Option B 형식 메시지 빌드 (페이지1에 컨텍스트 포함)
  4. KleinModelManager 로드 → FairytaleProcessor 인스턴스 생성
  5. NoopRedisPublisher 사용 (Redis 미세팅 환경 대응)
  6. 메시지마다 processor.handle_message 호출
  7. S3에 bible.json + 5 references + 7 pages 업로드되는지 확인

소요 시간: ~3-5분 (모델 로드 30~60s + ref 5장 ~60s + page 7장 ~120s)

실행:
    python _smoke_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env")


# ── Spring 응답 예시 데이터 ───────────────────────────────────────────────────

FAIRYTALE_DATA = {
    # 같은 줄거리지만 fairytale_id를 달리해 새 init 흐름 + 3-role 케이스 검증
    "fairytale_id": 21,
    "title": "KOREAN_TRADITIONAL의 COURAGE",
    "setting": "KOREAN_TRADITIONAL",
    "moral": "COURAGE",
    "character_type": "ANIMAL",
    "characters": {
        "HERO": "개구리",
        "VILLAIN": "늑대",
        "DISPATCHER": "사자",
    },
    "total_sentences": 19,
    "total_pages": 7,
    "content": (
        "옛날 옛날 한 옛날에, 푸른 연못에서 개구리가 살았어요. 개구리는 친구들과 함께 뛰어놀기를 좋아했으며, 매일 아침 \"오늘도 신나게 놀자!\"라고 외치며 하루를 시작했답니다.\n"
        "개구리는 수영을 좋아해서 \"물속에서 뛰어노는 게 최고야!\"라고 자주 말했어요.\n"
        "개구리는 친구들과 함께 연못가의 아름다운 꽃들을 보며 \"이런 멋진 곳에서 사는 게 정말 행복해!\"라고 항상 말했어요.\n"
        "하루는 개구리가 친구들과 신나게 노는 모습을 보던 자각이 있는 사자가 연못가로 찾아와 \"개구리야, 너희는 절대 저쪽 숲으로 가지 마라, 그곳에는 위험한 늑대가 살고 있단다!\"라고 경고했대요.\n"
        "개구리는 사자의 경고를 듣고 \"네, 알겠어요! 절대 숲으로 가지 않겠어요!\"라고 대답했어요.\n"
        "그런데 개구리는 친구들과 함께 새로운 놀이터를 찾아보기로 결심하고, \"모험을 떠나 보자!\"라고 외치며 숲으로 향했답니다.\n"
        "개구리는 친구들과 함께 숲 속의 신비로운 세계를 탐험하기로 결심하며 \"모험은 언제나 신나는 법이야! 혹시 위험이 있을지도 모르지만, 용기를 내서 나가보자!\"라고 외치며 힘차게 나아갔어요, 그러면서도 마음 속에서는 \"저 위험한 늑대를 만날지도 모르지만, 친구들과 함께라면 정말 멋진 모험이 될 거야!\"라고 생각했답니다, 그렇게 개구리는 새로운 경험을 찾아 떠나는 용기를 가지기로 결심했어요.\n"
        "숲 속 깊은 곳에서 개구리는 갑자기 무시무시한 늑대를 만났답니다.\n"
        "무서운 늑대가 나타나자 개구리는 가슴이 쿵쾅쿵쾅 뛰었지만, \"난 결코 물러서지 않을 거야!\"라고 외치며 용기를 내어 늑대에게 다가갔어요.\n"
        "늑대가 위협적으로 으르렁거리자, 개구리는 두려움을 느꼈지만 \"나는 친구들을 지켜야 해!\"라고 마음속으로 다짐하며, 용감하게 앞으로 나아가 늑대에게 \"넌 내가 두려운가? 난 절대 너에게 지지 않을 거야!\"라고 외쳐봤어요.\n"
        "늑대가 개구리를 향해 성큼성큼 다가오며 위협적으로 말했어요. \"너 같은 작은 개구리, 나를 이길 수 있을 거라 생각하나?\" 개구리는 두려움 속에서도 \"나는 포기하지 않을 거야!\"라고 외쳤답니다.\n"
        "개구리는 깊은 숨을 들이쉬고, \"이제 내 용기를 보여줄 시간이야!\"라고 외치며 힘껏 점프하여 늑대를 피했어요.\n"
        "개구리는 용기를 내어 \"이제 내가 진정한 힘을 보여줄게!\"라고 외치며 돌진했어요.\n"
        "개구리는 힘껏 점프하며 \"너에게 이길 수 있다는 걸 보여줄게!\"라고 외치며 늑대를 향해 돌진했답니다.\n"
        "그때 자라가 옆에서 \"개구리야, 힘내! 함께라면 반드시 이길 수 있어!\"라고 응원했어요.\n"
        "개구리는 자라의 응원을 받으며 더욱 힘을 내어 \"우리의 힘을 합쳐서 이겨내자!\"라고 외치며 용감하게 늑대를 물리쳤고, 결국 늑대는 개구리의 용기와 친구들의 힘에 놀랐어요.\n"
        "개구리는 자라와 함께 늑대를 물리친 후, 친구들과 함께 숲에서 신나는 모험을 계속하며 \"우리는 함께라면 어떤 위험도 이겨낼 수 있어!\"라고 외쳤답니다.\n"
        "그 후로 개구리는 용기 있는 마음으로 친구들과 함께 모험을 떠나며, 언제나 서로를 지키고 도와주겠다고 약속했답니다.\n"
        "그 후로 개구리는 용기 있는 마음을 잃지 않고 친구들과 함께 신나는 모험을 계속하며, 행복하게 살았답니다."
    ),
}


# ── Redis 미사용용 mock publisher ────────────────────────────────────────────

class MockPublisher:
    """Redis 접속 정보 없을 때 사용할 가짜 publisher — 콘솔에만 찍음."""

    @staticmethod
    def page_image_key(fairytale_id: int, page_no: int) -> str:
        return f"fairytale:{fairytale_id}:page:{page_no}:image_url"

    def publish_page_complete(self, fairytale_id: int, page_no: int, image_url: str, **_):
        key = self.page_image_key(fairytale_id, page_no)
        print(f"  [mock-redis] SET {key} = {image_url}")

    def publish_event(self, channel: str, payload: dict):
        print(f"  [mock-redis] PUB {channel} = {payload}")

    def ping(self) -> bool:
        return True


# ── 페이지 분할 ──────────────────────────────────────────────────────────────

def split_into_pages(content: str, total_pages: int) -> list[list[str]]:
    """content (\\n으로 구분된 문장들)를 total_pages 개로 균등 분할."""
    sentences = [s.strip() for s in content.split("\n") if s.strip()]
    n = len(sentences)
    base = n // total_pages
    extra = n % total_pages
    pages: list[list[str]] = []
    idx = 0
    for p in range(total_pages):
        size = base + (1 if p < extra else 0)
        pages.append(sentences[idx:idx + size])
        idx += size
    return pages


# ── 메시지 빌드 (Option B — 페이지1에 컨텍스트 포함) ────────────────────────

def build_messages(data: dict, pages: list[list[str]]) -> list[dict]:
    msgs: list[dict] = []
    for i, page_sentences in enumerate(pages, start=1):
        msg = {
            "fairytaleId": data["fairytale_id"],
            "pageNo": i,
            "sentences": page_sentences,
        }
        if i == 1:
            msg["setting"] = data["setting"]
            msg["character_type"] = data["character_type"]
            msg["characters"] = data["characters"]
        msgs.append(msg)
    return msgs


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. 페이지 분할 + 메시지 빌드
    pages = split_into_pages(FAIRYTALE_DATA["content"], FAIRYTALE_DATA["total_pages"])
    print(f"=== 페이지 분할 결과 ({len(pages)}개) ===")
    for i, p in enumerate(pages, 1):
        print(f"  P{i}: {len(p)}문장 — {p[0][:30]}...")
    print()

    messages = build_messages(FAIRYTALE_DATA, pages)
    print(f"=== 메시지 {len(messages)}개 빌드 완료 (페이지1에만 컨텍스트 포함) ===\n")

    # 2. 모델 로드
    from api_klein.pipeline.model_manager import KleinModelManager
    print("=== FLUX.2-klein-4B 모델 로드 ===")
    model_load_t0 = time.time()
    mgr = KleinModelManager.get()
    mgr.load()
    model_load_elapsed = time.time() - model_load_t0
    print(f"모델 로드 완료 ({model_load_elapsed:.1f}s)\n")

    # 3. processor 준비
    from api_klein.processor import FairytaleProcessor
    from api_klein.storage import S3Storage
    storage = S3Storage()
    publisher = MockPublisher()
    processor = FairytaleProcessor(storage=storage, publisher=publisher)
    print(f"=== processor 준비 완료 (bucket: {storage.bucket}) ===\n")

    # 4. 메시지 처리
    msg_times: list[float] = []
    pipeline_t0 = time.time()
    for i, msg in enumerate(messages, 1):
        print(f"--- [메시지 {i}/{len(messages)}] fairytaleId={msg['fairytaleId']} pageNo={msg['pageNo']} ---")
        msg_start = time.time()
        try:
            url = processor.handle_message(msg)
            elapsed = time.time() - msg_start
            msg_times.append(elapsed)
            print(f"--- 메시지 {i} 완료 ({elapsed:.1f}s) → {url}\n")
        except Exception as e:
            elapsed = time.time() - msg_start
            msg_times.append(elapsed)
            print(f"--- 메시지 {i} 실패 ({elapsed:.1f}s): {type(e).__name__}: {e}\n")
            raise
    pipeline_elapsed = time.time() - pipeline_t0
    print(f"=== 전체 메시지 처리 완료 — {pipeline_elapsed:.1f}s ===\n")

    # 5. S3 업로드 결과 검증
    fid = FAIRYTALE_DATA["fairytale_id"]
    roles = list(FAIRYTALE_DATA["characters"].keys())

    print(f"=== S3 업로드 결과 확인 ===")
    print(f"  bible.json: {'OK' if storage.object_exists(storage.bible_key(fid)) else 'MISSING'}")
    for role in roles:
        ok = storage.object_exists(storage.reference_key(fid, role))
        print(f"  references/{role}.png: {'OK' if ok else 'MISSING'}")
    for i in range(1, len(pages) + 1):
        ok = storage.object_exists(storage.page_key(fid, i))
        print(f"  pages/page_{i:02d}.png: {'OK' if ok else 'MISSING'}")

    print(f"\nS3 콘솔에서 확인: https://s3.console.aws.amazon.com/s3/buckets/{storage.bucket}?region={storage.region}&prefix=fairytales/{fid}/")

    # 6. 시간 요약 (메시지1은 init 포함이라 페이지 평균에서 제외)
    print(f"\n=== 시간 요약 ===")
    msg1_time = msg_times[0]
    page_times_only = msg_times[1:]  # init 제외
    avg_page = sum(page_times_only) / len(page_times_only) if page_times_only else 0.0
    init_estimate = msg1_time - avg_page  # init 단계 추정 = 메시지1 - 페이지 평균

    print(f"  모델 로드:                    {model_load_elapsed:>6.1f}s")
    print(f"  메시지1 (init + 페이지1):     {msg1_time:>6.1f}s")
    print(f"    └ init 추정 (bible+refs):   {init_estimate:>6.1f}s")
    print(f"    └ 페이지1 추정:             {avg_page:>6.1f}s")
    print(f"  페이지 2~{len(messages)} 평균:              {avg_page:>6.1f}s")
    print(f"  페이지당 최단/최장:            {min(page_times_only):>6.1f}s / {max(page_times_only):.1f}s")
    print(f"  파이프라인 총 (메시지 N개):    {pipeline_elapsed:>6.1f}s")
    print(f"  스크립트 총 (모델로드 포함):    {model_load_elapsed + pipeline_elapsed:>6.1f}s")
    print(f"\n  레퍼런스 1장 / 페이지 1장 개별 시간은 위쪽 [processor] 로그 참고")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨")
        sys.exit(1)
