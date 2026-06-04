"""GPT 모델 A/B 비교 도구 — bible 생성 + 페이지 분석을 두 모델로 동시 실행해
시간/출력 차이를 한 번에 보여준다.

워커 거치지 않고 GPT만 직접 호출 → 빠른 비교 (이미지 생성 안 함).
이미지 결과 차이는 워커로 두 fairytale_id 돌려서 S3에서 비교.

실행:
    # examples/sample_story.json 으로 GPT-5.5 vs GPT-4o 비교
    python benchmark/compare_models.py --story examples/sample_story.json

    # 다른 모델 쌍
    python benchmark/compare_models.py --story examples/sample_korean_human.json \\
        --model-a gpt-5.5 --model-b gpt-4o-mini

    # 페이지 분석은 1번만 (빠르게)
    python benchmark/compare_models.py --story examples/sample_story.json --pages 1

출력:
  - bible 생성 시간/출력 비교 (visual_description 차이)
  - 페이지별 분석 시간/출력 비교 (narrative_hint, focus_roles 차이)
  - 결과 JSON 파일도 저장 (benchmark/results/<timestamp>/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path 에 추가 (benchmark/ 에서 import 가능하게)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api_klein.pipeline.llm_prompt_extractor import (
    extract_character_bible,
    extract_page_scene,
)


def _load_story(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("setting", "char_species", "characters", "pages"):
        if key not in data:
            raise ValueError(f"story JSON 에 '{key}' 가 없음: {path}")
    return data


def _parse_sentences(raw) -> list[str]:
    if isinstance(raw, str):
        return [s.strip() for s in raw.split("\n") if s.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    return []


def _time_bible(model: str, setting: str, char_species: str, characters: dict[str, str]) -> tuple[dict, float]:
    print(f"  [{model}] bible 호출 중...")
    t0 = time.time()
    result = extract_character_bible(
        setting=setting,
        char_species=char_species,
        characters=characters,
        model=model,
    )
    elapsed = time.time() - t0
    print(f"  [{model}] bible 완료 ({elapsed:.2f}s)")
    return result, elapsed


def _time_page(
    model: str,
    sentences: list[str],
    character_bible: dict,
    setting: str,
    previous_scenes: list[dict],
) -> tuple[dict, float]:
    t0 = time.time()
    result = extract_page_scene(
        sentences=sentences,
        character_bible=character_bible,
        setting=setting,
        previous_scenes=previous_scenes,
        model=model,
    )
    elapsed = time.time() - t0
    return result, elapsed


def _print_bible_diff(bible_a: dict, bible_b: dict, model_a: str, model_b: str) -> None:
    """bible 출력의 의미 있는 차이를 표시."""
    roles = sorted(set(bible_a.keys()) | set(bible_b.keys()))
    print("\n── bible visual_description 비교 ──")
    for role in roles:
        desc_a = (bible_a.get(role, {}) or {}).get("visual_description", "")
        desc_b = (bible_b.get(role, {}) or {}).get("visual_description", "")

        print(f"\n  [{role}]")
        print(f"    {model_a} ({len(desc_a)} chars):")
        print(f"      {desc_a}")
        print(f"    {model_b} ({len(desc_b)} chars):")
        print(f"      {desc_b}")

        # 단어 단위로 차이 키워드 추출 (단순 비교)
        words_a = set(desc_a.lower().split())
        words_b = set(desc_b.lower().split())
        only_a = words_a - words_b
        only_b = words_b - words_a
        if only_a or only_b:
            print(f"    diff (단어 단위):")
            if only_a:
                print(f"      {model_a} 에만: {sorted(only_a)[:15]}")
            if only_b:
                print(f"      {model_b} 에만: {sorted(only_b)[:15]}")


def _print_page_diff(page_a: dict, page_b: dict, model_a: str, model_b: str, page_no: int) -> None:
    print(f"\n── page {page_no} 비교 ──")
    print(f"  focus_roles:")
    print(f"    {model_a}: {page_a.get('focus_roles')}")
    print(f"    {model_b}: {page_b.get('focus_roles')}")

    hint_a = page_a.get("narrative_hint", "")
    hint_b = page_b.get("narrative_hint", "")
    print(f"  narrative_hint:")
    print(f"    {model_a} ({len(hint_a)} chars):")
    print(f"      {hint_a}")
    print(f"    {model_b} ({len(hint_b)} chars):")
    print(f"      {hint_b}")

    # 가드 준수 빠른 체크
    print(f"  가드 준수 체크:")
    for label, hint in ((model_a, hint_a), (model_b, hint_b)):
        lone_count = hint.lower().count("the lone")
        has_while = "while" in hint.lower()
        verb_hints = sum(hint.lower().count(v) for v in ("walks", "stands", "runs", "sits", "looks", "smiles", "holds", "throws", "falls"))
        print(f"    {label}: 'the lone' x{lone_count}, while 포함={has_while}, 동작 동사≈{verb_hints}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True, help="examples/*.json 같은 동화 시나리오 JSON")
    ap.add_argument("--model-a", default="gpt-5.5", help="비교 모델 A (default: gpt-5.5)")
    ap.add_argument("--model-b", default="gpt-4o", help="비교 모델 B (default: gpt-4o)")
    ap.add_argument("--pages", type=int, default=0,
                    help="분석할 페이지 수 (0 = 전체 페이지, 1 = 첫 페이지만)")
    ap.add_argument("--save", action="store_true", help="결과를 benchmark/results/ 에 저장")
    args = ap.parse_args()

    story = _load_story(args.story)
    setting = story["setting"]
    char_species = story["char_species"]
    characters = dict(story["characters"])
    pages = story["pages"]
    if args.pages > 0:
        pages = pages[: args.pages]

    print(f"\n============================================================")
    print(f"  GPT 모델 비교: {args.model_a}  vs  {args.model_b}")
    print(f"  시나리오: {args.story}")
    print(f"  setting={setting} / char_species={char_species}")
    print(f"  characters={characters}")
    print(f"  pages={len(pages)}")
    print(f"============================================================\n")

    # ── BIBLE 생성 ──
    print("[1/2] bible 생성 시간 측정")
    bible_a, bible_time_a = _time_bible(args.model_a, setting, char_species, characters)
    bible_b, bible_time_b = _time_bible(args.model_b, setting, char_species, characters)

    print(f"\n  bible 시간 요약:")
    print(f"    {args.model_a}: {bible_time_a:.2f}s")
    print(f"    {args.model_b}: {bible_time_b:.2f}s")
    print(f"    차이: {bible_time_a - bible_time_b:+.2f}s "
          f"({(bible_time_a / bible_time_b):.2f}배)")

    _print_bible_diff(bible_a, bible_b, args.model_a, args.model_b)

    # ── 페이지 분석 ──
    print(f"\n\n[2/2] 페이지 분석 시간/출력 측정 (페이지 {len(pages)}개)")
    page_results: list[dict] = []
    page_times_a: list[float] = []
    page_times_b: list[float] = []
    history_a: list[dict] = []
    history_b: list[dict] = []

    for i, sentences_raw in enumerate(pages, start=1):
        sentences = _parse_sentences(sentences_raw)
        print(f"\n  ── page {i} 처리 중 ──")
        print(f"  텍스트: {sentences[0][:60]}...")

        page_a, time_a = _time_page(args.model_a, sentences, bible_a, setting, history_a)
        page_b, time_b = _time_page(args.model_b, sentences, bible_b, setting, history_b)
        page_times_a.append(time_a)
        page_times_b.append(time_b)
        print(f"  {args.model_a}: {time_a:.2f}s | {args.model_b}: {time_b:.2f}s | 차이 {time_a-time_b:+.2f}s")

        # 히스토리 갱신 (각 모델별 별도 흐름)
        history_a.append({
            "page_no": i, "sentences": sentences,
            "narrative_hint": page_a["narrative_hint"], "focus_roles": page_a["focus_roles"],
        })
        history_b.append({
            "page_no": i, "sentences": sentences,
            "narrative_hint": page_b["narrative_hint"], "focus_roles": page_b["focus_roles"],
        })

        _print_page_diff(page_a, page_b, args.model_a, args.model_b, i)
        page_results.append({
            "page_no": i, "sentences": sentences,
            args.model_a: {"time": time_a, **page_a},
            args.model_b: {"time": time_b, **page_b},
        })

    # ── 최종 요약 ──
    total_a = bible_time_a + sum(page_times_a)
    total_b = bible_time_b + sum(page_times_b)
    avg_page_a = sum(page_times_a) / len(page_times_a) if page_times_a else 0
    avg_page_b = sum(page_times_b) / len(page_times_b) if page_times_b else 0

    print(f"\n\n============================================================")
    print(f"  최종 요약")
    print(f"============================================================")
    print(f"  bible 시간:        {args.model_a} {bible_time_a:6.2f}s  |  {args.model_b} {bible_time_b:6.2f}s")
    print(f"  페이지 평균 시간:  {args.model_a} {avg_page_a:6.2f}s  |  {args.model_b} {avg_page_b:6.2f}s")
    print(f"  GPT 합계 ({len(pages)}p):    {args.model_a} {total_a:6.2f}s  |  {args.model_b} {total_b:6.2f}s")
    print(f"  → {args.model_b} 가 {(total_a - total_b):+.2f}s "
          f"{'빠름' if total_b < total_a else '느림'} ({total_a/total_b:.2f}배 차이)")

    # ── 저장 ──
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "benchmark" / "results" / f"{ts}_{Path(args.story).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "story_file": args.story,
            "setting": setting,
            "char_species": char_species,
            "characters": characters,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "bible": {
                args.model_a: {"time": bible_time_a, "output": bible_a},
                args.model_b: {"time": bible_time_b, "output": bible_b},
            },
            "pages": page_results,
            "summary": {
                "total_time": {args.model_a: total_a, args.model_b: total_b},
                "avg_page_time": {args.model_a: avg_page_a, args.model_b: avg_page_b},
            },
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  결과 저장: {out_dir / 'result.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
