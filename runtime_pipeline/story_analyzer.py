"""동화 텍스트에서 캐릭터·세계관을 자동 추출하는 분석기.

설계 원칙:
  - 동화가 생성된 직후 1회 실행 → StoryInput 스키마를 완성
  - 주인공을 미리 알 수 없으므로 텍스트 기반으로 추론
  - 특정 캐릭터(개구리, 토끼 등)에 종속되지 않고 어떤 텍스트에도 작동
  - LLM API 호출 없이 키워드 매핑으로 처리 (속도 최우선)

사용 흐름:
    story_data = analyze_story(full_story_text)
    # 반환값: build_story_plan()에 바로 전달 가능한 dict
    plan = build_story_plan(story_data)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 종(species) 탐지 테이블
# 형식: (탐지 키워드 목록, 정규화된 영어 species 이름)
# 우선순위: 앞에 있을수록 먼저 매칭
# ─────────────────────────────────────────────────────────────────────────────
_SPECIES_TABLE: list[tuple[list[str], str]] = [
    # 3음절 이상 키워드를 앞에 배치 — 부분 문자열 오탐 방지
    (["개구리", "frog"], "frog"),
    (["다람쥐", "squirrel"], "squirrel"),
    (["거북이", "거북", "turtle", "tortoise"], "turtle"),
    (["코끼리", "elephant"], "elephant"),
    (["원숭이", "monkey"], "monkey"),
    (["강아지", "dog", "puppy"], "dog"),   # "개" 단독 제거 — 개구리 등 오탐 방지
    (["고양이", "cat", "kitten"], "cat"),
    (["호랑이", "tiger"], "tiger"),
    (["토끼", "rabbit", "bunny"], "rabbit"),
    (["여우", "fox"], "fox"),
    (["사슴", "deer"], "deer"),
    (["사자", "lion"], "lion"),
    (["돼지", "pig"], "pig"),
    (["늑대", "wolf"], "wolf"),
    (["용", "dragon"], "dragon"),
    (["곰", "bear"], "bear"),
    (["뱀", "snake"], "snake"),
    (["쥐", "mouse", "rat"], "mouse"),
    # "말" 단독 제거 — "말하다", "말씀" 오탐 방지. 조랑말/망아지로 대체
    (["조랑말", "망아지", "horse"], "horse"),
    # "새" 단독 제거 — "새벽", "새로운" 오탐 방지. 구체적 새 이름으로 대체
    (["파랑새", "두루미", "까마귀", "참새", "제비", "까치", "bird"], "bird"),
    (["닭", "chicken", "rooster"], "bird"),
    (["오리", "duck"], "bird"),
    # "양" 단독 제거 — "양말", "양쪽" 오탐 방지. 어린양/양떼로 대체
    (["어린양", "양떼", "lamb", "sheep"], "deer"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 직업(job) 탐지 테이블
# 형식: (탐지 키워드 목록, 정규화된 job 이름)
# ─────────────────────────────────────────────────────────────────────────────
_JOB_TABLE: list[tuple[list[str], str]] = [
    (["임금님", "임금", "왕", "대왕", "king", "emperor"], "king"),
    (["왕비", "왕후", "queen"], "queen"),
    (["왕자", "prince"], "prince"),
    (["공주", "princess"], "princess"),
    (["사또", "원님", "목사", "magistrate"], "magistrate"),
    (["선비", "학자", "서생", "scholar"], "scholar"),
    (["농부", "nongbu", "farmer"], "farmer"),
    (["스님", "중", "monk"], "monk"),
    (["장군", "병사", "군사", "soldier"], "soldier"),
    (["마녀", "witch"], "witch"),
    (["마법사", "wizard"], "wizard"),
    (["상인", "장사꾼", "merchant"], "merchant"),
    (["백성", "마을 사람", "villager"], "villager"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 세계관(theme) 탐지 테이블
# 형식: (탐지 키워드 목록, theme key)
# 우선순위: 앞에 있을수록 먼저 매칭
# ─────────────────────────────────────────────────────────────────────────────
_THEME_TABLE: list[tuple[list[str], str]] = [
    # 한국 전래 — 문화 키워드가 있으면 최우선
    (
        [
            "한옥", "한복", "전래", "조선", "임금님", "임금", "사또", "원님",
            "대감", "포졸", "갓", "두루마기", "저고리", "치마", "한마을",
            "초가집", "기와집", "마을", "고을", "장터", "논밭", "gonryongpo",
            "hanbok", "hanok", "joseon", "korean folk",
        ],
        "KOREAN_TRADITIONAL",
    ),
    # 수중
    (["바다", "바닷속", "용궁", "물속", "산호", "해저", "ocean", "underwater", "sea kingdom"], "UNDERWATER"),
    # 하늘/천상
    (["하늘나라", "천국", "구름나라", "하늘 위", "sky", "heaven", "cloud kingdom"], "SKY_HEAVEN"),
    # 판타지
    (["마법", "요정", "마법사", "용", "성", "성castle", "dragon", "fairy", "magic", "wizard", "castle"], "FANTASY_WORLD"),
    # 유럽 중세
    (["기사", "성주", "knight", "castle", "medieval", "kingdom"], "EUROPEAN_MEDIEVAL"),
    # 현대 판타지
    (["도시", "현대", "city", "modern"], "MODERN_FANTASY"),
    # 자연/숲 — 기본값
    (["숲", "나무", "강", "산", "들판", "꽃밭", "forest", "woods", "nature"], "FOREST_NATURE"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 주인공 위치 키워드 (가장 처음/중심 등장을 암시)
# ─────────────────────────────────────────────────────────────────────────────
_PROTAGONIST_POSITION_HINTS = [
    "주인공", "어린", "작은", "귀여운", "옛날에", "옛날 옛적에",
    "한", "어느", "a young", "a small", "a cute", "once upon", "there was",
]


@dataclass
class AnalyzedStory:
    """analyze_story()의 결과 스키마."""

    theme: str
    protagonist_species: str | None
    protagonist_job: str | None
    protagonist_type: str
    supporting_species: list[str]
    confidence: str  # "high" | "medium" | "low"


def analyze_story(text: str) -> dict[str, Any]:
    """동화 전체 텍스트를 분석해 build_story_plan()에 전달할 dict를 반환한다.

    Args:
        text: 동화 전체 텍스트 (줄바꿈 포함 또는 연속 문자열)

    Returns:
        {
            "theme": str,
            "characters": [{"id": ..., "type": ..., "species": ..., "job": ...}],
            "lines": [str, ...]
        }
    """
    result = _analyze_internal(text)
    lines = _extract_lines(text)

    characters: list[dict[str, Any]] = []

    # 주인공 캐릭터
    protagonist: dict[str, Any] = {
        "id": "protagonist",
        "type": result.protagonist_type,
        "species": result.protagonist_species,
        "job": result.protagonist_job,
        "traits": ["cute", "small"] if result.protagonist_type == "animal" else ["cute"],
    }
    characters.append(protagonist)

    # 조연 동물이 있으면 추가 (최대 1명 — 프롬프트 복잡도 제한)
    if result.supporting_species:
        characters.append(
            {
                "id": "supporting",
                "type": "animal",
                "species": result.supporting_species[0],
                "job": None,
                "traits": ["friendly"],
            }
        )

    return {
        "theme": result.theme,
        "characters": characters,
        "lines": lines,
    }


def _analyze_internal(text: str) -> AnalyzedStory:
    """내부 분석 — 텍스트에서 종·직업·세계관을 추론한다."""
    normalized = text.lower()

    # ── 세계관 탐지 ──────────────────────────────────────────────────────────
    theme = _detect_theme(normalized)

    # ── 종(species) 탐지 ─────────────────────────────────────────────────────
    protagonist_species, supporting_species = _detect_species(normalized)

    # ── 직업 탐지 (종이 없을 때만 사용) ─────────────────────────────────────
    protagonist_job: str | None = None
    if protagonist_species is None:
        protagonist_job = _detect_job(normalized)

    # ── 캐릭터 타입 결정 ─────────────────────────────────────────────────────
    if protagonist_species is not None:
        protagonist_type = "animal"
    elif protagonist_job is not None:
        protagonist_type = "human"
    else:
        # 텍스트에서 인간임을 암시하는 단어 탐지
        human_hints = ["소년", "소녀", "아이", "어린이", "아저씨", "할머니", "할아버지",
                       "boy", "girl", "child", "man", "woman", "grandmother", "grandfather"]
        protagonist_type = "human" if any(h in normalized for h in human_hints) else "other"

    # ── 신뢰도 ───────────────────────────────────────────────────────────────
    if protagonist_species or protagonist_job:
        confidence = "high"
    elif protagonist_type != "other":
        confidence = "medium"
    else:
        confidence = "low"

    return AnalyzedStory(
        theme=theme,
        protagonist_species=protagonist_species,
        protagonist_job=protagonist_job,
        protagonist_type=protagonist_type,
        supporting_species=supporting_species,
        confidence=confidence,
    )


def _detect_theme(normalized: str) -> str:
    """텍스트에서 세계관을 탐지한다. 매칭 없으면 FOREST_NATURE 반환."""
    for keywords, theme_key in _THEME_TABLE:
        for kw in keywords:
            if kw.lower() in normalized:
                return theme_key
    return "FOREST_NATURE"


def _detect_species(normalized: str) -> tuple[str | None, list[str]]:
    """텍스트에서 동물 종을 탐지한다.

    Returns:
        (protagonist_species, [supporting_species, ...])
        protagonist는 텍스트에서 가장 먼저 언급된 종.
    """
    raw_found: list[tuple[int, int, str]] = []  # (start, end, species)

    for keywords, species in _SPECIES_TABLE:
        for kw in keywords:
            idx = normalized.find(kw.lower())
            if idx != -1:
                raw_found.append((idx, idx + len(kw), species))
                break  # 같은 species의 다른 키워드는 탐색 불필요

    if not raw_found:
        return None, []

    # 다른 더 긴 매칭에 완전히 포함된 짧은 매칭 제거
    # 예: "개" (dog, len=1) 가 "개구리" (frog, len=3) 안에 포함된 경우 제거
    filtered: list[tuple[int, str]] = []
    for start, end, species in raw_found:
        covered = any(
            s2 <= start and end <= e2 and (e2 - s2) > (end - start)
            for s2, e2, _ in raw_found
        )
        if not covered:
            filtered.append((start, species))

    if not filtered:
        return None, []

    # 위치순 정렬
    filtered.sort(key=lambda x: x[0])

    protagonist_species = filtered[0][1]
    supporting_species = [s for _, s in filtered[1:] if s != protagonist_species]

    return protagonist_species, supporting_species


def _detect_job(normalized: str) -> str | None:
    """텍스트에서 직업/신분을 탐지한다."""
    for keywords, job in _JOB_TABLE:
        for kw in keywords:
            if kw.lower() in normalized:
                return job
    return None


def _extract_lines(text: str) -> list[str]:
    """동화 텍스트를 문장 단위로 분리한다."""
    # 줄바꿈 기준으로 먼저 분리
    raw_lines = re.split(r"[\r\n]+", text.strip())
    lines = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        # 한 줄 안에 문장 부호 기준으로 추가 분리
        sentences = re.split(r"(?<=[.!?。！？])\s+", raw)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= 5:
                lines.append(sent)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수: 이미 characters 정보가 있을 때 theme만 보완
# ─────────────────────────────────────────────────────────────────────────────
def infer_theme_from_text(text: str) -> str:
    """characters는 이미 알고 있고 theme만 자동 탐지할 때 사용."""
    return _detect_theme(text.lower())
