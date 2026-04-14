"""동화 텍스트에서 캐릭터·세계관을 자동 추출하는 분석기 (Klein 독립 복사본).

LLM 실패 시 fallback으로 사용되는 키워드 매핑 분석기.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 종(species) 탐지 테이블
# ─────────────────────────────────────────────────────────────────────────────
_SPECIES_TABLE: list[tuple[list[str], str]] = [
    (["개구리", "frog"], "frog"),
    (["다람쥐", "squirrel"], "squirrel"),
    (["거북이", "거북", "turtle", "tortoise"], "turtle"),
    (["코끼리", "elephant"], "elephant"),
    (["원숭이", "monkey"], "monkey"),
    (["강아지", "dog", "puppy"], "dog"),
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
    (["조랑말", "망아지", "horse"], "horse"),
    (["파랑새", "두루미", "까마귀", "참새", "제비", "까치", "bird"], "bird"),
    (["닭", "chicken", "rooster"], "bird"),
    (["오리", "duck"], "bird"),
    (["어린양", "양떼", "lamb", "sheep"], "deer"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 직업(job) 탐지 테이블
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
    (["나무꾼", "woodcutter"], "woodcutter"),
    (["마녀", "witch"], "witch"),
    (["마법사", "wizard"], "wizard"),
    (["상인", "장사꾼", "merchant"], "merchant"),
    (["백성", "마을 사람", "villager"], "villager"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 세계관(theme) 탐지 테이블
# ─────────────────────────────────────────────────────────────────────────────
_THEME_TABLE: list[tuple[list[str], str]] = [
    (
        [
            "한옥", "한복", "전래", "조선", "임금님", "임금", "사또", "원님",
            "대감", "포졸", "갓", "두루마기", "저고리", "치마", "한마을",
            "초가집", "기와집", "마을", "고을", "장터", "논밭", "gonryongpo",
            "hanbok", "hanok", "joseon", "korean folk", "나무꾼", "선녀",
        ],
        "KOREAN_TRADITIONAL",
    ),
    (["바다", "바닷속", "용궁", "물속", "산호", "해저", "ocean", "underwater", "sea kingdom"], "UNDERWATER"),
    (["하늘나라", "천국", "구름나라", "하늘 위", "sky", "heaven", "cloud kingdom"], "SKY_HEAVEN"),
    (["마법", "요정", "마법사", "용", "성", "dragon", "fairy", "magic", "wizard", "castle"], "FANTASY_WORLD"),
    (["기사", "성주", "knight", "medieval", "kingdom"], "EUROPEAN_MEDIEVAL"),
    (["도시", "현대", "city", "modern"], "MODERN_FANTASY"),
    (["숲", "나무", "강", "산", "들판", "꽃밭", "forest", "woods", "nature"], "FOREST_NATURE"),
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
    """동화 전체 텍스트를 분석해 build_story_plan()에 전달할 dict를 반환한다."""
    result = _analyze_internal(text)
    lines = _extract_lines(text)

    characters: list[dict[str, Any]] = []

    protagonist: dict[str, Any] = {
        "id": "protagonist",
        "type": result.protagonist_type,
        "species": result.protagonist_species,
        "job": result.protagonist_job,
        "traits": ["cute", "small"] if result.protagonist_type == "animal" else ["cute"],
    }
    characters.append(protagonist)

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
    normalized = text.lower()
    theme = _detect_theme(normalized)
    protagonist_species, supporting_species = _detect_species(normalized)

    protagonist_job: str | None = None
    if protagonist_species is None:
        protagonist_job = _detect_job(normalized)

    if protagonist_species is not None:
        protagonist_type = "animal"
    elif protagonist_job is not None:
        protagonist_type = "human"
    else:
        human_hints = ["소년", "소녀", "아이", "어린이", "아저씨", "할머니", "할아버지",
                       "boy", "girl", "child", "man", "woman", "grandmother", "grandfather"]
        protagonist_type = "human" if any(h in normalized for h in human_hints) else "other"

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
    for keywords, theme_key in _THEME_TABLE:
        for kw in keywords:
            if kw.lower() in normalized:
                return theme_key
    return "FOREST_NATURE"


def _detect_species(normalized: str) -> tuple[str | None, list[str]]:
    raw_found: list[tuple[int, int, str]] = []
    for keywords, species in _SPECIES_TABLE:
        for kw in keywords:
            idx = normalized.find(kw.lower())
            if idx != -1:
                raw_found.append((idx, idx + len(kw), species))
                break

    if not raw_found:
        return None, []

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

    filtered.sort(key=lambda x: x[0])
    protagonist_species = filtered[0][1]
    supporting_species = [s for _, s in filtered[1:] if s != protagonist_species]
    return protagonist_species, supporting_species


def _detect_job(normalized: str) -> str | None:
    for keywords, job in _JOB_TABLE:
        for kw in keywords:
            if kw.lower() in normalized:
                return job
    return None


def _extract_lines(text: str) -> list[str]:
    raw_lines = re.split(r"[\r\n]+", text.strip())
    lines = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        sentences = re.split(r"(?<=[.!?。！？])\s+", raw)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= 5:
                lines.append(sent)
    return lines


def infer_theme_from_text(text: str) -> str:
    """characters는 이미 알고 있고 theme만 자동 탐지할 때 사용."""
    return _detect_theme(text.lower())
