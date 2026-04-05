# 데이터셋 수집 가이드

목표: 카테고리별 10~15장, 총 30~45장

---

## 📌 수집 방법

1. 아래 검색어로 Pinterest / Google Images 검색
2. 마음에 드는 이미지를 `dataset/raw/` 폴더에 저장
3. 파일명은 자유 (영문/숫자 권장): `korean_01.jpg`, `animal_03.png` 등
4. 이미지 다 모으면 → `python add_captions.py` 실행하면 캡션 자동 생성

---

## 🇰🇷 한국 전래동화 스타일 (목표: 10~15장)

**Pinterest 검색어:**
```
한국 전래동화 일러스트
korean fairy tale children illustration
hanbok children illustration storybook
전래동화 삽화 귀여운
korean folktale illustration cute
심청전 일러스트 동화
콩쥐팥쥐 일러스트
```

**Google Images 검색어:**
```
"전래동화" "일러스트" filetype:jpg
korean traditional fairy tale book illustration
hanbok girl boy storybook illustration soft colors
```

**원하는 스타일 특징:**
- 한복 입은 귀여운 어린이 캐릭터
- 수채화 또는 2D 일러스트 느낌
- 밝고 따뜻한 색감

---

## 🌍 서양 동화 스타일 (목표: 10~15장)

**Pinterest 검색어:**
```
children book illustration character 2D
fairytale children illustration soft colors
storybook illustration kids cute
children's book cover illustration
picture book art cute characters
```

**Google Images 검색어:**
```
children's book illustration style cute character
fairytale illustration 2D flat soft pastel colors
storybook art children character warm lighting
```

**원하는 스타일 특징:**
- 밝은 파스텔/비비드 색상
- 귀엽고 둥글둥글한 캐릭터 비율
- 따뜻한 실내/야외 배경

---

## 🐾 동물 캐릭터 스타일 (목표: 10~15장)

**Pinterest 검색어:**
```
cute animal children book illustration
storybook animal character art
fairytale animal illustration fox rabbit bear
children book animal character cute
동물 동화 일러스트 귀여운
```

**Google Images 검색어:**
```
cute animal storybook illustration children book
fairytale fox rabbit bear illustration soft colors
picture book animal character 2D illustration
```

**원하는 스타일 특징:**
- 귀여운 동물 캐릭터 (여우, 토끼, 곰, 거북이 등)
- 동화적인 배경 (숲, 연못, 들판)
- 아이들용 부드러운 색감

---

## ⚠️ 이미지 수집 시 주의사항

- **해상도**: 512x512 이상 권장 (자동으로 리사이즈 됨)
- **스타일 일관성**: 사진 스타일(실사)은 제외, 일러스트/만화 스타일만
- **다양성**: 같은 캐릭터 반복 사용 X, 다양한 장면/캐릭터
- **저작권**: 개인 프로젝트/연구 목적으로만 사용

---

## ✅ 완료 후 실행

```bash
# 이미지를 dataset/raw/ 에 넣은 후
python add_captions.py

# 데이터셋 현황 확인
python prepare_dataset.py --stats
```
