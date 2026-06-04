# examples

`test_send.py --story <path>`에 넘기는 동화 시퀀스 JSON 모음.
각 파일은 의도적으로 다른 `setting`/`char_species`/캐릭터 종을 골라
anatomy 가드와 문화권 가드를 두루 검증할 수 있도록 짜여 있다.

## 파일 목록

| 파일 | setting | char_species | 검증 포인트 |
|---|---|---|---|
| `sample_story.json` | FOREST_NATURE | ANIMAL | BIRD(까마귀) + MAMMAL(여우) + AMPHIBIAN(개구리) 종 혼합 |
| `sample_korean_human.json` | KOREAN_TRADITIONAL | HUMAN | 한국 전통 복식(한복/도포/저고리) + HUMAN 캐스팅 |
| `sample_fantasy_etc.json` | FANTASY_WORLD | ETC | 정령/요정 같은 ETC 캐스팅 + 판타지 화풍 |
| `sample_underwater_animal.json` | UNDERWATER | ANIMAL | FISH(물고기/상어) + REPTILE(거북이) 가드 + 수중 배경 |

각 파일은 1~3문장 가변 페이지 구성을 섞어 페이지 단위 가변성도 함께 검증한다.

## 사용

```powershell
# 동화 1편을 INIT + 시퀀스 페이지로 한 번에 publish
python test_send.py --id 9101 --story examples/sample_korean_human.json
python test_send.py --id 9102 --story examples/sample_fantasy_etc.json
python test_send.py --id 9103 --story examples/sample_underwater_animal.json
```

`--init-wait` / `--page-delay` / `--skip-init` 같은 옵션은 [test_send.py](../test_send.py) 상단 docstring 참고.
