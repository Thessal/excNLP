## Named Entity Tags

### 해양대

* proprietary/data/NER_KMOU.csv

개체이름, 시간표현, 수량표현

개체이름은 인 명(PER), 지명(LOC), 기관명(ORG), 기타(POH)로 나누었고, 
시간표현은 날짜(DAT), 시간(TIM), 기간 (DUR)로 나누었으며, 
수량표현은 통화(MNY), 비율(PNT), 기타 수량표현(NOH)으로 나누었다.

| 대분류 | 분류 | 태그 |
|---|---|---|
| 개체이름 | 인명 | PER |
| | 지명 | LOC |
| | 기관명 | ORG |
| | 기타 | POH |
| 시간표현 | 날짜 | DAT |
| | 시간 | TIM |
| | 기간 | DUR |
| 수량표현 | 통화 | MNY |
| | 비율 | PNT |
| | 기타 수량표현 | NOH |
| 해당없음 | | _ |
| ? | | nan |

```
set(sum([x.split('+') for x in tags],[]))
{'', 'EC', 'ECMAJ', 'ECVV', 'ECXSV', 'EF', 'EFNNB', 'EP', 'ETM', 'ETMNNG', 'ETMXPN', 'ETN', 'IC', 'JC', 'JKB', 'JKC', 'JKCMAG', 'JKG', 'JKGNNP', 'JKO', 'JKOVV', 'JKQ', 'JKS', 'JKSVA', 'JKV', 'JX', 'JXNNG', 'JXNP', 'JXVA', 'MAG', 'MAJ', 'MM', 'NNB', 'NNG', 'NNP', 'NP', 'NR', 'SE', 'SF', 'SH', 'SL', 'SN', 'SO', 'SP', 'SS', 'SW', 'SWNP', 'SWXSV', 'VA', 'VCN', 'VCP', 'VCPEC', 'VCPSN', 'VCPXSV', 'VV', 'VVNNP', 'VX', 'XPN', 'XR', 'XSA', 'XSB', 'XSN', 'XSV', 'XSVNP', '_', np.nan}
```

### 창원대 

CoNLL, Naver

* proprietary/data/NER_CNU.csv
* proprietary/data/NER_NAVER.csv

```
set([x.split('_')[0] for x in tags])
{'-', 'AFW', 'ANM', 'CVL', 'DAT', 'EVT', 'FLD', 'LOC', 'MAT', 'NUM', 'ORG', 'PER', 'PLT', 'TIM', 'TRM'}
set([x.split('_')[-1] for x in tags])
{'-', 'B', 'I'}
```

https://air.changwon.ac.kr/?page_id=10

| 개체명 | 범주 |태그 |정의 |
|---|---|---|---|
|1|PERSON|PER|실존, 가상 등 인물명에 해당 하는 것|
|2|FIELD|FLD|학문 분야 및 이론, 법칙, 기술 등|
|3|ARTIFACTS_WORKS|AFW|인공물로 사람에 의해 창조된 대상물|
|4|ORGANIZATION|ORG|기관 및 단체와 회의/회담을 모두 포함|
|5|LOCATION|LOC|지역명칭과 행정구역 명칭 등|
|6|CIVILIZATION|CVL|문명 및 문화에 관련된 용어|
|7|DATE|DAT|날짜|
|8|TIME|TIM|시간|
|9|NUMBER|NUM|숫자|
|10|EVENT|EVT|특정 사건 및 사고 명칭과 행사 등|
|11|ANIMAL|ANM|동물|
|12|PLANT|PLT|식물|
|13|MATERIAL|MAT|금속, 암석, 화학물질 등|
|14|TERM|TRM|의학 용어, IT곤련 용어 등 일반 용어를 총칭|

|태그 |정의 |
|---|---|
|*_B | Begin |
|*_I | Inside |
|- | Outside |