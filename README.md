# Urban Refurbishing Project — Data Join

간단한 목적: DATA 폴더의 CSV들을 스키마 정리 후 조인하고, 재현 가능한 방식으로 관리합니다.

## 구조
- `DATA/` 원본 데이터 (읽기 전용으로 취급)
- `src/` 스크립트와 유틸
- `outputs/` 처리 결과물 (gitignore 처리)

## 빠른 시작
1) Python 3.10+ 권장. 가상환경 생성 후 의존성 설치
2) 조인 스크립트 실행

## 데이터 구조 & 컬럼 유지 전략
지오코딩 및 후속 분석 단계에서 컬럼 손실을 방지하고 파생물(Data Products)을 명확히 구분하기 위한 전략입니다.

### 1. 컬럼 분류
- 식별/키: `사업번호` (가능하면 항상 유지, 없으면 surrogate index 사용)
- 위치 원천: `정비구역위치`, `자치구`, `법정동`, `대표지번`
- 단계/분류: `진행단계`, `진행단계_구간`, `신속통합기획`
- 수치/면적: `정비구역면적(㎡)`, `건축연면적(㎡)` 등 면적·세대 수치
- 용도/지정: `용도지역`, `용도지구`
- 계산/파생(추가됨): `lat`, `lon`, (디버그) `geocode_status`, `geocode_tag`, `geocode_query`, `success`, `precision_rank`, `jitter_applied`, `lat_raw`, `lon_raw`

### 2. 지오코딩 출력 모드 (scripts/geocode_location_priority.py)
| 모드 | 목적 | 포함 컬럼 |
|------|------|-----------|
| minimal (기본) | 후속 조인 전에 단순 좌표만 필요 | `lat, lon, success` |
| --append-latlon | 원본 컬럼 모두 + 좌표만 이어붙이기 | 원본 + `lat, lon` |
| --full-output | 디버그/정밀도 분석 포함 전체 | 원본 + 모든 `geocode_*` + `lat, lon, success` |
| --debug-out (보조) | append 모드 사용하면서 디버그 별도 저장 | 지정 경로에 full 유사 |

추가 옵션:
- `--prefer-lot-first`: 지번(combo) 후보를 우선 → 정밀도 향상
- `--refine-coarse --max-refine N`: 저정밀(coarse) 행에 2차 재시도
- `--dedupe-jitter --jitter-radius r`: 동일 좌표 시각 중첩 분산

### 3. 캐시 설계
`outputs/geocode_cache_location.json` : `{ query: [lat, lon] }`
- 실패도 `(None, None)` 로 기록해 불필요 재호출 최소화
- 향후 확장 제안: `{ query: {"lat":..,"lon":..,"provider":"nominatim","ts":..,"precision":..} }`

### 4. 버전 및 파일명 명명 규칙 (권장)
```
<주제>_geocoded_<YYYYMMDD_HHMM>.csv                # append 또는 minimal
<주제>_geocoded_enriched.csv                      # 최신 대표본 (append-latlon + refine + jitter)
<주제>_geocoded_debug.csv                         # --debug-out 결과 (full equivalent)
<주제>_syn_filtered.csv                            # 신속통합기획 필터링 결과
```

### 5. 컬럼 보존 체크리스트
- 조인/전처리 단계에서 드롭 전: 꼭 `config/join_keep_columns.yaml` 업데이트
- 파생 컬럼 추가 시: 원본 컬럼 명과 충돌하지 않도록 새 이름 사용
- notebook / 분석 스크립트는 `진행단계_구간` 우선, 없으면 `진행단계` fallback

### 6. 두 종류 핵심 데이터셋
1) 전체 사업장 (master): 최신 append-latlon or full 결과 → 분석 기본
2) 신속통합기획 subset: master에서 `신속통합기획` 값 존재 행 필터 → `syn_df`

subset 생성 시 컬럼 드롭 금지; 행 필터만 수행.

## DB(데이터셋) 관리 워크플로
1. 원본 수집/업데이트: `DATA/` 에만 추가 (수정 금지, 추후 diff 검증 가능)
2. 조인/정규화: (예: join_pipeline) → 중간 통합 CSV (핵심 단계/구간 포함) 생성
3. 지오코딩: `geocode_location_priority.py` 실행
	- 1차: append-latlon 모드로 캐시 warm-up (`--prefer-lot-first`, `--refine-coarse` 포함)
	- 2차: 필요 시 full-output or debug-out 확보
4. 품질 진단: `scripts/geocode_2.py` (중복, 정밀도, coarse rate)
5. 시각화/Notebook: enriched 파일 로드 → 행정동 경계 결합
6. Subset 파생: 신속통합기획 값 존재 필터 → 별도 파일명 저장
7. 배포/아카이브: 날짜 스냅샷(immutable) + `*_enriched` 최신 링크 공존

### 무결성 점검 포인트
- (A) 행 수 증가/감소: 원본 대비 지오코딩/필터링 단계에서 의도치 않은 drop 여부
- (B) 핵심 컬럼 존재 여부: `사업번호`, `정비구역위치`, `진행단계(_구간)`
- (C) geocode success rate / coarse rate (로그 출력)
- (D) 중복 좌표(동일 lat/lon 그룹) 크기 분포 (jitter 적용 전 수치 기록)

### 권장 자동화 (추후)
- pre-commit 훅: CSV 헤더 비교 → 누락/신규 컬럼 diff 경고
- nightly job: 최신 enriched 재생성 후 중복/정밀도 지표 Slack/메일 알림

## 자주 발생하는 문제 & 해결
| 증상 | 원인 | 해결 |
|------|------|------|
| 진행단계 컬럼 없음 | 전처리에서 드롭/파일 오타 | keep yaml 확인 후 재생성 |
| 많은 좌표 동일 | fallback(자치구/동) 과다 | lot-first + refine-coarse 재실행 |
| syn subset 시 컬럼 소실 | copy 후 drop 수행 | 행 필터만 하고 to_csv |
| jitter 후 실제 좌표 필요 | jitter가 시각화용 | `lat_raw/lon_raw` 사용 |
## 커밋 규칙 (제안)
- `feat:` 기능 추가, `fix:` 버그 수정, `docs:` 문서, `chore:` 반복작업/환경, `data:` 데이터 스키마 변경/매핑 추가
- 작은 단위로 커밋, 실행 가능 상태를 유지

## 작업 흐름 (제안)
1. `feature/<이슈-간단설명>` 브랜치 생성
2. 스크립트/매핑 업데이트 → 로컬 실행 → 결과 확인
3. PR 생성 → 코드/결과 검토 → `main`에 병합

## 실행 방법
설치 및 실행 방법은 아래 스크립트 헤더를 참고하세요.
