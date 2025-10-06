# DATA Directory Structure & Tracking Policy

This repository separates data assets by lifecycle stage to keep Git lean while preserving essential reference datasets.

## Folders (Simplified)

- raw/        : Original source files (받은 그대로·필수 참조). 소형 CSV/XLSX 또는 선택한 행정경계(shapefile). 최소만 Git 추적.
- processed/  : 분석·시각화 즉시 활용 최종 정제본 (스크립트로 재생성 가능, 대용량은 커밋 피하기 권장).
- cache/      : 지오코딩 캐시(JSON), API 응답, 임시 파생 인덱스 (전부 재생성 가능. 기본 ignore).

## Git Tracking Rules

- Keep: 선택한 raw 소스 + 필요하면 단일 경계폴더(`BND_ADM_DONG_PG`).
- Ignore: processed/ (내용), cache/ (내용) → `.gitignore` 에서 재귀 패턴 처리.
- Shapefiles: 기본적으로 무시, 화이트리스트 폴더만 예외 유지.
- Notebook 산출(이미지/HTML)은 추적 안 함; `.ipynb`만 유지.

## Naming Conventions

```
<topic>_geocoded_<YYYYMMDD_HHMM>.csv       # time-stamped
<topic>_geocoded_enriched.csv              # canonical enriched version
<topic>_geocoded_debug.csv                 # optional debug diagnostics
<topic>_syn_filtered.csv                   # subset (신속통합기획)
```

## Reproducibility Notes

1. Any file under ignored folders must be regenerable via scripts (document command in README or script --help).
2. Geocoding cache JSON lives in cache/ and can be optionally archived if reproducibility across API changes is critical.
3. Do not manually edit processed/ outputs – adjust upstream script logic instead.

## Adding New Authoritative Data

1. 새 원본 → raw/ 배치 (가능하면 파일명에 날짜 또는 출처 약어 포함).
2. 출처(URL, 수집일) README 상단 또는 `SOURCES.md`(선택) 기록.
3. 스크립트 실행 → 결과를 processed/ 로 생성 (원본 수정 금지, 재생성 가능 상태 유지).

## Whitelisting Additional Shapefiles

새로운 행정/분석 경계를 Git에 포함하려면:
1. `DATA/BND_<NAME>_PG/` 폴더 생성
2. `.gitignore` 에 예외 추가: `!DATA/BND_<NAME>_PG/**`
3. 필수 파일(.shp/.shx/.dbf/.prj/.cpg)만 유지 (거대한 중복/추가 인덱스 삭제)

## Cleanup Guidance

- cache/ 내부 캐시 JSON 오래된 것 수시 삭제 가능.
- processed/ 결과가 많아 용량 커지면 필요 최소만 남기고 재생성 전략 유지.
- 커밋 전에 대형 바이너리(raw 제외)가 추적되지 않았는지 확인.

## Future Enhancements

- (선택) DVC 또는 Git LFS: raw 파일이 커질 경우 고려.
- raw 무결성 해시(manifest) 자동 생성 스크립트.

