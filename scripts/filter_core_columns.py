#!/usr/bin/env python
"""핵심단계 CSV에서 지정된 핵심 컬럼만 추출하는 스크립트.

사용 예:
  python scripts/filter_core_columns.py \
      --input outputs/핵심단계_구간포함_preprocessed.csv \
      --out outputs/핵심단계_핵심컬럼.csv

기본 핵심 컬럼(요청 기반):
  사업번호, 자치구, 법정동, 운영구분, 대표지번, 진행단계, 상태, 토지등 소유자 수,
  정비구역명칭, 정비구역위치, 정비구역면적(㎡), 건축연면적(㎡), 용도지역, 용도지구,
  택지면적(㎡), 도로면적(㎡), 공원면적(㎡), 녹지면적(㎡), 공공공지면적(㎡), 학교면적(㎡),
  기타면적(㎡), 주용도, 건폐율, 용적률, 높이(m), 지상층수, 지하층수, 분양세대총수,
  60㎡이하, 60㎡초과~85㎡이하, 85㎡초과, 임대세대총수, (임대)40㎡이하,
  (임대)40㎡초과~50㎡이하, (임대)50㎡초과, 진행단계_구간

추가 규칙:
  - 요청 목록에 없던 면적과대플래그 등은 제거
  - 누락된 컬럼은 경고를 출력하고 건너뜀
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

CORE_COLUMNS = [
    "사업번호","자치구","법정동","운영구분","대표지번","진행단계","상태","토지등 소유자 수",
    "정비구역명칭","정비구역위치","정비구역면적(㎡)","건축연면적(㎡)","용도지역","용도지구",
    "택지면적(㎡)","도로면적(㎡)","공원면적(㎡)","녹지면적(㎡)","공공공지면적(㎡)","학교면적(㎡)",
    "기타면적(㎡)","주용도","건폐율","용적률","높이(m)","지상층수","지하층수","분양세대총수",
    "60㎡이하","60㎡초과~85㎡이하","85㎡초과","임대세대총수","(임대)40㎡이하",
    "(임대)40㎡초과~50㎡이하","(임대)50㎡초과","진행단계_구간"
]

ENCODINGS = ["utf-8-sig","utf-8","cp949","euc-kr"]

def read_csv_multi(path: Path):
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (last={last})")

def parse_args():
    ap = argparse.ArgumentParser(description="핵심 컬럼 필터")
    ap.add_argument('--input', required=True, type=Path, help='입력 CSV 경로')
    ap.add_argument('--out', required=True, type=Path, help='출력 CSV 경로')
    ap.add_argument('--list', action='store_true', help='핵심 컬럼 목록만 출력 후 종료')
    return ap.parse_args()

def main():
    args = parse_args()
    if args.list:
        print("\n".join(CORE_COLUMNS)); return 0
    if not args.input.exists():
        print('[ERROR] 입력 없음:', args.input); return 2
    df = read_csv_multi(args.input)
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        print('[WARN] 누락 컬럼 →', missing)
    keep = [c for c in CORE_COLUMNS if c in df.columns]
    out_df = df[keep].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'[DONE] 저장: {args.out} (cols={len(keep)})')
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
