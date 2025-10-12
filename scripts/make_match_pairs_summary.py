#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a concise Experiment-Control pairs summary from scale_loc_matches.csv

Usage:
  python scripts/make_match_pairs_summary.py \
    --in outputs/matching_scale_loc/scale_loc_matches.csv \
    --out outputs/matching_scale_loc/match_pairs_summary.csv \
    --limit 0  # per-experiment control rows limit (0 = no limit)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True, help='Input matches CSV path')
    ap.add_argument('--out', dest='out_path', required=True, help='Output summary CSV path')
    ap.add_argument('--limit', type=int, default=0, help='Per-experiment control row limit (0 = unlimited)')
    ap.add_argument('--order', choices=['asc','desc'], default='asc', help='Score sort order (asc: low score first, desc: high score first)')
    ap.add_argument('--encoding', default='utf-8-sig')
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"[ERROR] Input not found: {in_path}")

    # Read with fallback encodings
    df = None
    for enc in [args.encoding, 'utf-8', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(in_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise SystemExit(f"[ERROR] Failed to read CSV: {in_path}")

    # Column detection
    exp_id_col = '_exp_id' if '_exp_id' in df.columns else None
    exp_code_col = '_exp_code' if '_exp_code' in df.columns else None
    relax_col = '_relax' if '_relax' in df.columns else None
    score_col = '_score' if '_score' in df.columns else None

    ctrl_name_candidates = ['정비구역명칭','정비구역명','사업명','구역명','사업장명']
    ctrl_code_candidates = ['사업번호','사업코드','ID','id','식별자','정비구역코드']
    district_candidates = ['자치구','구','시군구명','시군구','구청']

    ctrl_name_col = next((c for c in ctrl_name_candidates if c in df.columns), None)
    ctrl_code_col = next((c for c in ctrl_code_candidates if c in df.columns), None)
    district_col = next((c for c in district_candidates if c in df.columns), None)

    cols = []
    if exp_id_col: cols.append(exp_id_col)
    if exp_code_col: cols.append(exp_code_col)
    if ctrl_name_col: cols.append(ctrl_name_col)
    if ctrl_code_col: cols.append(ctrl_code_col)
    if district_col: cols.append(district_col)
    if score_col: cols.append(score_col)
    if relax_col: cols.append(relax_col)

    if not cols:
        raise SystemExit('[ERROR] No recognizable columns to build summary')

    summary = df[cols].copy()

    # Sort: by exp_id then score with configurable order (default asc)
    sort_by = []
    ascending = []
    if exp_id_col:
        sort_by.append(exp_id_col); ascending.append(True)
    if score_col:
        sort_by.append(score_col); ascending.append(args.order == 'asc')
    if sort_by:
        summary = summary.sort_values(by=sort_by, ascending=ascending)

    # Optional per-experiment limit
    if args.limit and args.limit > 0 and exp_id_col:
        summary = summary.groupby(exp_id_col, group_keys=False).head(args.limit)

    # Rename to user-friendly headers
    rename_map = {}
    if exp_id_col: rename_map[exp_id_col] = '실험군(정비구역명칭)'
    if exp_code_col: rename_map[exp_code_col] = '실험군_코드'
    if ctrl_name_col: rename_map[ctrl_name_col] = '대조군(정비구역명칭)'
    if ctrl_code_col: rename_map[ctrl_code_col] = '대조군_코드'
    if district_col: rename_map[district_col] = '자치구'
    if score_col: rename_map[score_col] = '점수(_score)'
    if relax_col: rename_map[relax_col] = '완화사용(relax)'
    summary = summary.rename(columns=rename_map)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[SAVE] {out_path} rows={len(summary)}")

    # Preview first 30 rows
    with pd.option_context('display.max_colwidth', 64):
        print(summary.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
