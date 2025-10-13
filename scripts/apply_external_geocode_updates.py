#!/usr/bin/env python
"""Apply external (manually/web geocoded) coordinate updates onto an existing geocoded file.

Usage example:
python scripts/apply_external_geocode_updates.py \
  --base outputs/주택정비형_신통_geocoded_full_new.csv \
  --external external/a.csv \
  --out outputs/주택정비형_신통_geocoded_full_merged.csv \
  --key-col 0 \
  --ext-lat-col lat --ext-lon-col lon \
  --only-improve

If key-col is an integer it is treated as a positional column index of both CSVs.
You may alternatively pass a column name (string). If the external file has different header names for id/lat/lon
use the respective parameters.

By default rows are updated if:
  * external lat/lon are not null AND
  * (only-improve): base status is coarse (gu_only / gu_dong / dong) OR base lat/lon are duplicated centroid OR base lat/lon differ.

It will add columns:
  - lat_prev, lon_prev, geocode_status_prev (previous values when updated)
  - update_source = 'external'
  - improved_flag (True/False)

Duplicate centroid heuristic: if a coordinate appears >= dup-threshold times in the base file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys

COARSE_KEYWORDS = ['gu_only', 'gu_dong', 'dong']

def read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)

def resolve_key_series(df: pd.DataFrame, key_col):
    if isinstance(key_col, int):
        return df.iloc[:, key_col]
    return df[key_col]

def main():
    ap = argparse.ArgumentParser(description='Merge external geocode (more precise) into existing geocoded dataset.')
    ap.add_argument('--base', required=True, type=Path, help='Base geocoded file (csv/parquet) with lat/lon + geocode_status.')
    ap.add_argument('--external', required=True, type=Path, help='External coordinates file (csv/parquet).')
    ap.add_argument('--out', required=True, type=Path, help='Output merged file.')
    ap.add_argument('--key-col', required=True, help='Key column name or 0-based index shared across files.')
    ap.add_argument('--ext-lat-col', default='lat', help='External latitude column name.')
    ap.add_argument('--ext-lon-col', default='lon', help='External longitude column name.')
    ap.add_argument('--only-improve', action='store_true', help='Only overwrite when precision/coarse improvement expected.')
    ap.add_argument('--dup-threshold', type=int, default=4, help='Duplicate centroid threshold for considering improvement.')
    args = ap.parse_args()

    # Normalize key-col type
    try:
        key_col = int(args.key_col)
    except ValueError:
        key_col = args.key_col

    base = read_any(args.base)
    ext = read_any(args.external)

    if isinstance(key_col, int):
        base_key = resolve_key_series(base, key_col)
        ext_key = resolve_key_series(ext, key_col)
        key_name = base.columns[key_col]
    else:
        if key_col not in base.columns:
            sys.exit(f'[ERROR] key {key_col} not in base columns')
        if key_col not in ext.columns:
            sys.exit(f'[ERROR] key {key_col} not in external columns')
        base_key = base[key_col]
        ext_key = ext[key_col]
        key_name = key_col

    if args.ext_lat_col not in ext.columns or args.ext_lon_col not in ext.columns:
        sys.exit('[ERROR] External lat/lon columns not found.')

    ext_sub = ext[[key_name, args.ext_lat_col, args.ext_lon_col]].rename(columns={args.ext_lat_col: 'ext_lat', args.ext_lon_col: 'ext_lon'})

    merged = base.merge(ext_sub, on=key_name, how='left')

    # Duplicate centroid heuristic
    dup_counts = base.groupby(['lat','lon']).size().reset_index(name='dup_count')
    merged = merged.merge(dup_counts, on=['lat','lon'], how='left')

    def is_coarse(status: str) -> bool:
        if not isinstance(status, str):
            return False
        # remove cache/fallback decorations; coarse if any keyword present at start
        core = status.split('_cache')[0]
        for kw in COARSE_KEYWORDS:
            if kw in core:
                return True
        return False

    updated_flags = []
    lat_prev_list, lon_prev_list, status_prev_list = [], [], []

    for i, row in merged.iterrows():
        ext_lat = row.get('ext_lat')
        ext_lon = row.get('ext_lon')
        if pd.isna(ext_lat) or pd.isna(ext_lon):
            updated_flags.append(False)
            lat_prev_list.append(None); lon_prev_list.append(None); status_prev_list.append(None)
            continue
        # skip if identical already
        if not pd.isna(row['lat']) and not pd.isna(row['lon']) and float(row['lat']) == float(ext_lat) and float(row['lon']) == float(ext_lon):
            updated_flags.append(False)
            lat_prev_list.append(None); lon_prev_list.append(None); status_prev_list.append(None)
            continue
        do_update = True
        if args.only_improve:
            coarse = is_coarse(str(row.get('geocode_status','')))
            high_dup = (row.get('dup_count',0) or 0) >= args.dup_threshold
            # update if base coarse OR high duplicate OR base empty
            do_update = coarse or high_dup or pd.isna(row['lat']) or pd.isna(row['lon'])
        if do_update:
            lat_prev_list.append(row.get('lat'))
            lon_prev_list.append(row.get('lon'))
            status_prev_list.append(row.get('geocode_status'))
            merged.at[i,'lat'] = ext_lat
            merged.at[i,'lon'] = ext_lon
            merged.at[i,'geocode_status'] = (str(row.get('geocode_status')) + '_ext').strip('_')
            updated_flags.append(True)
        else:
            updated_flags.append(False)
            lat_prev_list.append(None); lon_prev_list.append(None); status_prev_list.append(None)

    merged['lat_prev'] = lat_prev_list
    merged['lon_prev'] = lon_prev_list
    merged['geocode_status_prev'] = status_prev_list
    merged['update_source'] = merged['ext_lat'].notna().map(lambda x: 'external' if x else None)
    merged['improved_flag'] = updated_flags

    # Output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == '.parquet':
        merged.to_parquet(args.out, index=False)
    else:
        merged.to_csv(args.out, index=False, encoding='utf-8-sig')

    improved = sum(updated_flags)
    print(f'[DONE] External integration finished: updated {improved} rows (base={len(base)}, external_match={ext_sub.shape[0]}).')
    print('Sample updated rows:')
    print(merged[merged['improved_flag']].head(5)[[key_name,'lat_prev','lon_prev','lat','lon','geocode_status_prev','geocode_status']])

if __name__ == '__main__':
    main()
