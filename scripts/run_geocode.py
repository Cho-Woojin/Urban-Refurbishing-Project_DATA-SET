#!/usr/bin/env python
"""CLI for batch geocoding CSV based on address column.

Example:
  python scripts/run_geocode.py --csv DATA/sample.csv --address-col 대표지번 \
      --cache outputs/geocode_cache.json --delay 1.0 --out outputs/geocoded.parquet

If lat/lon already exist they are preserved (missing rows retried).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

from geocode_pipeline import (
    load_csv_multi, normalize_addr, load_cache, save_cache,
    geocode_primary, geocode_retry
)


def parse_args():
    p = argparse.ArgumentParser(description='Batch geocode a CSV file with caching.')
    p.add_argument('--csv', required=True, type=Path, help='Input CSV path')
    p.add_argument('--address-col', required=True, help='Address column name')
    p.add_argument('--cache', type=Path, default=Path('geocode_cache.json'), help='Cache JSON path')
    p.add_argument('--delay', type=float, default=1.0, help='Delay seconds between geocoding queries (Nominatim >=1s 권장)')
    p.add_argument('--out', type=Path, required=True, help='Output file (.csv or .parquet)')
    p.add_argument('--city-prefix', default='서울특별시', help='Prefix city if address not starting with 서울')
    p.add_argument('--retry', action='store_true', help='Enable second pass retry with simplified address')
    p.add_argument('--user-agent', default='seoul-redev-geocode-cli', help='User-Agent for Nominatim')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        print('[ERROR] CSV not found:', args.csv); return 2

    df = load_csv_multi(args.csv)
    if args.address_col not in df.columns:
        print(f'[ERROR] Address column "{args.address_col}" not found. Available={list(df.columns)}')
        return 3

    # Normalize address
    df['정규화주소'] = df[args.address_col].apply(lambda x: normalize_addr(x, city=args.city_prefix))

    cache = load_cache(args.cache)

    # Prepare lat/lon
    if 'lat' not in df.columns: df['lat'] = None
    if 'lon' not in df.columns: df['lon'] = None

    need_geocode = df['lat'].isna().any() or df['lon'].isna().any()
    if need_geocode:
        geocode_primary(df, cache, args.delay, user_agent=args.user_agent)
        if args.retry:
            geocode_retry(df, cache, args.delay, user_agent=args.user_agent)
        save_cache(cache, args.cache)
    else:
        print('[INFO] Existing lat/lon complete → skipping geocode')

    # Output
    if args.out.suffix.lower() == '.parquet':
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print('[DONE] Wrote', args.out, 'rows=', len(df))
    return 0


if __name__ == '__main__':
    sys.exit(main())
