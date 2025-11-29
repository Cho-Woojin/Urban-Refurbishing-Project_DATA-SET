#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import argparse

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser(description='Export rows whose (lat,lon) coordinates are duplicated (>1 occurrences).')
    ap.add_argument('--input', required=True, type=Path, help='Geocoded full output file (csv or parquet) containing lat, lon columns')
    ap.add_argument('--out', required=True, type=Path, help='Output CSV path for duplicated rows')
    ap.add_argument('--min-count', type=int, default=2, help='Minimum occurrences to treat as duplicate cluster (default=2)')
    args = ap.parse_args()

    df = read_any(args.input)
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise SystemExit('lat/lon columns not found in input.')

    # Group by coordinates (exclude NaN)
    g = df.dropna(subset=['lat','lon']).groupby(['lat','lon']).size().reset_index(name='dup_count')
    dup_pairs = g[g['dup_count'] >= args.min_count].copy()
    if dup_pairs.empty:
        print('[INFO] No duplicate coordinate clusters found (min-count=%d).' % args.min_count)
        df.head(0).to_csv(args.out, index=False)
        return

    # Assign cluster id sorted by descending count
    dup_pairs = dup_pairs.sort_values('dup_count', ascending=False).reset_index(drop=True)
    dup_pairs['cluster_id'] = dup_pairs.index + 1

    merged = df.merge(dup_pairs, on=['lat','lon'], how='inner')
    merged = merged.sort_values(['dup_count','cluster_id'], ascending=[False, True])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding='utf-8-sig')

    print('[DONE] duplicate rows exported:')
    print('  total_rows          =', len(df))
    print('  duplicate_row_count =', len(merged))
    print('  duplicate_clusters  =', dup_pairs.shape[0])
    top = dup_pairs.head(10)
    print('\nTop clusters (first 10):')
    for _, r in top.iterrows():
        print(f"  cluster {int(r.cluster_id):>3}: count={int(r.dup_count)} lat={r.lat} lon={r.lon}")

if __name__ == '__main__':
    main()
