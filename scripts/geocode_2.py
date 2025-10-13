"""Geocode 결과 중복 좌표/정밀도 진단 스크립트

append-latlon 혹은 minimal 모드 출력에서는 geocode_status / geocode_tag 가 없을 수 있으므로
해당 컬럼이 없을 때는 관련 분석을 건너뛰고 안내 메시지를 출력한다.
"""

import pandas as pd, sys, argparse
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description='Geocode 결과 중복/정밀도 진단')
    ap.add_argument('--input', type=Path, default=Path('outputs/주택정비형_신통_geocoded_full.csv'))
    ap.add_argument('--refine-candidates-out', type=Path, help='coarse 또는 proxy-coarse 행 목록 CSV 경로')
    ap.add_argument('--top-n', type=int, default=15, help='상위 중복 좌표 출력 수')
    ap.add_argument('--cluster-threshold', type=int, default=3, help='refine 후보로 간주할 최소 클러스터 크기')
    return ap.parse_args()

args = parse_args()
INPUT = args.input
if not INPUT.exists():
    sys.exit(f"[ERROR] 입력 파일 없음: {INPUT}")

df = pd.read_csv(INPUT)
required_coord = {'lat','lon'}
if not required_coord <= set(df.columns):
    sys.exit(f"[ERROR] lat/lon 컬럼이 필요합니다. 현재 컬럼: {list(df.columns)}")

# 1) 좌표 중복 상위
dups = (df
        .groupby(['lat','lon'], dropna=True)
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False))
print('[TOP DUP POINTS]')
print(dups.head(args.top_n))

has_status = 'geocode_status' in df.columns
has_tag = 'geocode_tag' in df.columns

if has_status or has_tag:
    # 2) 가장 많이 중복된 좌표의 상태/태그 분포
    if not dups.empty:
        top_lat, top_lon = dups.iloc[0][['lat','lon']].tolist()
        same = df[(df['lat']==top_lat) & (df['lon']==top_lon)]
        if has_status:
            print('\n[STATUS DISTRIBUTION of TOP DUP]')
            print(same['geocode_status'].value_counts())
        if has_tag:
            print('\n[TAG DISTRIBUTION of TOP DUP]')
            print(same['geocode_tag'].value_counts().head(10))
else:
    print('\n[INFO] geocode_status / geocode_tag 컬럼 없음 -> append-latlon 또는 minimal 출력으로 판단. proxy precision 추정 모드로 전환.')

# 3) 정밀도 스코어링 (status 있을 때만)
if has_status:
    PRECISION_RANK = {
        'dong_bunji_ok': 4,
        'combo_ok': 4,
        'combo_main_ok': 3,
        'dong_main_ok': 3,
        'dong_ok': 2,
        'gu_dong_ok': 1,
        'gu_only_ok': 0,
        'refined_ok': 4,
    }
    df['precision_rank'] = df['geocode_status'].map(PRECISION_RANK).fillna(2)
    print('\n[PRECISION RANK DISTRIBUTION]')
    print(df['precision_rank'].value_counts())
    low = df[df['precision_rank']<=1]
    print(f"[METRIC] 저정밀 행 비율: {len(low)/len(df):.3f} ({len(low)}/{len(df)})")
else:
    # Proxy precision: 클러스터 크기 기반 Heuristic
    # count>=5 -> very_coarse(0), count>=3 -> coarse(1), count==2 -> med(2), unique -> fine(3)
    cluster_map = dups.set_index(['lat','lon'])['count']
    def proxy_rank(row):
        c = cluster_map.get((row['lat'], row['lon']), 1)
        if c>=5: return 0
        if c>=3: return 1
        if c==2: return 2
        return 3
    df['precision_rank_proxy'] = df.apply(proxy_rank, axis=1)
    print('\n[PROXY PRECISION RANK DISTRIBUTION] (no geocode_status)')
    print(df['precision_rank_proxy'].value_counts().sort_index())
    low = df[df['precision_rank_proxy']<=1]
    print(f"[METRIC] proxy 저정밀 행 비율: {len(low)/len(df):.3f} ({len(low)}/{len(df)})")

# 4) 중복 좌표 클러스터 규모 통계
if not dups.empty:
    multi = dups[dups['count']>1]
    if not multi.empty:
        print('\n[DUP CLUSTER STATS]')
        print(' - 중복 클러스터 수:', len(multi))
        print(' - 중복 포함 행 비율: {:.3f}'.format(multi['count'].sum()/len(df)))
        print(' - 최대 클러스터 크기:', multi['count'].max())
        # refine 후보 산출: (a) geocode_status 있는 경우 저정밀+클러스터 (b) 없는 경우 proxy rank<=1 + 클러스터 size>=threshold
        refine_mask = None
        if has_status:
            # precision_rank 계산된 상태
            if 'precision_rank' in df.columns:
                refine_mask = (df['precision_rank']<=1)
        else:
            refine_mask = (df['precision_rank_proxy']<=1)
        if refine_mask is not None:
            cluster_keys = set(tuple(x) for x in multi[['lat','lon']].values if x[0]==x[0] and x[1]==x[1])
            refine_rows = df[refine_mask & df.apply(lambda r: (r['lat'],r['lon']) in cluster_keys, axis=1)].copy()
            if not refine_rows.empty:
                # dup_count 계산: 좌표 기준으로 dups 와 merge
                refine_rows = refine_rows.merge(dups, on=['lat','lon'], how='left', suffixes=('','_dup'))
                refine_rows.rename(columns={'count':'dup_count'}, inplace=True)
                refine_rows = refine_rows[refine_rows['dup_count']>=args.cluster_threshold]
            print(f"[REFINE CANDIDATES] rows={len(refine_rows)} (cluster_threshold={args.cluster_threshold})")
            if args.refine_candidates_out and len(refine_rows)>0:
                args.refine_candidates_out.parent.mkdir(parents=True, exist_ok=True)
                refine_rows.to_csv(args.refine_candidates_out, index=False, encoding='utf-8-sig')
                print(f"[REFINE CANDIDATES] 저장: {args.refine_candidates_out}")
    else:
        print('\n[DUP CLUSTER STATS] 중복 좌표 없음')

print('\n[DONE] 분석 완료')