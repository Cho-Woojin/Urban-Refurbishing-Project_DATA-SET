"""Geocode 결과 중복 좌표/정밀도 진단 스크립트

append-latlon 혹은 minimal 모드 출력에서는 geocode_status / geocode_tag 가 없을 수 있으므로
해당 컬럼이 없을 때는 관련 분석을 건너뛰고 안내 메시지를 출력한다.
"""

import pandas as pd, sys
from pathlib import Path

INPUT = Path("outputs/주택정비형_신통_geocoded_20240613_1530.csv")
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
print(dups.head(10))

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
    print('\n[INFO] geocode_status / geocode_tag 컬럼 없음 -> append-latlon 또는 minimal 출력으로 판단. 상세 정밀도 분석 생략.')

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
    print('\n[INFO] geocode_status 미존재 -> precision_rank 계산 불가')

# 4) 중복 좌표 클러스터 규모 통계
if not dups.empty:
    multi = dups[dups['count']>1]
    if not multi.empty:
        print('\n[DUP CLUSTER STATS]')
        print(' - 중복 클러스터 수:', len(multi))
        print(' - 중복 포함 행 비율: {:.3f}'.format(multi['count'].sum()/len(df)))
        print(' - 최대 클러스터 크기:', multi['count'].max())
    else:
        print('\n[DUP CLUSTER STATS] 중복 좌표 없음')

print('\n[DONE] 분석 완료')