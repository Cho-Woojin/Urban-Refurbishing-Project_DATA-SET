#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
규모 + 위치(격자 클러스터) 기반 매칭 스크립트 (연도 무시)
=====================================================
요구사항:
- 연도 조건을 전혀 사용하지 않고 매칭
- 규모 변수: 정비구역면적(area), 토지등소유자수(members/owners), 용적률(floor_area_ratio), 세대총합계(total_households)
- 위치: 위도/경도를 격자(grid)로 클러스터링하여 동일 격자 우선 매칭 (격자 분해능 파라미터)
- 매칭률(요청한 controls_per_experiment 대비 확보율) 및 요약 지표 출력

사용 예시:
python scripts/match_scale_location.py \
  --csv outputs/주택재개발_DATA_full.csv \
  --output-dir outputs/matching_scale_loc \
  --controls-per-experiment 10 \
  --area-tol 0.25 --member-tol 0.25 --far-tol 0.30 --hh-tol 0.30 \
  --grid-resolution-deg 0.01

설명:
- 각 규모 변수별 허용 비율 tol 은 |ctrl-exp|/exp <= tol 방식 (exp=0 또는 NaN 은 제외)
- floor_area_ratio, total_households 중 없어도 남은 변수로 매칭 진행
- 위치 격자 동일 후보가 부족하면 (min_controls 미달) → 전체 후보 재평가(거리 패널티 방식) fallback
- 거리: 허버사인; 최대 거리 초과 시 clip / normalize
"""
from __future__ import annotations
import argparse, math, sys, json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# --------------------------- 컬럼 동의어 ---------------------------
COLUMN_SYNONYMS = {
    'id': ['사업번호','사업코드','ID','id','식별자','정비구역코드'],
    'project_name': ['정비구역명칭','정비구역명','사업명','구역명','사업장명','정비구역명'],  # 정비구역명칭 추가 및 우선 배치
    'exp_flag': ['신속통합기획','신속통합기획여부','신속통합기획_구분'],
    'district': ['자치구','구','시군구명','시군구','구청'],
    'area': ['정비구역면적(㎡)','정비구역면적','구역면적','사업면적','면적','면적(㎡)'],
    'members': ['토지등 소유자 수','토지등소유자수','조합원수','추정조합원수','조합원 수','세대총합계','세대수합계','세대수'],
    'floor_area_ratio': ['용적률'],
    'total_households': ['세대총합계','세대수합계'],
    'latitude': ['위도','lat','LAT','Lat'],
    'longitude': ['경도','lon','LON','Lon','LONG','Longitude','경도(E)','경도(East)'],
}
POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}

# --------------------------- 유틸 ---------------------------

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    mapping = {}
    lower = {c.lower(): c for c in df.columns}
    for canon, syns in COLUMN_SYNONYMS.items():
        for s in syns:
            if s in df.columns:
                mapping[canon] = s; break
            if s.lower() in lower:
                mapping[canon] = lower[s.lower()]; break
    return mapping

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(',','',regex=False).str.replace('%','',regex=False).str.strip(),
        errors='coerce'
    )

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --------------------------- 매칭 로직 ---------------------------

def build_experimental(df: pd.DataFrame, mapping: Dict[str,str], fixed_flag_values=None):
    flag_col = mapping.get('exp_flag')
    if flag_col and fixed_flag_values:
        mask_fixed = df[flag_col].isin(fixed_flag_values)
    elif flag_col:
        mask_fixed = df[flag_col].astype(str).str.strip().str.lower().isin(POSITIVE_FLAG_VALUES)
    else:
        mask_fixed = False
    # flag 기반 외 추가 키워드 (신속통합 등) row concat 검색
    if '_row_concat' not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        df['_row_concat'] = df[obj_cols].astype(str).agg(' '.join, axis=1)
    regex = r'(신속통합|신통|선정구역)'
    mask_regex = df['_row_concat'].str.contains(regex, regex=True, na=False)
    exp_mask = mask_fixed | mask_regex
    exp_df = df[exp_mask].copy()
    return exp_df


def make_grid_cluster(df: pd.DataFrame, lat_col: str, lon_col: str, resolution: float) -> pd.Series:
    # resolution (deg) ex 0.01 ≈ 1.1 km (latitude 기준)
    if lat_col not in df.columns or lon_col not in df.columns:
        return pd.Series([None]*len(df), index=df.index)
    lat = pd.to_numeric(df[lat_col], errors='coerce')
    lon = pd.to_numeric(df[lon_col], errors='coerce')
    grid_lat = np.floor(lat / resolution).astype('Int64')
    grid_lon = np.floor(lon / resolution).astype('Int64')
    cluster = grid_lat.astype(str) + '_' + grid_lon.astype(str)
    cluster[(lat.isna()) | (lon.isna())] = None
    return cluster


def compute_scale_diffs(exp_row, ctrl_df, cols_tols):
    rows = []
    for col, tol in cols_tols.items():
        if col not in ctrl_df.columns:
            continue
    # 벡터화 수행
    out = ctrl_df.copy()
    for col, tol in cols_tols.items():
        if col not in out.columns or pd.isna(exp_row.get(col)):
            out[f'_diff_{col}'] = np.nan
            out[f'_ok_{col}'] = False
            continue
        exp_val = exp_row[col]
        diff_abs = (out[col] - exp_val).abs()
        if exp_val == 0 or pd.isna(exp_val):
            diff_ratio = np.nan
            out[f'_diff_{col}'] = np.nan
            out[f'_ok_{col}'] = False
        else:
            diff_ratio = diff_abs / abs(exp_val)
            out[f'_diff_{col}'] = diff_ratio
            out[f'_ok_{col}'] = diff_ratio <= tol
    return out


def match_one(exp_row, ctrl_df, mapping, cols_tols, lat_col, lon_col, cluster_col, same_cluster_only,
              max_distance_km, weight_cols, weight_loc_same=0.0, weight_loc_other=1.0):
    # 1) (옵션) 동일 클러스터 필터
    if same_cluster_only and cluster_col in ctrl_df.columns:
        same_cluster_df = ctrl_df[ctrl_df[cluster_col] == exp_row.get(cluster_col)]
    else:
        same_cluster_df = ctrl_df
    work = compute_scale_diffs(exp_row, same_cluster_df, cols_tols)
    if work.empty:
        return work
    # 모든 scale 조건 True 인 행만 1차 필터
    ok_cols = [c for c in work.columns if c.startswith('_ok_')]
    cond_all = work[ok_cols].all(axis=1)
    filtered = work[cond_all].copy()
    if filtered.empty:
        return filtered
    # 위치 점수
    lat_e, lon_e = exp_row.get(lat_col), exp_row.get(lon_col)
    if lat_col in filtered.columns and lon_col in filtered.columns and not pd.isna(lat_e) and not pd.isna(lon_e):
        dists = haversine_km(lat_e, lon_e, filtered[lat_col].astype(float), filtered[lon_col].astype(float))
        dists = np.clip(dists, 0, max_distance_km)
        filtered['_dist_km'] = dists
        filtered['_loc_score'] = np.where(filtered[cluster_col] == exp_row.get(cluster_col), weight_loc_same, weight_loc_other * (dists / max_distance_km))
    else:
        filtered['_dist_km'] = np.nan
        filtered['_loc_score'] = 0.5  # 중립
    # 규모 점수 (가중 평균)
    scale_scores = []
    for col, w in weight_cols.items():
        dc = f'_diff_{col}'
        if dc in filtered.columns:
            scale_scores.append(w * filtered[dc])
    if scale_scores:
        filtered['_scale_score'] = np.nansum(scale_scores, axis=0)
    else:
        filtered['_scale_score'] = 0.0
    filtered['_score'] = filtered['_scale_score'] + filtered['_loc_score']
    return filtered

# --------------------------- 메인 ---------------------------

def main():
    ap = argparse.ArgumentParser(description='규모+위치(클러스터) 기반 매칭 (연도 무시)')
    ap.add_argument('--csv', required=False, help='입력 데이터 CSV 경로 (미지정 시 자동 탐색)')
    ap.add_argument('--output-dir', default='outputs/matching_scale_loc')
    ap.add_argument('--controls-per-experiment', type=int, default=10)
    ap.add_argument('--min-controls-per-experiment', type=int, default=5)
    ap.add_argument('--area-tol', type=float, default=0.30)  # 완화 (기존 0.25)
    ap.add_argument('--member-tol', type=float, default=0.30)  # 완화
    ap.add_argument('--far-tol', type=float, default=0.35, help='용적률 허용 비율 (완화 기본)')
    ap.add_argument('--hh-tol', type=float, default=0.35, help='세대총합계 허용 비율 (완화 기본)')
    ap.add_argument('--grid-resolution-deg', type=float, default=0.018, help='위경도 격자 해상도 (완화: 더 큰 격자)')
    ap.add_argument('--max-distance-km', type=float, default=8.0)
    ap.add_argument('--export-json', action='store_true')
    ap.add_argument('--encoding', default='utf-8-sig')
    ap.add_argument('--fixed-exp-values', default='1차선정구역,2차선정구역,기존구역(신통추진)')
    args = ap.parse_args()

    # 자동 CSV 탐색
    if not args.csv:
        candidates = [
            'outputs/주택재개발_DATA_full.csv'
        ]
        found = None
        for c in candidates:
            p = Path(c)
            if p.exists():
                found = p; break
        if found is None:
            # fallback: outputs 폴더 내 *_DATA_full*.csv 검색
            out_dir = Path('outputs')
            if out_dir.exists():
                globbed = list(out_dir.glob('*DATA_full*.csv'))
                if globbed:
                    found = globbed[0]
        if found is None:
            print('[ERROR] --csv 미지정이고 기본 후보에서도 CSV를 찾지 못했습니다. --csv 경로를 명시하세요.')
            sys.exit(1)
        else:
            args.csv = str(found)
            print(f'[AUTO] --csv 미지정 → 자동선택: {args.csv}')

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[ERROR] CSV 없음: {csv_path}'); sys.exit(1)

    tried = []
    for enc in [args.encoding,'utf-8','cp949','euc-kr']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f'[LOAD] {enc} -> {df.shape}')
            break
        except Exception as e:
            tried.append(f'{enc}:{e}')
            df = None
    if df is None:
        print('[ERROR] 로드 실패', tried); sys.exit(1)

    mapping = detect_columns(df)
    print('[MAP]', mapping)

    # 숫자 변환
    for k in ['area','members','floor_area_ratio','total_households']:
        col = mapping.get(k)
        if col and col in df.columns:
            df[col] = coerce_numeric(df[col])

    lat_col = mapping.get('latitude'); lon_col = mapping.get('longitude')
    if not lat_col or not lon_col:
        print('[WARN] 위도/경도 컬럼을 찾지 못함 -> 위치 점수 중립 처리')

    fixed_exp_values = {s.strip() for s in args.fixed_exp_values.split(',') if s.strip()}
    exp_df = build_experimental(df, mapping, fixed_flag_values=fixed_exp_values)
    print(f'[EXP] 실험군 {len(exp_df)}건 (중복 제거 전)')
    id_col = mapping.get('id') or mapping.get('project_name')
    name_col = mapping.get('project_name') or mapping.get('id')
    if id_col and id_col in exp_df.columns:
        exp_df = exp_df.drop_duplicates(subset=[id_col])
    print(f'[EXP] 중복 제거 후 {len(exp_df)}건')

    # 실험군/대조군 분리
    if id_col and id_col in df.columns:
        ctrl_df = df[~df[id_col].isin(exp_df[id_col])].copy()
    else:
        ctrl_df = df.loc[~df.index.isin(exp_df.index)].copy()
    print(f'[CTRL] 초기 후보 {len(ctrl_df)}건')

    # (추가) 핵심 규모 변수 결측 현황 로그
    for tag in ['area','members','floor_area_ratio','total_households']:
        col = mapping.get(tag)
        if col and col in df.columns:
            nn = df[col].notna().sum()
            print(f'[DIAG] {tag}({col}) non-null {nn}/{len(df)}')

    # 필수 규모값 결측 제거
    essential = [mapping.get('area'), mapping.get('members')]
    for c in essential:
        if c:
            before = len(exp_df); exp_df = exp_df[exp_df[c].notna()]; print(f'[EXP] 결측 제거 {c} {before}->{len(exp_df)}')
            before = len(ctrl_df); ctrl_df = ctrl_df[ctrl_df[c].notna()]; print(f'[CTRL] 결측 제거 {c} {before}->{len(ctrl_df)}')

    if exp_df.empty:
        print('[FATAL] 실험군 필수 값 모두 결측 -> 종료'); sys.exit(1)
    if ctrl_df.empty:
        print('[FATAL] 대조군 필수 값 모두 결측 -> 종료'); sys.exit(1)

    # 격자 클러스터
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        df['_cluster'] = make_grid_cluster(df, lat_col, lon_col, args.grid_resolution_deg)
        # 실험군/대조군 반영
        if id_col and id_col in df.columns:
            exp_df = df[df[id_col].isin(exp_df[id_col])].copy()
            ctrl_df = df[~df[id_col].isin(exp_df[id_col])].copy()
        else:
            exp_df['_cluster'] = make_grid_cluster(exp_df, lat_col, lon_col, args.grid_resolution_deg)
            ctrl_df['_cluster'] = make_grid_cluster(ctrl_df, lat_col, lon_col, args.grid_resolution_deg)
        print('[CLUSTER] 고유 클러스터 수(전체):', df['_cluster'].nunique(dropna=True))
    else:
        df['_cluster'] = None
        exp_df['_cluster'] = None
        ctrl_df['_cluster'] = None
        print('[WARN] 위치 클러스터링 불가 (위경도 부족)')

    # 매칭 파라미터 준비
    cols_tols = {}
    if mapping.get('area'): cols_tols[mapping['area']] = args.area_tol
    if mapping.get('members'): cols_tols[mapping['members']] = args.member_tol
    if mapping.get('floor_area_ratio') and mapping['floor_area_ratio'] in df.columns:
        cols_tols[mapping['floor_area_ratio']] = args.far_tol
    if mapping.get('total_households') and mapping['total_households'] in df.columns:
        cols_tols[mapping['total_households']] = args.hh_tol
    print('[PARAM] cols_tols=', cols_tols)

    # 가중치: 규모 변수 균등 + 위치
    n_scale = len(cols_tols) if cols_tols else 1
    weight_cols = {col: 1.0/n_scale for col in cols_tols.keys()}

    rows = []
    summary_rows = []
    for i, exp_row in exp_df.iterrows():
        # 내부 식별(코드)와 표기용 명칭 분리
        exp_code = exp_row.get(id_col, i)
        exp_label = exp_row.get(name_col, exp_code)
        exp_id = exp_label  # 사용자 요구: exp_id를 명칭으로 출력
        # 1단계: 동일 클러스터 안에서 시도
        res_same = match_one(exp_row, ctrl_df, mapping, cols_tols, lat_col, lon_col, '_cluster', True,
                              args.max_distance_km, weight_cols)
        picked = None
        relax_used = False
        if res_same.empty or len(res_same) < args.min_controls_per_experiment:
            # 2단계: 전체에서 위치 패널티 포함 재시도
            res_all = match_one(exp_row, ctrl_df, mapping, cols_tols, lat_col, lon_col, '_cluster', False,
                                 args.max_distance_km, weight_cols)
            if res_all.empty:
                print(f'[MATCH][MISS] exp_id={exp_id} 전체 fallback 도 0건 (cluster={exp_row.get("_cluster")})')
                picked = res_all  # empty
                relax_used = True
            else:
                if '_score' not in res_all.columns:
                    print(f'[WARN] exp_id={exp_id} fallback 결과에 _score 누락 -> 0으로 대체')
                    res_all['_score'] = 0.0
                picked = res_all.sort_values('_score').head(args.controls_per_experiment)
                relax_used = True
        else:
            if '_score' not in res_same.columns:
                print(f'[WARN] exp_id={exp_id} same-cluster 결과에 _score 누락 -> 0으로 대체')
                res_same['_score'] = 0.0
            picked = res_same.sort_values('_score').head(args.controls_per_experiment)
            relax_used = False

        if picked is not None and not picked.empty:
            picked = picked.copy()
            picked['_exp_id'] = exp_id  # 명칭 기반
            picked['_exp_code'] = exp_code  # 원래 코드 보존
            picked['_relax'] = relax_used
            rows.append(picked)
            summary_rows.append({
                'exp_id': exp_id,          # 표시용 명칭
                'exp_code': exp_code,      # 원 사업번호
                'picked': len(picked),
                'relax_used': relax_used,
                'cluster': exp_row.get('_cluster')
            })
        else:
            summary_rows.append({
                'exp_id': exp_id,
                'exp_code': exp_code,
                'picked': 0,
                'relax_used': True,
                'cluster': exp_row.get('_cluster')
            })

    matches = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    # (추가) 첫 3개 실험군 매칭 프리뷰
    if not matches.empty:
        print('[PREVIEW] top5 rows:\n', matches.head())
        first_ids = list(summary['exp_id'].head(3))
        for fid in first_ids:
            sub = matches[matches['_exp_id']==fid].head()
            if not sub.empty:
                print(f'[PREVIEW][exp_name={fid}] score head:\n', sub[['_exp_id','_exp_code','_score']].head())
            else:
                print(f'[PREVIEW][exp_name={fid}] (no rows)')

    # 매칭률 계산
    if not summary.empty:
        success = (summary['picked'] >= args.min_controls_per_experiment).sum()
        match_rate = success / len(summary)
        avg_controls = summary['picked'].mean()
    else:
        match_rate = 0.0; avg_controls = 0.0

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / 'scale_loc_matches.csv'
    summary_path = output_dir / 'scale_loc_summary.csv'
    matches.to_csv(matches_path, index=False, encoding='utf-8-sig')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'[SAVE] matches -> {matches_path} ({len(matches)} rows)')
    print(f'[SAVE] summary -> {summary_path}')
    print(f'[METRIC] match_rate(>=min_controls) = {match_rate:.2%}, avg_controls_per_exp = {avg_controls:.2f}')
    print(f'[INFO] 결과 디렉터리 절대경로: {output_dir.resolve()}')

    # 손상된 이전 export_json 블록 무시하고 새 블록 사용
    # export_json 블록 (정상)
    if args.export_json:
        meta = {
            'mapping': mapping,
            'n_experiments': int(len(summary)),
            'match_rate': float(match_rate),
            'avg_controls': float(avg_controls),
            'params': {
                'area_tol': args.area_tol,
                'member_tol': args.member_tol,
                'far_tol': args.far_tol,
                'hh_tol': args.hh_tol,
                'grid_resolution_deg': args.grid_resolution_deg,
                'max_distance_km': args.max_distance_km,
                'controls_per_experiment': args.controls_per_experiment,
                'min_controls_per_experiment': args.min_controls_per_experiment
            }
        }
        json_path = output_dir / 'scale_loc_metrics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f'[SAVE] metrics json -> {json_path}')

    print('[DONE] 규모+위치 매칭 완료')

if __name__ == '__main__':
    main()