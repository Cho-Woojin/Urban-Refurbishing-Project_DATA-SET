#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 주택재개발 데이터에서 신속통합기획 유무(FastTrack)로 구분하여 생존분석 수행
- 시간 단위: 월(duration_months)
- 출력 디렉터리: outputs/survival_runs/<YYYYMMDD_HHMMSS>/
  · 전체 KM (Fast vs NonFast)
  · 변수별 KM(중앙값 분할) - Fast/NonFast 각각
  · Cox(다변량) with FastTrack dummy + 공공성/통제/추가변수
  · metrics.json, dataset.csv
"""
from __future__ import annotations
import argparse, sys, re, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


COLUMN_SYNONYMS: Dict[str, list[str]] = {
    'id': ['사업번호','사업코드','ID','id','식별자','정비구역코드'],
    'name': ['정비구역명칭','정비구역명','사업명','구역명','사업장명'],
    'exp_flag': ['신속통합기획','신속통합기획여부','신속통합기획_구분'],
    'start_date': [
        '정비구역지정일','지정일','정비구역 지정일','정비구역지정고시일','지정고시일',
        '구역지정최초','구역지정변경(최종)','구역지정_최초','구역지정_변경(최종)'
    ],
    'event_date': [
        '조합설립인가(사업시행자 지정일)','조합설립인가_최초','조합설립인가_변경(최종)','조합설립인가',
        '조합설립인가일','조합설립 인가일'
    ],
    'rental_units': ['임대세대총수','임대세대','임대세대수','(임대)세대수','임대세대 합계'],
    'total_households': ['세대총합계','세대수합계','총세대수','분양세대총수'],
    'area': ['정비구역면적(㎡)','정비구역면적','구역면적','사업면적','면적','면적(㎡)'],
    'owners': ['토지등 소유자 수','토지등소유자수','토지 등 소유자 수'],
    'district': ['자치구','구','시군구','행정구역','자치 단체'],
    'far': ['용적률','용적률(%)','계획용적률','기준용적률'],
    'bcr': ['건폐율','건폐율(%)'],
    # New fields for engineered features
    'gfa': ['건축연면적(㎡)','건축연면적','연면적'],
    'land_residential': ['택지면적(㎡)','택지면적'],
    'road_area': ['도로면적(㎡)','도로면적'],
    'park_area': ['공원면적(㎡)','공원면적'],
    'green_area': ['녹지면적(㎡)','녹지면적'],
    'public_open_area': ['공공공지면적(㎡)','공공공지면적'],
    'school_area': ['학교면적(㎡)','학교면적'],
    'other_area': ['기타면적(㎡)','기타면적'],
    'unit_total': ['분양세대총수','총분양세대수'],
    'unit_small': ['60㎡이하','60이하','60m2이하'],
    'unit_mid': ['60㎡초과~85㎡이하','60초과~85이하','60~85㎡'],
    'unit_large': ['85㎡초과','85초과'],
    'demo_households': ['기존 가구수(멸실량)','멸실가구수','멸실량']
}

POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}
EXP_REGEX = r'(신속통합|신통|선정구역)'


def detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(',','',regex=False)
         .str.replace('%','',regex=False)
         .str.strip(),
        errors='coerce'
    )


def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce', infer_datetime_format=True)


def build_fast_flag(df: pd.DataFrame, exp_flag_col: Optional[str]) -> pd.Series:
    if exp_flag_col:
        flag_mask = df[exp_flag_col].astype(str).str.strip().str.lower().isin(POSITIVE_FLAG_VALUES)
    else:
        flag_mask = pd.Series([False]*len(df), index=df.index)
    if '_row_concat' not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        df['_row_concat'] = df[obj_cols].astype(str).agg(' '.join, axis=1)
    regex_mask = df['_row_concat'].str.contains(EXP_REGEX, regex=True, na=False)
    return (flag_mask | regex_mask)


def map_district_to_region(name: str) -> str:
    if not isinstance(name, str):
        return '기타'
    name = name.strip()
    core = {'종로구','중구','용산구'}
    northeast = {'성동구','광진구','동대문구','중랑구','성북구','강북구','도봉구','노원구'}
    northwest = {'은평구','서대문구','마포구'}
    southwest = {'강서구','양천구','구로구','금천구','영등포구','동작구','관악구'}
    southeast = {'서초구','강남구','송파구','강동구'}
    if name in core: return '도심권'
    if name in northeast: return '동북권'
    if name in northwest: return '서북권'
    if name in southwest: return '서남권'
    if name in southeast: return '동남권'
    return '기타'


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    try:
        ql, qu = np.nanquantile(s.astype(float), [lower, upper])
        return s.clip(ql, qu)
    except Exception:
        return s


def safe_logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    x = p.astype(float)
    x = np.where(np.isnan(x), np.nan, x)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser(description='전체 데이터: 신속통합기획 유무로 구분한 생존분석')
    ap.add_argument('--csv', default='outputs/주택재개발_DATA_full.csv', help='입력 CSV 경로')
    ap.add_argument('--outroot', default='outputs/survival_runs', help='출력 루트 디렉터리')
    ap.add_argument('--cutoff', default=datetime.today().strftime('%Y-%m-%d'), help='컷오프 날짜 YYYY-MM-DD (기본: 오늘)')
    ap.add_argument('--encoding', default='utf-8-sig')
    ap.add_argument('--plot-dpi', type=int, default=140)
    # preprocessing/modeling options
    ap.add_argument('--force-strata-region', action='store_true', help='권역 strata를 조건 충족 여부와 무관하게 강제 적용')
    ap.add_argument('--impute-region-median', action='store_true', help='권역별 중앙값으로 핵심 연속변수 결측 대치')
    ap.add_argument('--winsorize', action='store_true', help='연속변수를 하위/상위 1% 윈저라이즈')
    ap.add_argument('--use-logit-rental', action='store_true', help='임대비율 로짓 변환을 공변량으로 추가')
    ap.add_argument('--drop-low-variance', action='store_true', help='사건/비사건 조건부 분산이 매우 낮은 공변량 제외')
    ap.add_argument('--min-start-year', type=int, default=None, help='정비구역지정 최소 연도 필터(예: 2021)')
    ap.add_argument('--cox-covset', default='full', choices=['full','light','mini'], help='Cox 공변량 세트 선택: full/light/mini')
    ap.add_argument('--no-strata', action='store_true', help='권역 strata 사용 안 함')
    ap.add_argument('--cv-folds', type=int, default=None, help='교차검증 폴드 수(기본: 표본 규모에 따라 3 또는 5)')
    ap.add_argument('--min-penalizer', type=float, default=None, help='벌점 계수 하한(해당 값 이상만 탐색)')
    ap.add_argument('--max-penalizer', type=float, default=None, help='벌점 계수 상한(해당 값 이하만 탐색)')
    args = ap.parse_args()

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
            tried.append(f'{enc}:{e}'); df = None
    if df is None:
        print('[ERROR] 로드 실패', tried); sys.exit(1)

    col_id = detect_column(df, COLUMN_SYNONYMS['id'])
    col_name = detect_column(df, COLUMN_SYNONYMS['name'])
    col_flag = detect_column(df, COLUMN_SYNONYMS['exp_flag'])
    col_start = detect_column(df, COLUMN_SYNONYMS['start_date'])
    col_event = detect_column(df, COLUMN_SYNONYMS['event_date'])
    col_rent = detect_column(df, COLUMN_SYNONYMS['rental_units'])
    col_hh = detect_column(df, COLUMN_SYNONYMS['total_households'])
    col_area = detect_column(df, COLUMN_SYNONYMS['area'])
    col_owners = detect_column(df, COLUMN_SYNONYMS['owners'])
    col_district = detect_column(df, COLUMN_SYNONYMS['district'])
    col_far = detect_column(df, COLUMN_SYNONYMS['far'])
    col_bcr = detect_column(df, COLUMN_SYNONYMS['bcr'])
    # Additional columns for engineered features
    col_gfa = detect_column(df, COLUMN_SYNONYMS.get('gfa', []))
    col_land_res = detect_column(df, COLUMN_SYNONYMS.get('land_residential', []))
    col_road = detect_column(df, COLUMN_SYNONYMS.get('road_area', []))
    col_park = detect_column(df, COLUMN_SYNONYMS.get('park_area', []))
    col_green = detect_column(df, COLUMN_SYNONYMS.get('green_area', []))
    col_public_open = detect_column(df, COLUMN_SYNONYMS.get('public_open_area', []))
    col_school = detect_column(df, COLUMN_SYNONYMS.get('school_area', []))
    col_other = detect_column(df, COLUMN_SYNONYMS.get('other_area', []))
    col_unit_total = detect_column(df, COLUMN_SYNONYMS.get('unit_total', []))
    col_unit_small = detect_column(df, COLUMN_SYNONYMS.get('unit_small', []))
    col_unit_mid = detect_column(df, COLUMN_SYNONYMS.get('unit_mid', []))
    col_unit_large = detect_column(df, COLUMN_SYNONYMS.get('unit_large', []))
    col_demo = detect_column(df, COLUMN_SYNONYMS.get('demo_households', []))

    if not col_start:
        print('[FATAL] 시작일(정비구역지정일) 컬럼 탐지 실패'); sys.exit(1)

    start = parse_date(df[col_start])
    event = parse_date(df[col_event]) if col_event else pd.Series([pd.NaT]*len(df), index=df.index)
    cutoff = pd.to_datetime(args.cutoff)

    valid = start.notna()
    df = df[valid].copy(); start = start[valid]
    if not event.empty: event = event[valid]

    event_obs = event.notna() & (event <= cutoff)
    end_dates = np.where(event_obs, event, cutoff)
    end_dates = pd.to_datetime(end_dates)
    duration_days = (end_dates - start).dt.days
    ok = duration_days >= 0
    df = df[ok].copy(); duration_days = duration_days[ok]; event_obs = event_obs[ok]; start = start[ok]
    if not event.empty: event = event[ok]

    DAYS_PER_MONTH = 30.44
    duration_months = duration_days / DAYS_PER_MONTH

    rent = coerce_numeric(df[col_rent]) if col_rent else pd.Series([np.nan]*len(df), index=df.index)
    hh = coerce_numeric(df[col_hh]) if col_hh else pd.Series([np.nan]*len(df), index=df.index)
    rental_ratio = np.where((rent.notna() & hh.notna() & (hh>0)), (rent / hh), np.nan)

    area = coerce_numeric(df[col_area]) if col_area else pd.Series([np.nan]*len(df), index=df.index)
    owners = coerce_numeric(df[col_owners]) if col_owners else pd.Series([np.nan]*len(df), index=df.index)

    far_raw = coerce_numeric(df[col_far]) if col_far else pd.Series([np.nan]*len(df), index=df.index)
    bcr_raw = coerce_numeric(df[col_bcr]) if col_bcr else pd.Series([np.nan]*len(df), index=df.index)
    far_ratio = np.where(pd.notna(far_raw) & (far_raw > 1.0), far_raw/100.0, far_raw)
    bcr_ratio = np.where(pd.notna(bcr_raw) & (bcr_raw > 1.0), bcr_raw/100.0, bcr_raw)
    district = df[col_district].astype(str) if col_district else pd.Series([np.nan]*len(df), index=df.index)

    # Extra raw fields
    gfa = coerce_numeric(df[col_gfa]) if col_gfa else pd.Series([np.nan]*len(df), index=df.index)
    land_res = coerce_numeric(df[col_land_res]) if col_land_res else pd.Series([np.nan]*len(df), index=df.index)
    road_area = coerce_numeric(df[col_road]) if col_road else pd.Series([np.nan]*len(df), index=df.index)
    park_area = coerce_numeric(df[col_park]) if col_park else pd.Series([np.nan]*len(df), index=df.index)
    green_area = coerce_numeric(df[col_green]) if col_green else pd.Series([np.nan]*len(df), index=df.index)
    public_open_area = coerce_numeric(df[col_public_open]) if col_public_open else pd.Series([np.nan]*len(df), index=df.index)
    school_area = coerce_numeric(df[col_school]) if col_school else pd.Series([np.nan]*len(df), index=df.index)
    other_area = coerce_numeric(df[col_other]) if col_other else pd.Series([np.nan]*len(df), index=df.index)
    unit_total = coerce_numeric(df[col_unit_total]) if col_unit_total else pd.Series([np.nan]*len(df), index=df.index)
    unit_small = coerce_numeric(df[col_unit_small]) if col_unit_small else pd.Series([np.nan]*len(df), index=df.index)
    unit_mid = coerce_numeric(df[col_unit_mid]) if col_unit_mid else pd.Series([np.nan]*len(df), index=df.index)
    unit_large = coerce_numeric(df[col_unit_large]) if col_unit_large else pd.Series([np.nan]*len(df), index=df.index)
    demo_households = coerce_numeric(df[col_demo]) if col_demo else pd.Series([np.nan]*len(df), index=df.index)

    fast_flag = build_fast_flag(df, col_flag)
    fast_label = np.where(fast_flag, 'FastTrack', 'NonFast')

    out_df = pd.DataFrame({
        'id': df[col_id] if col_id else df.index,
        'name': df[col_name] if col_name else np.nan,
        'district': district,
        'start_date': start.dt.strftime('%Y-%m-%d'),
        'event_date': event.dt.strftime('%Y-%m-%d') if not event.empty else np.nan,
        'duration_days': duration_days,
        'duration_months': duration_months,
        'event_observed': event_obs.astype(int),
        'fast_flag': fast_flag.astype(int),
        'fast_label': fast_label,
        'rental_ratio': rental_ratio,
        'area_m2': area,
        'owners': owners,
        'households_total': hh,
        'far': far_raw,
        'bcr': bcr_raw,
        'far_ratio': far_ratio,
        'bcr_ratio': bcr_ratio,
        # engineered inputs source
        'gfa_m2': gfa,
        'land_res_m2': land_res,
        'road_area_m2': road_area,
        'park_area_m2': park_area,
        'green_area_m2': green_area,
        'public_open_area_m2': public_open_area,
        'school_area_m2': school_area,
        'other_area_m2': other_area,
        'unit_total': unit_total,
        'unit_small': unit_small,
        'unit_mid': unit_mid,
        'unit_large': unit_large,
        'demo_households': demo_households,
    })
    # Filter by min start year if provided
    if args.min_start_year is not None:
        try:
            sy = pd.to_datetime(out_df['start_date']).dt.year
            mask = sy >= int(args.min_start_year)
            out_df = out_df[mask].copy()
            # also subset d/e/labels for KM reference variables below by recomputing directly from out_df
        except Exception:
            pass

    def log1p_std(x: pd.Series) -> pd.Series:
        x1 = np.log1p(x.astype(float))
        m, s = np.nanmean(x1), np.nanstd(x1)
        return (x1 - m) / (s if s and s>0 else 1.0)

    out_df['rental_ratio_fill'] = out_df['rental_ratio']
    over1 = out_df['rental_ratio_fill'] > 1.0
    out_df.loc[over1, 'rental_ratio_fill'] = out_df.loc[over1, 'rental_ratio_fill'] / 100.0
    out_df['std_rental_ratio'] = (out_df['rental_ratio_fill'] - np.nanmean(out_df['rental_ratio_fill'])) / (np.nanstd(out_df['rental_ratio_fill']) or 1.0)
    # 규모 원변수는 '면적'만 유지. 소유자수/세대수는 면적 대비 밀도로 대체
    # 안전한 밀도 계산(면적>0에서만 계산)
    area_vals = out_df['area_m2'].astype(float).values
    owners_vals = out_df['owners'].astype(float).values
    hh_vals = out_df['households_total'].astype(float).values
    owners_per_area = np.divide(owners_vals, area_vals, out=np.full_like(area_vals, np.nan, dtype=float), where=area_vals>0)
    households_density = np.divide(hh_vals, area_vals, out=np.full_like(area_vals, np.nan, dtype=float), where=area_vals>0)
    out_df['owners_per_area'] = owners_per_area
    out_df['households_density'] = households_density
    out_df['logstd_area'] = log1p_std(out_df['area_m2'])
    out_df['std_owners_per_area'] = (out_df['owners_per_area'] - np.nanmean(out_df['owners_per_area'])) / (np.nanstd(out_df['owners_per_area']) or 1.0)
    out_df['std_households_density'] = (out_df['households_density'] - np.nanmean(out_df['households_density'])) / (np.nanstd(out_df['households_density']) or 1.0)
    out_df['std_far'] = (out_df['far_ratio'] - np.nanmean(out_df['far_ratio'])) / (np.nanstd(out_df['far_ratio']) or 1.0)
    out_df['std_bcr'] = (out_df['bcr_ratio'] - np.nanmean(out_df['bcr_ratio'])) / (np.nanstd(out_df['bcr_ratio']) or 1.0)
    # realized FAR: prefer land_residential if available else area
    land_for_far = np.where(out_df['land_res_m2'].notna() & (out_df['land_res_m2']>0), out_df['land_res_m2'], out_df['area_m2'])
    realized_far = np.divide(out_df['gfa_m2'].astype(float), land_for_far.astype(float), out=np.full_like(owners_vals, np.nan, dtype=float), where=pd.notna(land_for_far.astype(float)) & (land_for_far.astype(float)>0))
    out_df['realized_far'] = realized_far
    out_df['std_realized_far'] = (out_df['realized_far'] - np.nanmean(out_df['realized_far'])) / (np.nanstd(out_df['realized_far']) or 1.0)
    # land-use composition ratios (per total area)
    def safe_ratio(num):
        return np.divide(num.astype(float), area_vals, out=np.full_like(area_vals, np.nan, dtype=float), where=area_vals>0)
    out_df['road_ratio'] = safe_ratio(out_df['road_area_m2'])
    out_df['park_ratio'] = safe_ratio(out_df['park_area_m2'])
    out_df['green_ratio'] = safe_ratio(out_df['green_area_m2'])
    out_df['public_open_ratio'] = safe_ratio(out_df['public_open_area_m2'])
    out_df['school_ratio'] = safe_ratio(out_df['school_area_m2'])
    out_df['other_ratio'] = safe_ratio(out_df['other_area_m2'])
    out_df['open_space_ratio'] = out_df[['park_ratio','green_ratio','public_open_ratio']].sum(axis=1, min_count=1)
    out_df['open_space_ratio_std'] = (out_df['open_space_ratio'] - np.nanmean(out_df['open_space_ratio'])) / (np.nanstd(out_df['open_space_ratio']) or 1.0)
    # unit size shares & diversity
    unit_base = np.where(out_df['unit_total'].notna() & (out_df['unit_total']>0), out_df['unit_total'], out_df['households_total'])
    def safe_share(cnt):
        return np.divide(cnt.astype(float), unit_base.astype(float), out=np.full_like(unit_base.astype(float), np.nan, dtype=float), where=pd.notna(unit_base.astype(float)) & (unit_base.astype(float)>0))
    out_df['share_small'] = safe_share(out_df['unit_small'])
    out_df['share_mid'] = safe_share(out_df['unit_mid'])
    out_df['share_large'] = safe_share(out_df['unit_large'])
    # Shannon entropy of size composition (normalized by log(k))
    def shannon_entropy(row):
        p = np.array([row.get('share_small', np.nan), row.get('share_mid', np.nan), row.get('share_large', np.nan)], dtype=float)
        p = p[~np.isnan(p)]
        if p.size == 0:
            return np.nan
        # renormalize to sum 1 if partly missing
        s = p.sum()
        if s <= 0:
            return np.nan
        p = p / s
        ent = -np.sum(p * np.log(p))
        return ent / np.log(len(p))
    out_df['unit_diversity'] = out_df.apply(shannon_entropy, axis=1)
    out_df['unit_diversity_std'] = (out_df['unit_diversity'] - np.nanmean(out_df['unit_diversity'])) / (np.nanstd(out_df['unit_diversity']) or 1.0)
    # demolition density
    out_df['demolition_density'] = np.divide(out_df['demo_households'].astype(float), area_vals, out=np.full_like(area_vals, np.nan, dtype=float), where=area_vals>0)
    out_df['demolition_density_std'] = (out_df['demolition_density'] - np.nanmean(out_df['demolition_density'])) / (np.nanstd(out_df['demolition_density']) or 1.0)
    # start year (calendar time effect)
    try:
        out_df['start_year'] = pd.to_datetime(out_df['start_date']).dt.year.astype(float)
        out_df['start_year_std'] = (out_df['start_year'] - np.nanmean(out_df['start_year'])) / (np.nanstd(out_df['start_year']) or 1.0)
    except Exception:
        out_df['start_year_std'] = np.nan
    # optional preprocessing
    core_cont = ['rental_ratio_fill','area_m2','owners_per_area','households_density','far_ratio','bcr_ratio',
                 'realized_far','open_space_ratio','unit_diversity','demolition_density']
    core_cont = [c for c in core_cont if c in out_df.columns]
    if getattr(args, 'impute_region_median', False):
        out_df['region'] = out_df['district'].apply(map_district_to_region)
        for c in core_cont:
            out_df[c] = out_df.groupby('region')[c].transform(lambda x: x.fillna(x.median()))
    if getattr(args, 'winsorize', False):
        for c in core_cont:
            out_df[c] = winsorize_series(out_df[c])
    if getattr(args, 'use_logit_rental', False):
        out_df['rental_ratio_logit'] = safe_logit(out_df['rental_ratio_fill'])
        rr = out_df['rental_ratio_logit']
        out_df['std_rental_ratio_logit'] = (rr - np.nanmean(rr)) / (np.nanstd(rr) or 1.0)

    # 비선형(간단): 제곱항 추가(표준화 후)
    out_df['logstd_area_sq'] = out_df['logstd_area']**2
    out_df['std_owners_per_area_sq'] = out_df['std_owners_per_area']**2
    out_df['std_households_density_sq'] = out_df['std_households_density']**2

    # 상호작용: fast_flag × (면적, 밀도들)
    out_df['fast_x_area'] = out_df['fast_flag'] * out_df['logstd_area']
    out_df['fast_x_owners_per_area'] = out_df['fast_flag'] * out_df['std_owners_per_area']
    out_df['fast_x_households_density'] = out_df['fast_flag'] * out_df['std_households_density']

    # 권역(strata) 매핑
    out_df['region'] = out_df['district'].apply(map_district_to_region)

    # outdir with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(args.outroot) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_path = outdir / f'dataset_{timestamp}.csv'
    out_df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    print('[SAVE] dataset ->', dataset_path)

    # Plots & Cox
    try:
        import matplotlib.pyplot as plt
        try:
            import matplotlib
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test

        # Overall KM: Fast vs NonFast
        d = out_df['duration_months']; e = out_df['event_observed']
        labels = out_df['fast_label']
        fig, ax = plt.subplots()
        km = KaplanMeierFitter()
        for g in ['FastTrack','NonFast']:
            idx = labels == g
            if idx.sum() < 5: continue
            km.fit(durations=d[idx], event_observed=e[idx], label=g)
            km.plot(ax=ax)
        # logrank Fast vs NonFast
        try:
            idx_f = labels=='FastTrack'; idx_n = labels=='NonFast'
            lr = logrank_test(d[idx_f], d[idx_n], e[idx_f], e[idx_n])
            p_fast = float(lr.p_value)
        except Exception:
            p_fast = np.nan
        ax.set_title(f'KM: Fast vs NonFast (p={p_fast:.3g})')
        ax.set_xlabel('months'); ax.set_ylabel('survival'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(outdir / f'km_fast_vs_nonfast_{timestamp}.png', dpi=args.plot_dpi); plt.close()
        print('[SAVE] overall KM')
        # save overall summary
        pd.DataFrame([{'metric':'km_fast_vs_nonfast_p','value':p_fast}]).to_csv(outdir / f'summary_overall_{timestamp}.csv', index=False, encoding='utf-8-sig')

        # Variable-wise KM (median split), within Fast and NonFast
        def km_by_median_in_group(var: pd.Series, label: str, group: str, fname_prefix: str):
            s = pd.to_numeric(var, errors='coerce')
            mask = s.notna() & d.notna() & e.notna() & (labels==group)
            if mask.sum() < 10:
                return None
            s = s[mask]; dd = d[mask]; ee = e[mask]
            med = float(np.nanmedian(s))
            grp = np.where(s <= med, f'Low (≤{med:.3g})', f'High (>{med:.3g})')
            fig, ax = plt.subplots(); km = KaplanMeierFitter()
            for g in np.unique(grp):
                idx = grp == g
                km.fit(dd[idx], ee[idx], label=g)
                km.plot(ax=ax)
            # logrank
            try:
                lr = logrank_test(dd[grp==np.unique(grp)[0]], dd[grp==np.unique(grp)[1]], ee[grp==np.unique(grp)[0]], ee[grp==np.unique(grp)[1]])
                pval = float(lr.p_value)
            except Exception:
                pval = np.nan
            ax.set_title(f'{group} | {label} median split (p={pval:.3g})')
            ax.set_xlabel('months'); ax.set_ylabel('survival'); ax.grid(True, alpha=0.3)
            path = outdir / f'{fname_prefix}_{group}_{timestamp}.png'
            plt.tight_layout(); plt.savefig(path, dpi=args.plot_dpi); plt.close()
            print(f'[SAVE] {group} {label} ->', path)
            return pval

        var_list = [
            ('rental_ratio_fill','임대세대비율','km_rental'),
            ('area_m2','정비구역면적','km_area'),
            ('owners_per_area','소유자밀도(명/㎡)','km_owners_density'),
            ('households_density','세대밀도(세대/㎡)','km_households_density'),
            ('far','용적률(원자료)','km_far'),
            ('bcr','건폐율(원자료)','km_bcr'),
            ('realized_far','실현용적률(연면적/면적)','km_realized_far'),
            ('open_space_ratio','개방·녹지면적비','km_open_space'),
        ]
        km_rows = []
        for col,label,prefix in var_list:
            if col not in out_df.columns: continue
            p1 = km_by_median_in_group(out_df[col], label, 'FastTrack', prefix)
            p2 = km_by_median_in_group(out_df[col], label, 'NonFast', prefix)
            km_rows.append({'variable':col,'label':label,'group':'FastTrack','km_p':p1})
            km_rows.append({'variable':col,'label':label,'group':'NonFast','km_p':p2})
        if km_rows:
            pd.DataFrame(km_rows).to_csv(outdir / f'summary_km_{timestamp}.csv', index=False, encoding='utf-8-sig')

        # Cox: include fast_flag dummy + covariates
        covset = getattr(args, 'cox_covset', 'full')
        if covset == 'mini':
            covs = [
                'fast_flag',
                'std_rental_ratio',
                'logstd_area',
            ]
        elif covset == 'light':
            covs = [
                'fast_flag',
                'std_rental_ratio',
                'logstd_area',
                'std_realized_far',
                'std_owners_per_area',
                'open_space_ratio_std',
                'start_year_std',
            ]
        else:
            covs = [
                'fast_flag','std_rental_ratio','logstd_area',
                'std_owners_per_area','std_households_density',
                'std_far','std_bcr',
                # nonlinear terms
                'logstd_area_sq','std_owners_per_area_sq','std_households_density_sq',
                # interactions
                'fast_x_area','fast_x_owners_per_area','fast_x_households_density',
                # engineered additions
                'std_realized_far','open_space_ratio_std','unit_diversity_std','demolition_density_std','start_year_std',
            ]
        cox_df_full = out_df[['duration_months','event_observed','district','region'] + covs].dropna(subset=['duration_months','event_observed']+covs)
        # region strata(권역) 조건: 2~6개 권역, 각 n>=10 또는 강제 옵션
        use_strata = False
        try:
            vc = cox_df_full['region'].dropna().value_counts()
            if 2 <= len(vc) <= 6 and (vc.min() >= 10):
                use_strata = True
        except Exception:
            use_strata = False
        if getattr(args, 'force_strata_region', False):
            use_strata = True
        if getattr(args, 'no_strata', False):
            use_strata = False
        # 항상 district는 제거(문자열)
        base_fit = cox_df_full.drop(columns=['district'], errors='ignore')
        df_fit = base_fit.dropna(subset=['region']) if use_strata else base_fit.drop(columns=['region'], errors='ignore')
        print(f"[COX] N={len(df_fit)} with covs={len(covs)}; strata_by_region={use_strata}")

        # Penalized Cox CV
        from lifelines.utils import concordance_index
        penalizers = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
        if args.min_penalizer is not None:
            penalizers = [p for p in penalizers if p >= args.min_penalizer]
        if args.max_penalizer is not None:
            penalizers = [p for p in penalizers if p <= args.max_penalizer]
        l1_ratios = [0.0, 0.2, 0.5, 0.8]
        X_cols = covs
        # drop rows with NA in used columns
        cv_df = df_fit[['duration_months','event_observed'] + X_cols + (['region'] if use_strata else [])].dropna()
        # adaptive CV folds
        n = len(cv_df)
        k = int(args.cv_folds) if getattr(args, 'cv_folds', None) else (3 if n < 60 else 5)
        k = max(2, min(k, n))
        # EPV warning
        try:
            events = int(cv_df['event_observed'].sum())
            epv = events / max(1, len(X_cols))
            if epv < 5:
                print(f"[WARN] 낮은 EPV: events={events}, covariates={len(X_cols)}, EPV={epv:.2f}. 변수 축소 또는 벌점 강화 권장.")
        except Exception:
            pass
        # low-variance drop option
        if getattr(args, 'drop_low_variance', False) and len(cv_df) > 0:
            ev = cv_df['event_observed'].astype(bool)
            drop_cols = []
            for c in list(X_cols):
                try:
                    v1 = np.nanvar(cv_df.loc[ev, c].astype(float))
                    v0 = np.nanvar(cv_df.loc[~ev, c].astype(float))
                    if (v1 < 1e-6) or (v0 < 1e-6):
                        drop_cols.append(c)
                except Exception:
                    pass
            if drop_cols:
                X_cols = [c for c in X_cols if c not in drop_cols]
                cv_df = cv_df[['duration_months','event_observed'] + X_cols + (['region'] if use_strata else [])]
        # 간단 KFold 구현
        rng = np.random.default_rng(42)
        idx = np.arange(n); rng.shuffle(idx)
        folds = np.array_split(idx, k)
        cv_rows = []
        for pen in penalizers:
            for l1 in l1_ratios:
                c_index_list = []
                for i in range(k):
                    test_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                    tr = cv_df.iloc[train_idx]
                    te = cv_df.iloc[test_idx]
                    cph_cv = CoxPHFitter(penalizer=pen, l1_ratio=l1)
                    try:
                        if use_strata:
                            cph_cv.fit(tr, duration_col='duration_months', event_col='event_observed', strata=['region'])
                        else:
                            cph_cv.fit(tr, duration_col='duration_months', event_col='event_observed')
                        # test concordance
                        risk = cph_cv.predict_partial_hazard(te).values.ravel()
                        cidx = float(concordance_index(te['duration_months'], -risk, te['event_observed']))
                    except Exception:
                        cidx = np.nan
                    c_index_list.append(cidx)
                cv_rows.append({'penalizer':pen,'l1_ratio':l1,'c_index_mean':np.nanmean(c_index_list),'c_index_std':np.nanstd(c_index_list)})
        cv_dfres = pd.DataFrame(cv_rows).sort_values(['c_index_mean','penalizer'], ascending=[False, True])
        cv_path = outdir / f'cv_results_{timestamp}.csv'; cv_dfres.to_csv(cv_path, index=False, encoding='utf-8-sig')
        print('[SAVE] CV results ->', cv_path)
        best = cv_dfres.iloc[0].to_dict()
        print(f"[CV] best penalizer={best['penalizer']}, l1_ratio={best['l1_ratio']}, c-index={best['c_index_mean']:.3f}")

        # Fit final model with best penalization
        cph = CoxPHFitter(penalizer=float(best['penalizer']), l1_ratio=float(best['l1_ratio']))
        if use_strata:
            cph.fit(df_fit, duration_col='duration_months', event_col='event_observed', strata=['region'])
        else:
            cph.fit(df_fit, duration_col='duration_months', event_col='event_observed')
        try:
            summary = cph.summary.reset_index().rename(columns={'index':'variable'})
        except Exception:
            summary = cph.summary.copy()
        if 'exp(coef)' not in summary.columns and 'coef' in summary.columns:
            summary['HR'] = np.exp(summary['coef'])
        sum_path = outdir / f'cox_summary_{timestamp}.csv'
        summary.to_csv(sum_path, index=False, encoding='utf-8-sig')
        print('[SAVE] cox summary ->', sum_path)
        # brief cox (robust to lifelines/pandas version)
        if 'variable' not in summary.columns:
            if 'covariate' in summary.columns:
                summary['variable'] = summary['covariate']
            else:
                summary = summary.reset_index().rename(columns={'index':'variable'})
        brief_cols = ['variable','coef']
        if 'exp(coef)' in summary.columns:
            brief_cols.append('exp(coef)')
        elif 'HR' in summary.columns:
            brief_cols.append('HR')
        if 'p' in summary.columns:
            brief_cols.append('p')
        summary.loc[:, brief_cols].to_csv(outdir / f'summary_cox_{timestamp}.csv', index=False, encoding='utf-8-sig')

        # Forest plot
        try:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            plot_df = summary.copy()
            hr = plot_df['exp(coef)'] if 'exp(coef)' in plot_df.columns else plot_df.get('HR', np.exp(plot_df['coef']))
            se = plot_df.get('se(coef)', None)
            if 'exp(coef) lower 95%' in plot_df.columns:
                low = plot_df['exp(coef) lower 95%']
                high = plot_df['exp(coef) upper 95%']
            elif se is not None:
                low = np.exp(plot_df['coef'] - 1.96*se); high = np.exp(plot_df['coef'] + 1.96*se)
            else:
                low = np.nan*hr; high = np.nan*hr
            names = plot_df.get('variable', plot_df.index)
            y = np.arange(len(hr))[::-1]
            ax.errorbar(hr, y, xerr=[hr-low, high-hr], fmt='o', color='C0', ecolor='gray', capsize=3)
            ax.axvline(1.0, color='red', linestyle='--', alpha=0.6)
            ax.set_yticks(y); ax.set_yticklabels(names)
            ax.set_xlabel('Hazard Ratio (HR)')
            ax.set_title('Cox HR (95% CI)')
            plt.tight_layout(); plt.savefig(outdir / f'cox_forest_{timestamp}.png', dpi=args.plot_dpi); plt.close()
            print('[SAVE] forest')
        except Exception as e:
            print('[WARN] forest 실패:', e)

        # Metrics
        metrics = {}
        try:
            metrics['concordance_index'] = float(getattr(cph, 'concordance_index_', np.nan))
        except Exception: pass
        try:
            metrics['AIC_partial'] = float(getattr(cph, 'AIC_partial_', np.nan))
        except Exception: pass
        try:
            lr = cph.log_likelihood_ratio_test()
            metrics['lr_test_p'] = float(getattr(lr, 'p_value', np.nan))
            metrics['lr_test_stat'] = float(getattr(lr, 'test_statistic', np.nan))
        except Exception: pass
        with open(outdir / f'metrics_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print('[SAVE] metrics')

        # District summaries (overall and by fast_label)
        try:
            dist_overall = out_df.groupby('district', dropna=False).agg(
                n=('id','count'),
                events=('event_observed','sum'),
                median_months=('duration_months','median')
            ).reset_index().sort_values('n', ascending=False)
            dist_overall.to_csv(outdir / f'district_summary_overall_{timestamp}.csv', index=False, encoding='utf-8-sig')
            dist_by_fast = out_df.groupby(['district','fast_label'], dropna=False).agg(
                n=('id','count'),
                events=('event_observed','sum'),
                median_months=('duration_months','median')
            ).reset_index().sort_values(['district','fast_label'])
            dist_by_fast.to_csv(outdir / f'district_summary_by_fast_{timestamp}.csv', index=False, encoding='utf-8-sig')
            print('[SAVE] district summaries')
        except Exception as e:
            print('[WARN] district summary 실패:', e)

        # RMST export
        try:
            from lifelines import KaplanMeierFitter
            tau = float(np.nanpercentile(out_df['duration_months'], 80))
            rmst_rows = []
            for lab in ['FastTrack','NonFast']:
                m = (out_df['fast_label'] == lab)
                kmfit = KaplanMeierFitter().fit(out_df.loc[m, 'duration_months'], out_df.loc[m, 'event_observed'], label=lab)
                t = kmfit.survival_function_.index.values
                s = kmfit.survival_function_[lab].values
                t = np.clip(t, 0, tau)
                rmst = float(np.trapz(y=s, x=t))
                rmst_rows.append({'group': lab, 'tau': tau, 'rmst': rmst})
            pd.DataFrame(rmst_rows).to_csv(outdir / f'rmst_{timestamp}.csv', index=False, encoding='utf-8-sig')
            print('[SAVE] rmst')
        except Exception as e:
            print('[WARN] rmst 실패:', e)

        # AFT(Weibull) exploratory
        try:
            from lifelines import WeibullAFTFitter
            aft = WeibullAFTFitter()
            aft.fit(df_fit, duration_col='duration_months', event_col='event_observed')
            aft.summary.to_csv(outdir / f'aft_weibull_summary_{timestamp}.csv', index=True, encoding='utf-8-sig')
            print('[SAVE] AFT(Weibull) summary')
        except Exception as e:
            print('[WARN] AFT 실패:', e)

    except ImportError:
        print('[WARN] lifelines/matplotlib 미설치: 시각화/모형 생략')

    print('[DONE] survival by fast-flag complete ->', outdir)


if __name__ == '__main__':
    main()
