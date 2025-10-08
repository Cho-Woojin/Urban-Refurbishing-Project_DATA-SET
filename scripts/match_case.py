#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case-Control 매칭 스크립트
-------------------------------------------------
목표:
- 실험군(Experimental): 신속통합기획 주택정비형 재개발 + 정비계획 확정된 약 16건 추출
- 대조군(Control): 2018~2021 착수(또는 정비구역 지정) 일반 주택재개발 사례 중
  (자치구/생활권, 면적, 조합원수, 지정연도) 유사 후보에서 매칭 (총 약 150건 목표)

매칭 기준:
1) 동일 자치구 (최우선) 또는 동일 생활권 (부 차선) — 둘 다 다르면 후보 제외
2) 면적(구역면적) ±20% 이내
3) 조합원수(또는 세대/추정 조합원) ±20% 이내
4) 정비구역 지정 연도 차이 ±2년 이내

출력:
- matches CSV: 실험군 1건당 상위 K(controls_per_experiment)개의 대조군 후보 + 유사도 점수
- summary  CSV: 매 실험군별 확보된 대조군 수, 확장(완화) 여부
- diagnostics(선택): 조건 위반 사유 요약

사용 예시:
python scripts/match_case.py \
  --csv "outputs/주택재개발_DATA_full.csv" \
  --output-dir "outputs/matching" \
  --controls-per-experiment 10

필요 패키지: pandas, numpy
"""
from __future__ import annotations
import argparse, sys, re, math, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# --------------------------------------------------
# 설정 기본값
# --------------------------------------------------
DEFAULT_EXP_KEYWORDS = ["신속통합기획", "주택정비형 재개발"]  # (레거시 옵션) 고정 플래그 기반으로 대체됨
DEFAULT_PLAN_STATUS_KEYWORDS = ["정비계획 확정", "정비계획확정", "정비계획수립", "정비계획고시"]  # (레거시) 진행단계_구간 사용 시 무시 가능
DEFAULT_CONTROL_YEAR_RANGE = (2018, 2021)  # inclusive
DEFAULT_AREA_TOL = 0.20
DEFAULT_MEMBER_TOL = 0.20
DEFAULT_YEAR_TOL = 2
DEFAULT_CONTROLS_PER_EXP = 10
DEFAULT_MIN_CONTROLS_PER_EXP = 5

# 점수 가중치 (합이 1일 필요는 없음)
WEIGHT_DISTRICT = 0.0   # 동일 자치구 패널티 0
WEIGHT_LIVINGZONE = 0.15  # 자치구 다르지만 생활권 동일시 패널티
# 새로운 우선순위: 1) 규모(면적/조합원) 2) 위치 & 연도 (동일 비중)
WEIGHT_AREA = 0.35
WEIGHT_MEMBER = 0.35
WEIGHT_LOCATION = 0.15
WEIGHT_YEAR = 0.15

# 완화 단계 (매칭 부족 시 차례대로 적용)
RELAX_STEPS = [
    {"area_tol_mul": 1.15, "member_tol_mul": 1.15, "year_tol_add": 0, "dist_km_mul": 1.2},
    {"area_tol_mul": 1.30, "member_tol_mul": 1.30, "year_tol_add": 1, "dist_km_mul": 1.6},
    {"area_tol_mul": 1.50, "member_tol_mul": 1.50, "year_tol_add": 2, "dist_km_mul": 2.2},
    {"area_tol_mul": 1.70, "member_tol_mul": 1.70, "year_tol_add": 3, "dist_km_mul": 3.0},
]

# 실험군 플래그 컬럼 값에서 True 로 간주할 문자열
POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}

# 실험군 패턴 기본 (dataset의 '신통', '선정구역' 등 변형 대응)
DEFAULT_EXP_REGEX_PATTERNS = [r"신속통합", r"신통", r"선정구역", r"신속 통합", r"신속.*기획"]

# 계획확정 단계로 간주할 대체 상태 (데이터셋의 진행단계 값 매핑)
DEFAULT_PLAN_STATUS_ACCEPT = [
    "정비구역지정",   # 정비계획 고시 후 지정 단계
    "사업시행인가",   # 계획 확정 이후 진행됨
    "조합설립인가",   # 일부 분석에서 확정 이후로 간주 가능(옵션)
    "관리처분인가"    # 고도 단계지만 넓게 포함할 수 있음 (옵션)
]

# 후보 컬럼 사전 (유연한 매핑)
COLUMN_SYNONYMS = {
    "project_name": ["사업명", "정비구역명", "구역명", "사업장명"],
    "project_type": ["사업유형", "사업구분", "정비유형", "정비방식"],
    "exp_flag":     ["신속통합기획", "신속통합기획여부"],
    "plan_status":  ["사업단계", "진행단계", "진행단계_구간", "현황", "추진단계", "상태"],
    "district":     ["자치구", "구", "시군구명", "시군구", "구청"],
    "living_zone":  ["생활권", "생활권역", "생활권구분"],
    # 면적: 데이터셋에 '정비구역면적(㎡)' 형태 존재 -> 추가
    "area":         ["구역면적", "면적", "사업면적", "면적(㎡)", "면적_m2", "정비구역면적(㎡)", "정비구역면적"],
    # 조합원/세대 관련: '토지등 소유자 수', '분양세대총수', '세대총합계' 등 대체 가능
    "members":      ["조합원수", "추정조합원수", "조합원 수", "세대수", "추정세대수", "토지등 소유자 수", "분양세대총수", "세대총합계"],
    "designation_date": ["정비구역지정일", "지정일", "정비구역 지정일", "정비구역지정고시일", "지정고시일"],
    "designation_year": ["정비구역지정연도", "지정연도", "지정년도"],
    "id":           ["사업코드", "코드", "ID", "식별자"],
}

DATE_COL_PREFER = ["designation_date", "designation_year"]  # date 우선
YEAR_FALLBACK_CANDIDATES = [
    "지정연도","지정년도","사업시작연도","착공연도","추진연도","인가연도",
    # 진행단계 관련 날짜 필드: 조합설립인가, 사업시행인가, 관리처분계획인가 등
    "조합설립인가(사업시행자 지정일)", "사업시행인가_최초", "사업시행인가_변경(최종)",
    "관리처분계획인가_최초", "관리처분계획인가_변경(최종)", "착공"
]

# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """COLUMN_SYNONYMS 기반으로 실제 컬럼 매핑 사전을 생성."""
    mapping = {}
    cols_lower_map = {c.lower(): c for c in df.columns}
    for canon, syns in COLUMN_SYNONYMS.items():
        for s in syns:
            if s in df.columns:
                mapping[canon] = s
                break
            if s.lower() in cols_lower_map:
                mapping[canon] = cols_lower_map[s.lower()]
                break
    return mapping


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def extract_year_from_date(series: pd.Series) -> pd.Series:
    """YYYY, YYYY-MM-DD, YYYY.MM.DD 등에서 연도만 추출."""
    def _year(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        m = re.search(r'(19|20)\d{2}', s)
        if m:
            return int(m.group())
        return np.nan
    return series.map(_year)


def filter_experimental(
    df: pd.DataFrame,
    mapping: Dict[str,str],
    exp_keywords: List[str],
    plan_status_keywords: List[str],
    require_all_keywords: bool = True,
    flag_positive_values: Optional[set] = None,
    relax_plan_status: bool = False,
    scan_all_columns: bool = True,
    exp_regex_patterns: Optional[List[str]] = None,
    plan_status_accept: Optional[List[str]] = None,
    fixed_flag_value_set: Optional[set] = None,
) -> pd.DataFrame:
    """실험군 필터링 고도화.
    로직 계층:
      1) exp_flag 컬럼이 있고 값이 긍정( POSITIVE_FLAG_VALUES )이면 기본 포함
      2) project_type 컬럼 또는 (scan_all_columns=True) 인 경우 전체 문자열 연결 후 키워드 검색
      3) plan_status 키워드 (relax_plan_status=True 면 조건 실패 시 무시)
    """
    df_work = df.copy()
    type_col = mapping.get("project_type")
    flag_col = mapping.get("exp_flag")
    status_col = mapping.get("plan_status")

    if flag_positive_values is None:
        flag_positive_values = POSITIVE_FLAG_VALUES

    def norm(s):
        return str(s).strip().replace(' ', '') if not pd.isna(s) else ''

    df_work['_type_norm'] = df_work[type_col].map(norm) if type_col else ''
    df_work['_flag_norm'] = df_work[flag_col].map(norm) if flag_col else ''
    df_work['_status_norm'] = df_work[status_col].map(norm) if status_col else ''

    # 전체 문자열 결합 (필요 시 1회 생성)
    if scan_all_columns and '_row_concat' not in df_work.columns:
        obj_cols = [c for c in df_work.columns if df_work[c].dtype == object and not c.startswith('_')]
        df_work['_row_concat'] = df_work[obj_cols].astype(str).agg(' '.join, axis=1).str.replace(' ', '')

    # (신규) 고정 값 세트 기반 실험군 선택 (신속통합기획 컬럼 값이 명시된 세트에 속하면 실험군)
    if fixed_flag_value_set and flag_col:
        flag_mask = df[flag_col].isin(fixed_flag_value_set)
    else:
        if flag_col:
            flag_mask = df_work['_flag_norm'].str.lower().isin(flag_positive_values)
        else:
            flag_mask = False

    # 정규식 패턴 기반 (신통/선정구역 등) 추가 탐지
    regex_mask = False
    if exp_regex_patterns:
        for pat in exp_regex_patterns:
            try:
                # _row_concat 우선, 없으면 type/flag
                target_series = df_work.get('_row_concat', df_work['_type_norm'])
                regex_mask = regex_mask | target_series.str.contains(pat, case=False, regex=True, na=False)
            except re.error:
                pass

    # 키워드 기반
    kw_norms = [k.replace(' ', '') for k in exp_keywords if k.strip()]
    if require_all_keywords and kw_norms:
        kw_mask = True
        for k in kw_norms:
            left = df_work['_type_norm'].str.contains(re.escape(k), regex=True)
            right = df_work['_flag_norm'].str.contains(re.escape(k), regex=True)
            if scan_all_columns:
                concat_mask = df_work['_row_concat'].str.contains(re.escape(k), regex=True)
            else:
                concat_mask = False
            kw_mask = kw_mask & (left | right | concat_mask)
    else:
        if kw_norms:
            pat = '|'.join(map(re.escape, kw_norms))
            base_mask = df_work['_type_norm'].str.contains(pat) | df_work['_flag_norm'].str.contains(pat)
            if scan_all_columns:
                base_mask = base_mask | df_work['_row_concat'].str.contains(pat)
            kw_mask = base_mask
        else:
            kw_mask = False

    # 고정 세트 사용 시 키워드/정규식은 보조 (flag_mask 우선)
    base_exp_mask = flag_mask | (kw_mask | regex_mask if not fixed_flag_value_set else False)

    # 계획 확정 상태
    if status_col and not fixed_flag_value_set:  # 고정 플래그 세트 사용 시 진행단계 조건 생략(사용자 요구사항 단순화)
        # 우선 수용 리스트(대체 상태) 체크
        accept_mask = False
        if plan_status_accept:
            accept_norm = [s.replace(' ', '') for s in plan_status_accept]
            accept_mask = df_work['_status_norm'].isin(accept_norm)
        if plan_status_keywords and not relax_plan_status:
            pat_status = '|'.join([re.escape(k.replace(' ', '')) for k in plan_status_keywords])
            status_mask = df_work['_status_norm'].str.contains(pat_status, regex=True) | accept_mask
            final_mask = base_exp_mask & status_mask
            if final_mask.sum() == 0:
                final_mask = base_exp_mask  # 완화
        else:
            final_mask = base_exp_mask
    else:
        final_mask = base_exp_mask

    exp_df = df_work[final_mask].copy()
    return exp_df


def filter_control_candidates(df: pd.DataFrame, mapping: Dict[str,str], exp_ids: set, year_range: Tuple[int,int]) -> pd.DataFrame:
    """대조군 후보: 실험군 제외 + 연도 범위 필터."""
    year_col = mapping.get("designation_year")
    date_col = mapping.get("designation_date")

    df_work = df.copy()
    # 이미 build_year_series 에서 _year 생성되어 있으면 활용
    if '_year' not in df_work.columns or df_work['_year'].isna().all():
        if year_col and year_col in df_work.columns:
            years = coerce_numeric(df_work[year_col])
        elif date_col and date_col in df_work.columns:
            years = extract_year_from_date(df_work[date_col])
        else:
            years = pd.Series([np.nan]*len(df_work), index=df_work.index)
        df_work['_year'] = years

    y0, y1 = year_range
    year_mask = (df_work['_year'] >= y0) & (df_work['_year'] <= y1)
    if isinstance(exp_ids, set):
        id_col = mapping.get('id') or mapping.get('project_name')
        if id_col and id_col in df_work.columns:
            not_exp = ~df_work[id_col].isin(exp_ids)
        else:
            not_exp = True
    else:
        not_exp = True
    filtered = df_work[year_mask & not_exp].copy()
    # Fallback: 연도 범위 내 0건이면 전체 연도(비어있지 않은)로 자동 확장 시도 (사용자 명시 범위 보존 안내)
    if filtered.empty and df_work['_year'].notna().sum() > 0:
        y_min = int(df_work['_year'].min())
        y_max = int(df_work['_year'].max())
        print(f'[CTRL][FALLBACK] 지정 year_range {year_range} 내 후보 0건 → 데이터 연도범위 ({y_min}-{y_max}) 로 재시도')
        filtered = df_work[(df_work['_year'] >= y_min) & (df_work['_year'] <= y_max) & not_exp].copy()
    return filtered


def build_year_series(df: pd.DataFrame, mapping: Dict[str,str], manual_year_col: Optional[str] = None) -> pd.Series:
    # 사용자 지정 우선
    if manual_year_col and manual_year_col in df.columns:
        cand_num = coerce_numeric(df[manual_year_col])
        if cand_num.notna().sum():
            print(f"[YEAR] manual '{manual_year_col}' numeric 추출 성공 non-null={cand_num.notna().sum()}")
            return cand_num
        # 숫자 실패 → 날짜 패턴 연도 추출 시도
        cand_year = extract_year_from_date(df[manual_year_col])
        if cand_year.notna().sum():
            print(f"[YEAR] manual '{manual_year_col}' 날짜에서 연도 추출 성공 non-null={cand_year.notna().sum()}")
            return cand_year
        else:
            print(f"[YEAR][WARN] manual '{manual_year_col}' 컬럼에서 연도 추출 실패 → fallback 진행")
    year_col = mapping.get("designation_year")
    date_col = mapping.get("designation_date")
    if year_col and year_col in df.columns:
        years = coerce_numeric(df[year_col])
    elif date_col and date_col in df.columns:
        years = extract_year_from_date(df[date_col])
    else:
        # 추가 fallback 후보
        picked = None
        for c in YEAR_FALLBACK_CANDIDATES:
            if c in df.columns:
                cand = coerce_numeric(df[c])
                if cand.notna().sum():
                    picked = cand; break
        if picked is None:
            years = pd.Series([np.nan]*len(df), index=df.index)
        else:
            years = picked
    return years


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_similarity(exp_row: pd.Series, cand_df: pd.DataFrame, mapping: Dict[str,str],
                       area_tol: float, member_tol: float, year_tol: int,
                       allow_year_missing: bool = False,
                       max_distance_km: float = 5.0) -> pd.DataFrame:
    area_col = mapping.get('area')
    mem_col = mapping.get('members')
    district_col = mapping.get('district')
    lat_col = mapping.get('latitude')
    lon_col = mapping.get('longitude')
    id_col = mapping.get('id') or mapping.get('project_name')
    name_col = mapping.get('project_name') or mapping.get('id')

    exp_area = exp_row.get(area_col)
    exp_mem = exp_row.get(mem_col)
    exp_year = exp_row.get('_year')
    exp_district = str(exp_row.get(district_col, '')).strip()
    exp_lat = exp_row.get(lat_col) if lat_col else np.nan
    exp_lon = exp_row.get(lon_col) if lon_col else np.nan

    dfc = cand_df.copy()
    dfc['_area_diff_ratio'] = np.abs(dfc[area_col] - exp_area) / exp_area
    dfc['_mem_diff_ratio']  = np.abs(dfc[mem_col] - exp_mem) / exp_mem
    if allow_year_missing:
        dfc['_year_diff'] = np.where(dfc['_year'].notna() & pd.notna(exp_year), np.abs(dfc['_year'] - exp_year), 0)
    else:
        dfc['_year_diff'] = np.abs(dfc['_year'] - exp_year)

    cond = (dfc['_area_diff_ratio'] <= area_tol) & (dfc['_mem_diff_ratio'] <= member_tol)
    if not allow_year_missing:
        cond = cond & (dfc['_year_diff'] <= year_tol)
    dfc = dfc[cond].copy()
    if dfc.empty:
        return dfc

    use_distance = lat_col in dfc.columns and lon_col in dfc.columns and pd.notna(exp_lat) and pd.notna(exp_lon)
    if use_distance:
        try:
            dfc['_dist_km'] = haversine_km(exp_lat, exp_lon, dfc[lat_col].astype(float), dfc[lon_col].astype(float))
        except Exception:
            dfc['_dist_km'] = np.nan
        dfc['_dist_km'] = dfc['_dist_km'].clip(upper=max_distance_km)
        dist_norm = dfc['_dist_km'] / max_distance_km
    else:
        dist_norm = 0.5  # 위치정보 없음 중립값
    # 자치구 동일 보너스 (거리 낮아도 동일 자치구면 약간 추가 가산) → 거리는 패널티이므로 동일 자치구면 dist_norm * 0.8
    if district_col in dfc.columns:
        same_dist_mask = dfc[district_col].astype(str).str.strip() == exp_district
        # 동일 자치구면 사실상 거리를 20% 감소한 것과 동등한 효과
        if use_distance:
            dist_norm = np.where(same_dist_mask, dist_norm * 0.8, dist_norm)
        else:
            # 거리 없으면 동일 자치구 0.4, 다르면 0.6 로 차등
            dist_norm = np.where(same_dist_mask, 0.4, 0.6)
    dfc['_loc_pen'] = WEIGHT_LOCATION * dist_norm

    if allow_year_missing:
        dfc['_score'] = (
            WEIGHT_AREA   * dfc['_area_diff_ratio'] +
            WEIGHT_MEMBER * dfc['_mem_diff_ratio'] +
            dfc['_loc_pen']
        )
    else:
        dfc['_score'] = (
            WEIGHT_AREA   * dfc['_area_diff_ratio'] +
            WEIGHT_MEMBER * dfc['_mem_diff_ratio'] +
            dfc['_loc_pen'] +
            WEIGHT_YEAR   * (dfc['_year_diff'] / max(year_tol,1))
        )

    keep_cols = [c for c in [id_col, name_col, district_col, lat_col, lon_col] if c]
    add_cols = ['_area_diff_ratio','_mem_diff_ratio','_year_diff','_score']
    if '_dist_km' in dfc.columns:
        add_cols.insert(0,'_dist_km')
    out = dfc[keep_cols + add_cols].copy().sort_values('_score')
    return out


def match_controls(exp_df: pd.DataFrame, control_df: pd.DataFrame, mapping: Dict[str,str],
                   area_tol: float, member_tol: float, year_tol: int,
                   controls_per_exp: int, min_controls: int,
                   allow_year_missing: bool = False,
                   max_distance_km: float = 5.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """실험군별 매칭을 수행하고 (matches, summary) 반환."""
    area_col = mapping.get('area')
    mem_col = mapping.get('members')
    id_col  = mapping.get('id') or mapping.get('project_name')
    name_col= mapping.get('project_name') or mapping.get('id')

    rows = []
    summaries = []
    for idx, exp_row in exp_df.iterrows():
        exp_id = exp_row.get(id_col, idx)
        exp_name = exp_row.get(name_col, exp_id)
        base_area_tol = area_tol
        base_mem_tol = member_tol
        base_year_tol = year_tol

        matched = None
        relax_level = -1
        # 순차 완화
        for step_i, step in enumerate([None] + RELAX_STEPS):
            if step is not None:
                area_tol_use = base_area_tol * step['area_tol_mul']
                mem_tol_use  = base_mem_tol  * step['member_tol_mul']
                year_tol_use = base_year_tol + step['year_tol_add']
                dist_km_use  = max_distance_km * step.get('dist_km_mul',1.0)
                relax_level = step_i
            else:
                area_tol_use, mem_tol_use, year_tol_use, dist_km_use = base_area_tol, base_mem_tol, base_year_tol, max_distance_km
                relax_level = 0
            sub = compute_similarity(exp_row, control_df, mapping, area_tol_use, mem_tol_use, year_tol_use,
                                     allow_year_missing=allow_year_missing,
                                     max_distance_km=dist_km_use)
            if not sub.empty:
                matched = sub.head(controls_per_exp)
                need_more = len(matched) < min_controls
                if not need_more:
                    break  # 기준 충족 -> 종료
        if matched is None:
            summaries.append({
                'exp_id': exp_id,
                'exp_name': exp_name,
                'matched': 0,
                'relax_level': None,
                'note': 'NO_MATCH'
            })
            continue

        matched = matched.copy()
        matched.loc[:, '_exp_id'] = exp_id
        matched.loc[:, '_exp_name'] = exp_name
        matched.loc[:, '_relax_level'] = relax_level
        rows.append(matched)
        summaries.append({
            'exp_id': exp_id,
            'exp_name': exp_name,
            'matched': len(matched),
            'relax_level': relax_level,
            'note': 'OK' if len(matched) >= min_controls else 'LOW_MATCH'
        })

    matches = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return matches, summary

# --------------------------------------------------
# 메인 실행
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Case-Control Matching for Redevelopment Projects')
    ap.add_argument('--csv', required=True, help='Full dataset CSV 경로')
    ap.add_argument('--output-dir', default='outputs/matching', help='결과 저장 폴더')
    ap.add_argument('--exp-keywords', default=','.join(DEFAULT_EXP_KEYWORDS), help='실험군 키워드 (콤마 구분)')
    ap.add_argument('--plan-status-keywords', default=','.join(DEFAULT_PLAN_STATUS_KEYWORDS), help='계획확정 상태 키워드 (콤마)')
    ap.add_argument('--year-range', default=f'{DEFAULT_CONTROL_YEAR_RANGE[0]}:{DEFAULT_CONTROL_YEAR_RANGE[1]}', help='대조군 후보 지정연도 범위 예: 2018:2021')
    ap.add_argument('--area-tol', type=float, default=DEFAULT_AREA_TOL, help='면적 허용 비율 (±)')
    ap.add_argument('--member-tol', type=float, default=DEFAULT_MEMBER_TOL, help='조합원 허용 비율 (±)')
    ap.add_argument('--year-tol', type=int, default=DEFAULT_YEAR_TOL, help='연도 차이 허용 (±)')
    ap.add_argument('--controls-per-experiment', type=int, default=DEFAULT_CONTROLS_PER_EXP, help='실험군 1건당 최대 대조군 수')
    ap.add_argument('--min-controls-per-experiment', type=int, default=DEFAULT_MIN_CONTROLS_PER_EXP, help='충족되었다고 간주할 최소 매칭 수')
    ap.add_argument('--require-all-exp-keywords', action='store_true', help='실험군 키워드 모두 포함(AND). 기본적으로 AND 강제.')
    ap.add_argument('--relax-plan-status', action='store_true', help='계획확정(진행단계) 상태 조건을 만족 레코드 없을 때 무시')
    ap.add_argument('--exp-flag-positive', default='Y,예,1,True,신속,신속통합기획', help='exp_flag 컬럼 True 로 간주할 값 목록(콤마)')
    ap.add_argument('--scan-all-columns', action='store_true', help='project_type 컬럼 없거나 불완전할 때 전체 문자열 컬럼에서 키워드 검색')
    ap.add_argument('--exp-regex', default=None, help='실험군 정규식 패턴(콤마 구분). 기본: 신속통합|신통|선정구역 등')
    ap.add_argument('--plan-status-accept', default=None, help='계획확정 대체로 인정할 진행단계 값(콤마). 기본 내장 리스트 존재')
    ap.add_argument('--year-col', default=None, help='지정연도 수동 지정 컬럼명')
    ap.add_argument('--allow-year-missing', action='store_true', help='연도 추출 실패 시 연도 조건 생략')
    ap.add_argument('--max-distance-km', type=float, default=5.0, help='동일 자치구 없을 때 거리 기반 필터 최대 km')
    # district-strict-first 제거: 거리 중심 점수화로 대체
    ap.add_argument('--abort-if-empty-exp', action='store_true', help='실험군 0건이면 종료(기본은 진단 출력 후 종료 동일)')
    ap.add_argument('--print-inspect', action='store_true', help='초기 주요 컬럼 고유값 샘플 출력')
    ap.add_argument('--export-diagnostics', action='store_true', help='추가 매핑/결측 진단 CSV 저장')
    ap.add_argument('--encoding', default='utf-8-sig', help='CSV 인코딩 우선 시도')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[ERROR] CSV 파일 없음: {csv_path}')
        sys.exit(1)

    # 다중 인코딩 로드
    tried = []
    for enc in [args.encoding, 'utf-8', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f'[LOAD] 인코딩 {enc} -> {df.shape}')
            break
        except Exception as e:
            tried.append(f'{enc}:{e}')
            df = None
    if df is None:
        print('[ERROR] CSV 로드 실패:', tried)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = detect_columns(df)
    print('[MAP] 컬럼 매핑:', mapping)
    if args.print_inspect:
        for k in ['project_type','exp_flag','plan_status','district']:
            col = mapping.get(k)
            if col and col in df.columns:
                vals = df[col].dropna().astype(str).str.strip().value_counts().head(12)
                print(f'[INSPECT] {k}({col}) 상위값:\n{vals}\n')
        # '최초' 패턴 컬럼 후보 안내
        first_cols = [c for c in df.columns if '최초' in c]
        if first_cols:
            print('[INFO] "최초" 포함 컬럼 후보:', first_cols[:15])

    # 숫자 변환
    for k in ['area','members']:
        c = mapping.get(k)
        if c and c in df.columns:
            df[c] = coerce_numeric(df[c])

    # 연도 시리즈 구축
    years = build_year_series(df, mapping, manual_year_col=args.year_col)
    df['_year'] = years

    # 실험군 선정
    exp_keywords = [s for s in args.exp_keywords.split(',') if s.strip()]
    plan_keywords = [s for s in args.plan_status_keywords.split(',') if s.strip()]
    flag_positive_values = {v.lower() for v in args.exp_flag_positive.split(',') if v.strip()}
    # 정규식 / 대체 상태 파싱
    if args.exp_regex:
        exp_regex_patterns = [p for p in args.exp_regex.split(',') if p.strip()]
    else:
        exp_regex_patterns = DEFAULT_EXP_REGEX_PATTERNS
    if args.plan_status_accept:
        plan_status_accept = [s.strip() for s in args.plan_status_accept.split(',') if s.strip()]
    else:
        plan_status_accept = DEFAULT_PLAN_STATUS_ACCEPT

    # 고정 실험군 값 세트 (사용자 명시: 1차선정구역, 2차선정구역, 기존구역(신통추진))
    FIXED_EXP_VALUES = {"1차선정구역", "2차선정구역", "기존구역(신통추진)"}

    exp_df = filter_experimental(
        df, mapping,
        exp_keywords=exp_keywords,
        plan_status_keywords=plan_keywords,
        require_all_keywords=True,
        flag_positive_values=flag_positive_values,
        relax_plan_status=args.relax_plan_status,
        scan_all_columns=args.scan_all_columns,
        exp_regex_patterns=exp_regex_patterns,
        plan_status_accept=plan_status_accept,
        fixed_flag_value_set=FIXED_EXP_VALUES,
    )
    print(f'[EXP] 실험군 후보 {len(exp_df)}건')

    # ID 중복 제거 (동일 사업 중복행 있을 수 있음)
    id_col = mapping.get('id') or mapping.get('project_name')
    if id_col and id_col in exp_df.columns:
        exp_df = exp_df.drop_duplicates(subset=[id_col])
        print(f'[EXP] 중복 제거 후 {len(exp_df)}건')

    # 대조군 후보
    year_range = tuple(int(x) for x in args.year_range.split(':'))
    exp_ids = set(exp_df[id_col].tolist()) if id_col and id_col in exp_df.columns else set()
    control_df = filter_control_candidates(df, mapping, exp_ids=exp_ids, year_range=year_range)
    # 연도 전부 NaN 이고 allow-year-missing 이면 연도 필터 제거
    if control_df.empty and df['_year'].notna().sum() == 0 and args.allow_year_missing:
        print('[CTRL][INFO] 연도 추출 실패 → 연도 조건 무시 후 전체 재평가')
        control_df = df[~df.get(id_col, pd.Series(index=df.index)).isin(exp_ids)].copy()
        control_df['_year'] = np.nan
    print(f'[CTRL] 대조군 연도 후보 {len(control_df)}건 (연도 범위 {year_range})')

    # 디버그: 핵심 수치 결측 현황
    area_col = mapping.get('area'); mem_col = mapping.get('members')
    if area_col in df.columns:
        print(f'[DEBUG] area non-null total={df[area_col].notna().sum()} / {len(df)} (sample head)')
        print(df[area_col].head())
    if mem_col in df.columns:
        print(f'[DEBUG] members non-null total={df[mem_col].notna().sum()} / {len(df)} (sample head)')
        print(df[mem_col].head())

    # 기본 결측 제거 (면적/조합원/연도 없으면 매칭 불가)
    essential_cols = [mapping.get('area'), mapping.get('members')]
    ctrl_before = len(control_df)
    for c in essential_cols:
        if c:
            control_df = control_df[control_df[c].notna()]
    if not args.allow_year_missing:
        control_df = control_df[control_df['_year'].notna()]
    print(f'[CTRL] 필수 결측 제거 {ctrl_before} -> {len(control_df)}')

    # 실험군도 필수 값 없는 것 제거
    exp_before = len(exp_df)
    for c in essential_cols:
        if c:
            exp_df = exp_df[exp_df[c].notna()]
    if not args.allow_year_missing:
        exp_df = exp_df[exp_df['_year'].notna()]
    print(f'[EXP] 필수 결측 제거 {exp_before} -> {len(exp_df)}')

    if exp_df.empty:
        print('[ERROR] 실험군 0건 — 진단 참고:')
        for tag, logical in [('exp_flag','exp_flag'),('plan_status','plan_status'),('project_type','project_type')]:
            col = mapping.get(logical)
            if col and col in df.columns:
                uniq = df[col].dropna().astype(str).str.strip().unique()[:30]
                print(f'  - {tag}({col}) 유니크 샘플(최대30): {uniq}')
        print('  - 사용된 실험군 정규식 패턴:', exp_regex_patterns)
        print('  - 인정된 계획단계 목록:', plan_status_accept)
        if not args.abort_if_empty_exp:
            sys.exit(1)
        else:
            return

    matches, summary = match_controls(
        exp_df, control_df, mapping,
        area_tol=args.area_tol,
        member_tol=args.member_tol,
        year_tol=args.year_tol,
        controls_per_exp=args.controls_per_experiment,
        min_controls=args.min_controls_per_experiment,
        allow_year_missing=args.allow_year_missing,
        max_distance_km=args.max_distance_km
    )

    matches_path = output_dir / 'case_control_matches.csv'
    summary_path = output_dir / 'case_control_summary.csv'

    if not matches.empty:
        matches.to_csv(matches_path, index=False, encoding='utf-8-sig')
        print(f'[SAVE] 매칭 결과: {matches_path} ({len(matches)} rows)')
    else:
        print('[WARN] 매칭 결과 비어 있음')

    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'[SAVE] 요약: {summary_path}')

    if args.export_diagnostics:
        diag = {
            'exp_keywords': exp_keywords,
            'plan_status_keywords': plan_keywords,
            'mapping': mapping,
            'params': {
                'area_tol': args.area_tol,
                'member_tol': args.member_tol,
                'year_tol': args.year_tol,
                'controls_per_experiment': args.controls_per_experiment,
                'min_controls_per_experiment': args.min_controls_per_experiment,
                'year_range': year_range,
            },
            'counts': {
                'exp': len(exp_df),
                'control_candidates': len(control_df)
            }
        }
        diag_path = output_dir / 'matching_diagnostics.json'
        json.dump(diag, open(diag_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f'[SAVE] 진단: {diag_path}')

    print('[DONE] Matching complete.')

if __name__ == '__main__':
    main()
