#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
신속통합기획(실험군) 샘플 파일럿 생존분석 스크립트
-------------------------------------------------
목표 이벤트: 정비구역지정일(start) → 조합설립인가(event)
검열(censor): 컷오프 날짜까지 event 미도달 시 일 단위 검열

모델:
- KM 생존곡선 (전체 신속통합기획 샘플)
- Cox PH: 공공성(임대세대비율) + 통제(정비구역면적, 토지등소유자수)
  · 수치 안정화를 위해 log1p 변환 후 표준화(평균0, 표준편차1)

입력 기본: outputs/주택재개발_DATA_full.csv
출력: outputs/survival/
  - pilot_fasttrack_km.png (KM plot)
  - pilot_fasttrack_cox_summary.csv (Cox 요약)
  - pilot_fasttrack_dataset.csv (분석 데이터셋)
"""
from __future__ import annotations
import argparse, sys, re, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


# --------------------------- 컬럼 동의어 ---------------------------
COLUMN_SYNONYMS: Dict[str, list[str]] = {
    # 식별/명칭
    'id': ['사업번호','사업코드','ID','id','식별자','정비구역코드'],
    'name': ['정비구역명칭','정비구역명','사업명','구역명','사업장명'],
    # 실험군 플래그
    'exp_flag': ['신속통합기획','신속통합기획여부','신속통합기획_구분'],
    # 날짜: 시작(정비구역 지정) / 이벤트(조합설립인가)
    'start_date': [
        '정비구역지정일','지정일','정비구역 지정일','정비구역지정고시일','지정고시일',
        '구역지정최초','구역지정변경(최종)','구역지정_최초','구역지정_변경(최종)'
    ],
    'event_date': [
        '조합설립인가(사업시행자 지정일)','조합설립인가_최초','조합설립인가_변경(최종)','조합설립인가',
        '조합설립인가일','조합설립 인가일'
    ],
    # 공공성
    'rental_units': ['임대세대총수','임대세대','임대세대수','(임대)세대수','임대세대 합계'],
    'total_households': ['세대총합계','세대수합계','총세대수','분양세대총수'],
    # 통제
    'area': ['정비구역면적(㎡)','정비구역면적','구역면적','사업면적','면적','면적(㎡)'],
    'owners': ['토지등 소유자 수','토지등소유자수','토지 등 소유자 수'],
    # 추가 변수
    'district': ['자치구','구','시군구','행정구역','자치 단체'],
    'far': ['용적률','용적률(%)','계획용적률','기준용적률'],
    'bcr': ['건폐율','건폐율(%)']
}

POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}
EXP_REGEX = r'(신속통합|신통|선정구역)'


def detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
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


def build_fasttrack_subset(df: pd.DataFrame, exp_flag_col: Optional[str]) -> pd.DataFrame:
    if exp_flag_col:
        flag_mask = df[exp_flag_col].astype(str).str.strip().str.lower().isin(POSITIVE_FLAG_VALUES)
    else:
        flag_mask = pd.Series([False]*len(df), index=df.index)
    # 보조: 전체 텍스트에서 정규식 검색
    if '_row_concat' not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        df['_row_concat'] = df[obj_cols].astype(str).agg(' '.join, axis=1)
    regex_mask = df['_row_concat'].str.contains(EXP_REGEX, regex=True, na=False)
    sub = df[flag_mask | regex_mask].copy()
    # 그룹 구분: 1차선정구역 / 2차선정구역 / 기존구역(신통추진)
    def classify_group(row) -> str:
        text = ''
        if exp_flag_col and exp_flag_col in sub.columns and pd.notna(row.get(exp_flag_col)):
            text += str(row[exp_flag_col])
        if '_row_concat' in sub.columns and pd.notna(row.get('_row_concat')):
            text += ' ' + str(row['_row_concat'])
        t = text.replace(' ', '')
        if '1차선정구역' in t:
            return '1차선정구역'
        if '2차선정구역' in t:
            return '2차선정구역'
        if '기존구역(신통추진)' in t or '기존구역' in t:
            return '기존선정구역'
        return '기타'
    sub['_exp_group'] = sub.apply(classify_group, axis=1)
    return sub


def main():
    ap = argparse.ArgumentParser(description='신속통합기획 샘플 파일럿 생존분석')
    ap.add_argument('--csv', default='outputs/주택재개발_DATA_full.csv', help='입력 CSV 경로')
    ap.add_argument('--cutoff', default=datetime.today().strftime('%Y-%m-%d'), help='컷오프 날짜 YYYY-MM-DD (기본: 오늘)')
    ap.add_argument('--outdir', default='outputs/survival', help='출력 디렉터리')
    ap.add_argument('--encoding', default='utf-8-sig')
    ap.add_argument('--plot-dpi', type=int, default=140)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[ERROR] CSV 없음: {csv_path}'); sys.exit(1)

    # 다중 인코딩 로드
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

    # 컬럼 탐지
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

    for k, v in [('start_date',col_start), ('event_date',col_event)]:
        if not v:
            print(f'[WARN] {k} 컬럼 탐지 실패 -> 일부 검열/이벤트 계산 불가')

    # 신속통합기획 샘플 구축
    fast_df = build_fasttrack_subset(df, col_flag)
    print(f'[EXP] 신속통합기획 샘플 {len(fast_df)}건')

    # 필수: 시작일 존재
    if not col_start or col_start not in fast_df.columns:
        print('[FATAL] 시작일(정비구역지정일) 컬럼을 찾지 못함. 종료.'); sys.exit(1)

    start = parse_date(fast_df[col_start])
    event = parse_date(fast_df[col_event]) if col_event else pd.Series([pd.NaT]*len(fast_df), index=fast_df.index)
    cutoff = pd.to_datetime(args.cutoff)

    # 음수/비정상 날짜 정리
    valid_mask = start.notna()
    fast_df = fast_df[valid_mask].copy()
    start = start[valid_mask]
    if not event.empty:
        event = event[valid_mask]

    # 검열 로직
    # event_obs: event <= cutoff
    event_obs = event.notna() & (event <= cutoff)
    # duration: (event if observed else cutoff) - start
    end_dates = np.where(event_obs, event, cutoff)
    end_dates = pd.to_datetime(end_dates)
    duration_days = (end_dates - start).dt.days

    # 음수 제거 (이상치)
    ok = duration_days >= 0
    fast_df = fast_df[ok].copy()
    duration_days = duration_days[ok]
    event_obs = event_obs[ok]
    start = start[ok]
    if not event.empty:
        event = event[ok]

    # 일 → 월 환산 (평균 월 일수 사용)
    DAYS_PER_MONTH = 30.44
    duration_months = duration_days / DAYS_PER_MONTH

    # 공공성: 임대세대비율
    if col_rent and col_rent in fast_df.columns:
        rent = coerce_numeric(fast_df[col_rent])
    else:
        rent = pd.Series([np.nan]*len(fast_df), index=fast_df.index)
    if col_hh and col_hh in fast_df.columns:
        hh = coerce_numeric(fast_df[col_hh])
    else:
        hh = pd.Series([np.nan]*len(fast_df), index=fast_df.index)
    rental_ratio = np.where((rent.notna() & hh.notna() & (hh>0)), (rent / hh), np.nan)

    # 통제: 면적, 토지등소유자수
    area = coerce_numeric(fast_df[col_area]) if col_area else pd.Series([np.nan]*len(fast_df), index=fast_df.index)
    owners = coerce_numeric(fast_df[col_owners]) if col_owners else pd.Series([np.nan]*len(fast_df), index=fast_df.index)

    # 추가: 용적률(FAR), 건폐율(BCR), 자치구, 전체세대수(hh)
    far_raw = coerce_numeric(fast_df[col_far]) if col_far else pd.Series([np.nan]*len(fast_df), index=fast_df.index)
    bcr_raw = coerce_numeric(fast_df[col_bcr]) if col_bcr else pd.Series([np.nan]*len(fast_df), index=fast_df.index)
    # 퍼센트 형태일 가능성이 높아 1보다 큰 값은 100으로 나눠 비율로 변환
    far_ratio = np.where(pd.notna(far_raw) & (far_raw > 1.0), far_raw/100.0, far_raw)
    bcr_ratio = np.where(pd.notna(bcr_raw) & (bcr_raw > 1.0), bcr_raw/100.0, bcr_raw)
    district = fast_df[col_district].astype(str) if col_district else pd.Series([np.nan]*len(fast_df), index=fast_df.index)

    # 분석 데이터프레임 구성
    out_df = pd.DataFrame({
        'id': fast_df[col_id] if col_id else fast_df.index,
        'name': fast_df[col_name] if col_name else np.nan,
        'start_date': start.dt.strftime('%Y-%m-%d'),
        'event_date': event.dt.strftime('%Y-%m-%d') if not event.empty else np.nan,
        'duration_days': duration_days,
        'duration_months': duration_months,
        'event_observed': event_obs.astype(int),
        'rental_ratio': rental_ratio,
        'area_m2': area,
        'owners': owners,
        'households_total': hh,
        'far': far_raw,
        'bcr': bcr_raw,
        'district': district,
        'exp_group': fast_df.get('_exp_group', pd.Series(['기타']*len(fast_df), index=fast_df.index)),
    })

    # Cox용 전처리: log1p & 표준화
    def log1p_std(x: pd.Series) -> pd.Series:
        x1 = np.log1p(x.astype(float))
        m, s = np.nanmean(x1), np.nanstd(x1)
        return (x1 - m) / (s if s and s>0 else 1.0)

    out_df['rental_ratio_fill'] = out_df['rental_ratio']
    # rental_ratio는 0~1 범위를 기대. 이상치/100 스케일일 경우 1 넘으면 100으로 나눔
    over1 = out_df['rental_ratio_fill'] > 1.0
    out_df.loc[over1, 'rental_ratio_fill'] = out_df.loc[over1, 'rental_ratio_fill'] / 100.0

    out_df['logstd_area'] = log1p_std(out_df['area_m2'])
    out_df['logstd_owners'] = log1p_std(out_df['owners'])
    out_df['logstd_households'] = log1p_std(out_df['households_total'])
    out_df['std_rental_ratio'] = (out_df['rental_ratio_fill'] - np.nanmean(out_df['rental_ratio_fill'])) / (np.nanstd(out_df['rental_ratio_fill']) or 1.0)
    # FAR/BCR 표준화 (비율 스케일 기반)
    out_df['far_ratio'] = far_ratio
    out_df['bcr_ratio'] = bcr_ratio
    out_df['std_far'] = (out_df['far_ratio'] - np.nanmean(out_df['far_ratio'])) / (np.nanstd(out_df['far_ratio']) or 1.0)
    out_df['std_bcr'] = (out_df['bcr_ratio'] - np.nanmean(out_df['bcr_ratio'])) / (np.nanstd(out_df['bcr_ratio']) or 1.0)

    # 결측 제거(Cox용)
    covs = ['std_rental_ratio','logstd_area','logstd_owners','logstd_households','std_far','std_bcr']
    base_cols = ['duration_months','event_observed']
    # district은 strata 후보로만 사용 (결측 허용하며 별도 처리)
    cox_all = out_df[base_cols + covs + ['district']].copy()
    cox_df = cox_all.dropna(subset=base_cols + covs)
    print(f"[DATA] Cox N={len(cox_df)} (from {len(out_df)}) [time unit: months]; covariates={covs}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dataset_path = outdir / 'pilot_fasttrack_dataset.csv'
    out_df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    print(f'[SAVE] dataset -> {dataset_path}')

    # KM Plot (전체)
    try:
        import matplotlib.pyplot as plt
        # Windows: 한글 폰트 설정 (맑은 고딕)
        try:
            import matplotlib
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test

        km = KaplanMeierFitter()
        km.fit(durations=out_df['duration_months'], event_observed=out_df['event_observed'])
        ax = km.plot(ci_show=True)
        ax.set_title('KM: 정비구역지정 → 조합설립인가 (신속통합기획 샘플)')
        ax.set_xlabel('months')
        ax.set_ylabel('survival probability')
        ax.grid(True, alpha=0.3)
        km_path = outdir / 'pilot_fasttrack_km.png'
        plt.tight_layout()
        plt.savefig(km_path, dpi=args.plot_dpi)
        plt.close()
        print(f'[SAVE] KM plot -> {km_path}')

        # Cox PH
        cph = CoxPHFitter()
        # 자치구 strata 조건: 그룹 수 2~5, 각 그룹 n>=3일 때만 적용
        use_strata = False
        try:
            dist_counts = cox_df['district'].dropna().value_counts()
            if 2 <= len(dist_counts) <= 5 and (dist_counts.min() >= 3):
                use_strata = True
        except Exception:
            use_strata = False
        fit_kwargs = dict(duration_col='duration_months', event_col='event_observed')
        if use_strata:
            print(f"[COX] using district strata with groups={list(dist_counts.index)}")
            df_fit = cox_df.dropna(subset=['district'])
            cph.fit(df_fit, strata=['district'], **fit_kwargs)
        else:
            df_fit = cox_df.drop(columns=['district'], errors='ignore')
            cph.fit(df_fit, **fit_kwargs)
        # lifelines summary: index=변수명. pandas 버전에 따라 reset_index 컬럼명이 다를 수 있음
        try:
            summary = cph.summary.reset_index().rename(columns={'index':'variable'})
        except Exception:
            summary = cph.summary.copy()
        # 해석 편의: HR = exp(coef)
        if 'exp(coef)' not in summary.columns and 'coef' in summary.columns:
            summary['HR'] = np.exp(summary['coef'])
        cox_path = outdir / 'pilot_fasttrack_cox_summary.csv'
        summary.to_csv(cox_path, index=False, encoding='utf-8-sig')
        print(f'[SAVE] Cox summary -> {cox_path}')
        # 안전한 요약 프린트
        var_col = 'variable' if 'variable' in summary.columns else ('covariate' if 'covariate' in summary.columns else summary.columns[0])
        hr_col = 'exp(coef)' if 'exp(coef)' in summary.columns else ('HR' if 'HR' in summary.columns else None)
        cols_to_show = [var_col, 'coef'] + ([hr_col] if hr_col else [])
        try:
            print(summary[cols_to_show].head().to_string(index=False))
        except Exception:
            print(summary.head().to_string(index=False))

        # ---------------- Visuals: KM by variable (median split) ----------------
        def km_by_median(var_series: pd.Series, label: str, filename: str):
            s = pd.to_numeric(var_series, errors='coerce')
            mask = s.notna() & out_df['duration_months'].notna() & out_df['event_observed'].notna()
            if mask.sum() < 10:
                return None
            s = s[mask]
            d = out_df.loc[mask, 'duration_months']
            e = out_df.loc[mask, 'event_observed']
            med = np.nanmedian(s)
            grp = np.where(s <= med, f'Low (≤ {med:.3g})', f'High (> {med:.3g})')
            km = KaplanMeierFitter()
            fig, ax = plt.subplots()
            for g in np.unique(grp):
                idx = grp == g
                km.fit(durations=d[idx], event_observed=e[idx], label=g)
                km.plot(ax=ax, ci_show=True)
            # log-rank
            idx_low = grp == np.unique(grp)[0]
            idx_high = grp == np.unique(grp)[1]
            try:
                lr = logrank_test(d[idx_low], d[idx_high], e[idx_low], e[idx_high])
                pval = lr.p_value
            except Exception:
                pval = np.nan
            ax.set_title(f'KM by {label} (median split)\nlog-rank p={pval:.3g}')
            ax.set_xlabel('months'); ax.set_ylabel('survival probability'); ax.grid(True, alpha=0.3)
            path = outdir / filename
            plt.tight_layout(); plt.savefig(path, dpi=args.plot_dpi); plt.close()
            print(f'[SAVE] KM by {label} -> {path}')
            return float(pval) if not pd.isna(pval) else None

        p_rental = km_by_median(out_df['rental_ratio_fill'], '임대세대비율', 'km_by_rental_ratio.png')
        p_area   = km_by_median(out_df['area_m2'], '정비구역면적', 'km_by_area.png')
        p_owner  = km_by_median(out_df['owners'], '토지등소유자수', 'km_by_owners.png')
        # 추가 변수 KM (median split)
        _p_hh  = km_by_median(out_df['households_total'], '세대총합계', 'km_by_households.png')
        _p_far = km_by_median(out_df['far'], '용적률(원자료)', 'km_by_far.png')
        _p_bcr = km_by_median(out_df['bcr'], '건폐율(원자료)', 'km_by_bcr.png')

        # ---------------- KM by selection group ----------------
        try:
            grp_col = 'exp_group'
            if grp_col in out_df.columns:
                gvals = out_df[grp_col].fillna('기타').astype(str)
                d = out_df['duration_months']; e = out_df['event_observed']
                km = KaplanMeierFitter(); fig, ax = plt.subplots()
                for g in sorted(gvals.unique()):
                    idx = gvals == g
                    if idx.sum() < 3:  # 너무 적으면 스킵
                        continue
                    km.fit(durations=d[idx], event_observed=e[idx], label=g)
                    km.plot(ax=ax)
                # 다집단 로그순위검정
                try:
                    mv = multivariate_logrank_test(durations=d, groups=gvals, event_observed=e)
                    pval = float(mv.p_value)
                except Exception:
                    pval = np.nan
                ax.set_title(f'KM by Selection Group (신속통합기획)\nmultivariate log-rank p={pval:.3g}')
                ax.set_xlabel('months'); ax.set_ylabel('survival probability'); ax.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(outdir / 'km_by_exp_group.png', dpi=args.plot_dpi); plt.close()
                print(f'[SAVE] KM by group -> {outdir / "km_by_exp_group.png"}')
                # 그룹 요약 내보내기
                grp_sum = out_df.groupby(grp_col).agg(
                    n=('id','count'),
                    events=('event_observed','sum'),
                    median_months=('duration_months','median')
                ).reset_index()
                grp_sum.to_csv(outdir / 'group_summary.csv', index=False, encoding='utf-8-sig')
                print(f'[SAVE] group summary -> {outdir / "group_summary.csv"}')
                # 자치구 요약
                if 'district' in out_df.columns:
                    dist_sum = out_df.groupby('district', dropna=False).agg(
                        n=('id','count'),
                        events=('event_observed','sum'),
                        median_months=('duration_months','median')
                    ).reset_index().sort_values('n', ascending=False)
                    dist_sum.to_csv(outdir / 'district_summary.csv', index=False, encoding='utf-8-sig')
                    print(f'[SAVE] district summary -> {outdir / "district_summary.csv"}')
        except Exception as e:
            print('[WARN] KM by group 실패:', e)

        # ---------------- Visuals: Cox forest (HR with CI) ----------------
        try:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            plot_df = summary.copy()
            # lifelines columns names can vary; standardize
            if 'exp(coef) lower 95%' in plot_df.columns:
                low = plot_df['exp(coef) lower 95%']
                high = plot_df['exp(coef) upper 95%']
                hr = plot_df['exp(coef)'] if 'exp(coef)' in plot_df.columns else plot_df.get('HR', np.exp(plot_df['coef']))
                names = plot_df.get('variable', plot_df.index)
            else:
                # compute CI from coef +/- 1.96*se(coef) if available
                hr = plot_df['exp(coef)'] if 'exp(coef)' in plot_df.columns else plot_df.get('HR', np.exp(plot_df['coef']))
                se = plot_df.get('se(coef)', None)
                if se is not None:
                    low = np.exp(plot_df['coef'] - 1.96*se)
                    high = np.exp(plot_df['coef'] + 1.96*se)
                else:
                    low = np.nan*hr; high = np.nan*hr
                names = plot_df.get('variable', plot_df.index)
            y = np.arange(len(hr))[::-1]
            ax.errorbar(hr, y, xerr=[hr-low, high-hr], fmt='o', color='C0', ecolor='gray', capsize=3)
            ax.axvline(1.0, color='red', linestyle='--', alpha=0.6)
            ax.set_yticks(y); ax.set_yticklabels(names)
            ax.set_xlabel('Hazard Ratio (HR)')
            ax.set_title('Cox Model Effects (HR with 95% CI)')
            plt.tight_layout(); plt.savefig(outdir / 'cox_forest.png', dpi=args.plot_dpi); plt.close()
            print(f'[SAVE] forest -> {outdir / "cox_forest.png"}')
        except Exception as e:
            print('[WARN] forest plot 실패:', e)

        # ---------------- Visuals: Partial survival curves per covariate ----------------
        try:
            covs = ['std_rental_ratio','logstd_area','logstd_owners']
            labels = {'std_rental_ratio':'임대세대비율(표준화)','logstd_area':'면적(log1p 표준화)','logstd_owners':'토지등소유자수(log1p 표준화)'}
            med_vals = {k: float(np.nanmedian(cox_df[k])) for k in covs}
            quant = [0.25, 0.5, 0.75]
            for v in covs:
                grid_vals = [float(np.nanquantile(cox_df[v], q)) for q in quant]
                fig, ax = plt.subplots()
                for gv, q in zip(grid_vals, quant):
                    row = {k: med_vals[k] for k in covs}; row[v] = gv
                    surv = cph.predict_survival_function(pd.DataFrame([row]))
                    ax.plot(surv.index, surv.values[:,0], label=f'{labels[v]} q{int(q*100)}')
                ax.set_title(f'Partial survival by {labels[v]} (others at median)')
                ax.set_xlabel('months'); ax.set_ylabel('survival'); ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(outdir / f'partial_survival_{v}.png', dpi=args.plot_dpi); plt.close()
                print(f'[SAVE] partial -> {outdir / f"partial_survival_{v}.png"}')
        except Exception as e:
            print('[WARN] partial survival 생성 실패:', e)

        # ---------------- Model metrics export ----------------
        metrics = {}
        try:
            metrics['concordance_index'] = float(getattr(cph, 'concordance_index_', np.nan))
        except Exception:
            metrics['concordance_index'] = None
        try:
            # available in lifelines
            metrics['AIC_partial'] = float(getattr(cph, 'AIC_partial_', np.nan))
        except Exception:
            pass
        try:
            ll = float(getattr(cph, 'log_likelihood_', np.nan))
            ll0 = float(getattr(cph, 'log_likelihood_null_', np.nan))
            n = int(len(cox_df))
            if not np.isnan(ll) and not np.isnan(ll0) and n>0:
                r2_nagelkerke = (1 - np.exp((2/n)*(ll0-ll))) / (1 - np.exp((2/n)*ll0)) if ll0 != 0 else np.nan
                metrics['log_likelihood'] = ll
                metrics['log_likelihood_null'] = ll0
                metrics['r2_nagelkerke'] = float(r2_nagelkerke)
        except Exception:
            pass
        try:
            lr = cph.log_likelihood_ratio_test()
            metrics['lr_test_p'] = float(getattr(lr, 'p_value', np.nan))
            metrics['lr_test_stat'] = float(getattr(lr, 'test_statistic', np.nan))
        except Exception:
            pass
        with open(outdir / 'pilot_fasttrack_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print('[SAVE] metrics ->', outdir / 'pilot_fasttrack_metrics.json')
    except ImportError as e:
        print('[WARN] lifelines/matplotlib 미설치로 KM/Cox 생략. 패키지 설치 후 재실행하세요.')

    print('[DONE] Pilot survival analysis complete.')


if __name__ == '__main__':
    main()
