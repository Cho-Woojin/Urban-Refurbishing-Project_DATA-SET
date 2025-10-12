from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ENCODINGS_TRY = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}
EXP_REGEX = r'(?:신속통합|신통|선정구역)'

START_DATE_CANDIDATES = [
    '정비구역지정일','지정일','정비구역 지정일','정비구역지정고시일','지정고시일',
    '구역지정최초','구역지정변경(최종)','구역지정_최초','구역지정_변경(최종)'
]
EVENT_DATE_CANDIDATES = [
    '조합설립인가(사업시행자 지정일)','조합설립인가_최초','조합설립인가_변경(최종)','조합설립인가',
    '조합설립인가일','조합설립 인가일'
]


def read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (마지막 에러: {last_err})")


def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i","u","f"):
        return s.astype(float)
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",","",regex=False)
         .str.replace("%","",regex=False)
         .str.strip(),
        errors="coerce"
    )


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def coalesce_dates(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    s = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    for c in candidates:
        if c in df.columns:
            s = s.combine_first(to_datetime_safe(df[c]))
    return s


def build_fast_flag(df: pd.DataFrame) -> pd.Series:
    flag_cols = ["신속통합기획", "신속통합기획여부", "신속통합기획_구분"]
    present = [c for c in flag_cols if c in df.columns]
    if present:
        mask = pd.Series(False, index=df.index)
        for c in present:
            val = df[c].astype(str).str.strip().str.lower()
            mask = mask | val.isin(POSITIVE_FLAG_VALUES)
    else:
        mask = pd.Series(False, index=df.index)
    if '_row_concat' not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        try:
            df['_row_concat'] = df[obj_cols].astype(str).agg(' '.join, axis=1)
        except Exception:
            df['_row_concat'] = ""
    regex_mask = df['_row_concat'].str.contains(EXP_REGEX, regex=True, na=False)
    return (mask | regex_mask).astype(int)


def welch_ttest(x: pd.Series, y: pd.Series):
    try:
        from scipy.stats import ttest_ind
        res = ttest_ind(x.dropna().astype(float), y.dropna().astype(float), equal_var=False)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float('nan'), float('nan')


def chisq_p(ct: pd.DataFrame) -> float:
    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, exp = chi2_contingency(ct.fillna(0).values)
        return float(p)
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser(description="유의 결과 요약 차트(한 장)")
    ap.add_argument("--input", default=str(Path("outputs")/"주택재개발_DATA_full.csv"))
    ap.add_argument("--outpath", default=str(Path("outputs")/"analysis"/"significant_summary.png"))
    ap.add_argument("--min-start-year", type=int, default=2021)
    ap.add_argument("--cutoff-date", type=str, default=datetime.today().strftime('%Y-%m-%d'))
    args = ap.parse_args()

    src = Path(args.input)
    df = read_csv_smart(src)

    # Fast flag
    df['fast_flag'] = build_fast_flag(df)

    # Dates and duration
    start_dates = coalesce_dates(df, START_DATE_CANDIDATES)
    event_dates = coalesce_dates(df, EVENT_DATE_CANDIDATES)
    if not start_dates.isna().all():
        mask_year = start_dates.dt.year >= int(args.min_start_year)
        df = df.loc[mask_year].copy(); start_dates = start_dates.loc[df.index]; event_dates = event_dates.loc[df.index]
    cutoff = pd.to_datetime(args.cutoff_date)
    event_obs = event_dates.notna() & (event_dates <= cutoff)
    end_dates = pd.to_datetime(np.where(event_obs, event_dates, cutoff))
    duration_days = (end_dates - start_dates).dt.days
    duration_months = duration_days / 30.44
    duration_months = duration_months.mask(duration_months < 0, np.nan)
    df['duration_months'] = duration_months
    df['event_observed'] = event_obs.astype(int)

    # Prepare variables
    bcr_col = detect_column(df, ['건폐율','건폐율(%)']) or '건폐율'
    pub_priv_col = detect_column(df, ['공공/민간']) or '공공/민간'
    bcr = to_numeric_safe(df[bcr_col]) if bcr_col in df.columns else pd.Series([np.nan]*len(df), index=df.index)

    # Significance checks
    # 1) Log-rank on duration
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        dd = df.dropna(subset=['duration_months'])
        g0 = dd.loc[dd['fast_flag']==0, ['duration_months','event_observed']]
        g1 = dd.loc[dd['fast_flag']==1, ['duration_months','event_observed']]
        lr_p = float('nan')
        if (len(g0)>0) and (len(g1)>0):
            lr = logrank_test(g1['duration_months'], g0['duration_months'], event_observed_A=g1['event_observed'], event_observed_B=g0['event_observed'])
            lr_p = float(lr.p_value)
    except Exception:
        lr_p = float('nan')

    # 2) Welch t-test on BCR
    b0 = bcr[df['fast_flag']==0]
    b1 = bcr[df['fast_flag']==1]
    _, bcr_p = welch_ttest(b1, b0)

    # 3) Chi-square on public/private distribution
    ct = pd.crosstab(df[pub_priv_col].fillna('(결측)'), df['fast_flag'])
    chi_p = chisq_p(ct)

    # Plot settings
    sns.set(style='whitegrid')
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rc('font', family='Malgun Gothic')  # Windows Malgun
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A) KM curves
    ax = axes[0]
    try:
        km0 = KaplanMeierFitter(label='NonFast')
        km1 = KaplanMeierFitter(label='Fast')
        if (len(g0)>0): km0.fit(g0['duration_months'], event_observed=g0['event_observed'])
        if (len(g1)>0): km1.fit(g1['duration_months'], event_observed=g1['event_observed'])
        km0.plot_survival_function(ax=ax, ci_show=True)
        km1.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title(f'소요기간 KM (로그랭크 p={lr_p:.3g})')
        ax.set_xlabel('개월'); ax.set_ylabel('생존확률')
    except Exception:
        ax.text(0.5, 0.5, 'KM 계산 불가', ha='center', va='center')
        ax.set_axis_off()

    # B) BCR box+strip
    ax = axes[1]
    tmp = pd.DataFrame({'건폐율': bcr, '그룹': np.where(df['fast_flag']==1, 'Fast', 'NonFast')}).dropna()
    if not tmp.empty:
        sns.boxplot(data=tmp, x='그룹', y='건폐율', ax=ax, showfliers=False)
        sns.stripplot(data=tmp, x='그룹', y='건폐율', ax=ax, color='black', alpha=0.4, jitter=True)
        ax.set_title(f'건폐율 (Welch p={bcr_p:.3g})')
        ax.set_xlabel(''); ax.set_ylabel('건폐율(%)')
    else:
        ax.text(0.5, 0.5, '건폐율 데이터 없음', ha='center', va='center')
        ax.set_axis_off()

    # C) Public/Private stacked percent bars
    ax = axes[2]
    dist = (ct.T / ct.T.sum(axis=1, min_count=1).values.reshape(-1,1) * 100).T  # categories x group
    if not dist.empty and dist.shape[1] > 0:
        groups = dist.columns.tolist()  # 0,1 order
        cats = dist.index.tolist()
        # Map group names
        disp = {'0': 'NonFast', '1': 'Fast'}
        gnames = ['NonFast' if g==0 else 'Fast' for g in groups]
        bottom = np.zeros(len(gnames))
        colors = sns.color_palette('Set2', n_colors=len(cats))
        for i, cat in enumerate(cats):
            vals = [dist.loc[cat, g] if g in dist.columns else 0 for g in groups]
            ax.bar(gnames, vals, bottom=bottom, label=str(cat), color=colors[i])
            bottom += np.array(vals)
        ax.set_ylim(0, 100)
        ax.set_ylabel('구성비(%)')
        ax.set_title(f'공공/민간 분포 (Chi2 p={chi_p:.3g})')
        ax.legend(title='카테고리', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.text(0.5, 0.5, '공공/민간 데이터 없음', ha='center', va='center')
        ax.set_axis_off()

    plt.tight_layout()
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    try:
        fig.savefig(str(outpath.with_suffix('.pdf')))
    except Exception:
        pass
    print('[SAVE]', outpath)


if __name__ == '__main__':
    main()
