from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


ENCODINGS_TRY = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


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
    if s.dtype.kind in ("i", "u", "f"):
        return s.astype(float)
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("㎥", "", regex=False)
        .str.replace("㎡", "", regex=False)
        .str.replace("m²", "", regex=False)
        .str.replace("m^2", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


POSITIVE_FLAG_VALUES = {"y","yes","1","true","t","예","유","신속","신속통합기획"}
EXP_REGEX = r'(?:신속통합|신통|선정구역)'


# 날짜 컬럼 동의어(생존분석 스크립트와 정합)
START_DATE_CANDIDATES = [
    '정비구역지정일','지정일','정비구역 지정일','정비구역지정고시일','지정고시일',
    '구역지정최초','구역지정변경(최종)','구역지정_최초','구역지정_변경(최종)'
]
EVENT_DATE_CANDIDATES = [
    '조합설립인가(사업시행자 지정일)','조합설립인가_최초','조합설립인가_변경(최종)','조합설립인가',
    '조합설립인가일','조합설립 인가일'
]


def detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def coalesce_dates(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """여러 날짜 후보 컬럼을 순서대로 결합하여 첫 유효값을 선택."""
    s = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    for c in candidates:
        if c in df.columns:
            s = s.combine_first(to_datetime_safe(df[c]))
    return s


def build_fast_flag(df: pd.DataFrame) -> pd.Series:
    # 1) 명시 플래그 컬럼 우선
    flag_cols = ["신속통합기획", "신속통합기획여부", "신속통합기획_구분"]
    present = [c for c in flag_cols if c in df.columns]
    if present:
        mask = pd.Series(False, index=df.index)
        for c in present:
            val = df[c].astype(str).str.strip().str.lower()
            mask = mask | val.isin(POSITIVE_FLAG_VALUES)
    else:
        mask = pd.Series(False, index=df.index)
    # 2) 텍스트 전체 검색 보조(정규식)
    if '_row_concat' not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        try:
            df['_row_concat'] = df[obj_cols].astype(str).agg(' '.join, axis=1)
        except Exception:
            df['_row_concat'] = ""
    regex_mask = df['_row_concat'].str.contains(EXP_REGEX, regex=True, na=False)
    return (mask | regex_mask).astype(int)


def welch_ttest(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    try:
        from scipy.stats import ttest_ind
        res = ttest_ind(x.dropna().astype(float), y.dropna().astype(float), equal_var=False)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def chisq_test(table: pd.DataFrame) -> Tuple[float, float]:
    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, exp = chi2_contingency(table.fillna(0).values)
        return float(chi2), float(p)
    except Exception:
        return float("nan"), float("nan")


def summarize_continuous(
    df: pd.DataFrame,
    group_col: str,
    cont_cols: List[str],
    zero_as_missing_cols: Optional[List[str]] = None,
    min_exclude_zero: bool = False,
) -> pd.DataFrame:
    rows = []
    for col in cont_cols:
        if col not in df.columns:
            continue
        s = to_numeric_safe(df[col])
        # 옵션: 특정 컬럼은 0을 결측으로 간주
        if zero_as_missing_cols and col in zero_as_missing_cols:
            s = s.mask(s == 0, np.nan)
        d = df.assign(__val=s)
        g0 = d.loc[d[group_col] == 0, "__val"].dropna()
        g1 = d.loc[d[group_col] == 1, "__val"].dropna()
        if g0.empty and g1.empty:
            continue
        t_stat, p_val = welch_ttest(g1, g0)
        def stats(x: pd.Series) -> dict:
            if x.empty:
                return {"n": 0, "mean": np.nan, "std": np.nan, "p50": np.nan, "p25": np.nan, "p75": np.nan, "min": np.nan, "max": np.nan}
            # 최소값 계산에서만 0을 제외하는 옵션
            if min_exclude_zero:
                x_for_min = x[x != 0]
            else:
                x_for_min = x
            min_val = float(np.nanmin(x_for_min)) if x_for_min.shape[0] > 0 else np.nan
            return {
                "n": int(x.shape[0]),
                "mean": float(np.nanmean(x)),
                "std": float(np.nanstd(x, ddof=1)) if x.shape[0] > 1 else np.nan,
                "p50": float(np.nanpercentile(x, 50)),
                "p25": float(np.nanpercentile(x, 25)),
                "p75": float(np.nanpercentile(x, 75)),
                "min": min_val,
                "max": float(np.nanmax(x)),
            }
        s0 = stats(g0)
        s1 = stats(g1)
        rows.append({
            "variable": col,
            "group": "Fast",
            **{f"fast_{k}": v for k, v in s1.items()},
            **{f"nonfast_{k}": v for k, v in s0.items()},
            "t_stat": t_stat,
            "p_value": p_val,
        })
    out = pd.DataFrame(rows)
    return out


def summarize_categorical(df: pd.DataFrame, group_col: str, cat_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        # 상위 카테고리 수 과도 시(>100) 스킵
        nunique = df[col].nunique(dropna=False)
        if nunique == 0 or nunique > 100:
            continue
        # 교차표
        ct = pd.crosstab(df[col].fillna("(결측)"), df[group_col], dropna=False)
        # 카이제곱 p값
        _, p_val = chisq_test(ct)
        total_fast = ct.get(1, pd.Series(dtype=int)).sum()
        total_nonfast = ct.get(0, pd.Series(dtype=int)).sum()
        for level, row in ct.iterrows():
            n_fast = int(row.get(1, 0))
            n_nonfast = int(row.get(0, 0))
            pct_fast = (n_fast / total_fast) * 100 if total_fast > 0 else np.nan
            pct_nonfast = (n_nonfast / total_nonfast) * 100 if total_nonfast > 0 else np.nan
            rows.append({
                "variable": col,
                "category": level,
                "n_fast": n_fast,
                "pct_fast": pct_fast,
                "n_nonfast": n_nonfast,
                "pct_nonfast": pct_nonfast,
                "p_value": p_val,
            })
    out = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fast vs NonFast 베이스라인(2021+) 요약 생성")
    ap.add_argument("--input", required=True, help="입력 CSV 경로")
    ap.add_argument("--outdir", default=str(Path("outputs") / "analysis"), help="출력 폴더")
    ap.add_argument("--min-start-year", type=int, default=2021, help="구역지정최초 기준 최소 연도 필터")
    ap.add_argument("--cutoff-date", type=str, default=datetime.today().strftime('%Y-%m-%d'), help="사건 미발생시 소요기간 계산용 컷오프 날짜(YYYY-MM-DD), 기본: 오늘")
    ap.add_argument("--zero-as-missing-cols", type=str, default="", help="0을 결측으로 처리할 컬럼명을 콤마로 구분하여 지정")
    ap.add_argument("--min-exclude-zero", action="store_true", help="최소값(및 범위) 계산에서 0을 제외")
    ap.add_argument("--no-median", action="store_true", help="중앙값 및 IQR 표기를 출력에서 제외")
    ap.add_argument("--preview-rows", type=int, default=100, help="연속형(논문용 포맷) 미리보기 최대 행 수")
    ap.add_argument("--preview-include", type=str, default="", help="미리보기에서 반드시 포함할 변수명(쉼표구분, 지정 순서 우선)")
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv_smart(src)

    # Fast 플래그
    df["fast_flag"] = build_fast_flag(df)

    # 시작일/사건일 병합 시리즈 생성(생존분석과 정합)
    start_dates_all = coalesce_dates(df, START_DATE_CANDIDATES)
    event_dates_all_full = coalesce_dates(df, EVENT_DATE_CANDIDATES)

    # 2021+ 필터: 시작일 기준(유효한 날짜 중 연도 필터 통과 건만 유지)
    if not start_dates_all.isna().all():
        mask_year = start_dates_all.dt.year >= int(args.min_start_year)
        df = df.loc[mask_year].copy()
        start_dates_all = start_dates_all.loc[df.index]
        event_dates_all_full = event_dates_all_full.loc[df.index]
    else:
        # 시작일 전부 결측이면 필터 미적용
        start_dates_all = start_dates_all.loc[df.index]
        event_dates_all_full = event_dates_all_full.loc[df.index]

    # 컷오프/사건일 정리 및 소요기간(개월) 계산
    try:
        cutoff_dt = pd.to_datetime(args.cutoff_date)
    except Exception:
        cutoff_dt = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    event_dates_all = event_dates_all_full
    # 사건이 컷오프 이전/동일이면 사건, 아니면 검열
    event_obs = event_dates_all.notna() & (event_dates_all <= cutoff_dt)
    end_dates = pd.to_datetime(np.where(event_obs, event_dates_all, cutoff_dt))
    duration_days = (end_dates - start_dates_all).dt.days
    DAYS_PER_MONTH = 30.44
    duration_months = duration_days / DAYS_PER_MONTH
    # 음수는 데이터 오류 가능성 -> NaN 처리
    duration_months = duration_months.mask(duration_months < 0, np.nan)
    # 컬럼 추가
    df["소요기간(개월)"] = duration_months
    df["__event_observed"] = event_obs.astype(float)
    # 그룹 크기 보고용
    n_fast = int((df["fast_flag"] == 1).sum())
    n_nonfast = int((df["fast_flag"] == 0).sum())

    # 연속형/범주형 후보 컬럼
    cont_candidates = [
        "소요기간(개월)",
        "정비구역면적(㎡)", "택지면적(㎡)", "도로면적(㎡)", "공원면적(㎡)", "녹지면적(㎡)",
        "공공공지면적(㎡)", "학교면적(㎡)", "기타면적(㎡)", "토지등 소유자 수", "세대총합계",
        "분양세대총수", "임대세대총수", "60㎡이하", "60㎡초과~85㎡이하", "85㎡초과",
        "(임대)40㎡이하", "(임대)40㎡초과~50㎡이하", "(임대)50㎡초과", "건폐율", "용적률",
        "높이(m)", "지상층수", "지하층수",
    ]
    cat_candidates = [
        "자치구", "공공/민간", "일반/재촉지구", "주용도", "용도지역", "용도지구",
        "진행단계", "진행단계_군집",
    ]
    # CSV 컬럼 순서 유지: 원본 CSV 컬럼 순서대로 후보 교집합을 취함
    cont_set = set(cont_candidates)
    cat_set = set(cat_candidates)
    cont_cols = [c for c in df.columns if c in cont_set]
    cat_cols = [c for c in df.columns if c in cat_set]

    zero_as_missing_cols = [c.strip() for c in args.zero_as_missing_cols.split(",") if c.strip()]
    cont_summary = summarize_continuous(
        df,
        group_col="fast_flag",
        cont_cols=cont_cols,
        zero_as_missing_cols=zero_as_missing_cols,
        min_exclude_zero=bool(args.min_exclude_zero),
    )
    # 중앙값/IQR 제외 시 관련 컬럼 제거
    if args.no_median and not cont_summary.empty:
        drop_cols = [
            "fast_p50","fast_p25","fast_p75",
            "nonfast_p50","nonfast_p25","nonfast_p75",
        ]
        cont_summary = cont_summary[[c for c in cont_summary.columns if c not in drop_cols]]

    # CSV 컬럼 순서에 따른 정렬용 맵
    var_order = {v: i for i, v in enumerate(cont_cols)}

    # 연속형/범주형 요약 생성
    cat_summary = summarize_categorical(df, group_col="fast_flag", cat_cols=cat_cols)

    # 연속형: 변수 순서 정렬(CSV 순)
    if not cont_summary.empty:
        cont_summary['__ord'] = cont_summary['variable'].map(lambda v: var_order.get(v, 10**9))
        cont_summary = cont_summary.sort_values(['__ord','variable']).drop(columns='__ord')

    # 메타 정보 시트도 함께 저장
    meta = pd.DataFrame({
        "metric": ["N_fast", "N_nonfast", "N_total"],
        "value": [n_fast, n_nonfast, n_fast + n_nonfast],
    })

    # 저장
    out_cont = outdir / "baseline_fast_continuous.csv"
    out_cat = outdir / "baseline_fast_categorical.csv"
    out_meta = outdir / "baseline_fast_meta.csv"
    cont_summary.to_csv(out_cont, index=False, encoding="utf-8-sig")
    cat_summary.to_csv(out_cat, index=False, encoding="utf-8-sig")
    meta.to_csv(out_meta, index=False, encoding="utf-8-sig")

    # 논문용 포맷(연속형): mean±sd / median[IQR] / range / overall mean
    def fmt(v: Optional[float], nd: int = 2) -> str:
        return (f"{v:.{nd}f}" if pd.notna(v) else "-")

    formatted_rows = []
    if not cont_summary.empty:
        for _, r in cont_summary.iterrows():
            var = r["variable"]
            # overall mean
            overall = np.nan
            if var in df.columns:
                overall = np.nanmean(to_numeric_safe(df[var]))
            fast_mean_sd = f"{fmt(r.get('fast_mean'))} ± {fmt(r.get('fast_std'))}"
            nonfast_mean_sd = f"{fmt(r.get('nonfast_mean'))} ± {fmt(r.get('nonfast_std'))}"
            fast_median_iqr = f"{fmt(r.get('fast_p50'))} [{fmt(r.get('fast_p25'))}–{fmt(r.get('fast_p75'))}]" if not args.no_median else None
            nonfast_median_iqr = f"{fmt(r.get('nonfast_p50'))} [{fmt(r.get('nonfast_p25'))}–{fmt(r.get('nonfast_p75'))}]" if not args.no_median else None
            fast_range = f"{fmt(r.get('fast_min'))}–{fmt(r.get('fast_max'))}"
            nonfast_range = f"{fmt(r.get('nonfast_min'))}–{fmt(r.get('nonfast_max'))}"
            row = {
                "variable": var,
                "Fast (mean±SD)": fast_mean_sd,
                "NonFast (mean±SD)": nonfast_mean_sd,
                "Fast (range)": fast_range,
                "NonFast (range)": nonfast_range,
                "overall_mean": overall,
                "p_value": r.get("p_value"),
            }
            if not args.no_median:
                row.update({
                    "Fast (median [IQR])": fast_median_iqr,
                    "NonFast (median [IQR])": nonfast_median_iqr,
                })
            formatted_rows.append(row)
    cont_formatted = pd.DataFrame(formatted_rows)
    if not cont_formatted.empty:
        cont_formatted['__ord'] = cont_formatted['variable'].map(lambda v: var_order.get(v, 10**9))
        cont_formatted = cont_formatted.sort_values(['__ord','variable']).drop(columns='__ord')
    out_cont_fmt = outdir / "baseline_fast_continuous_formatted.csv"
    cont_formatted.to_csv(out_cont_fmt, index=False, encoding="utf-8-sig")

    # Excel 통합본(옵션)
    try:
        with pd.ExcelWriter(outdir / "baseline_fast.xlsx", engine="xlsxwriter") as xl:
            meta.to_excel(xl, sheet_name="meta", index=False)
            cont_summary.to_excel(xl, sheet_name="continuous_raw", index=False)
            cont_formatted.to_excel(xl, sheet_name="continuous_formatted", index=False)
            cat_summary.to_excel(xl, sheet_name="categorical", index=False)
    except Exception:
        pass

    # 생존비교(KM/로그랭크) 요약 준비
    surv_md = ""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        dd = df.dropna(subset=["소요기간(개월)"])
        g0 = dd.loc[dd["fast_flag"]==0, ["소요기간(개월)", "__event_observed"]]
        g1 = dd.loc[dd["fast_flag"]==1, ["소요기간(개월)", "__event_observed"]]
        if (len(g0)>0) and (len(g1)>0):
            # 로그랭크
            lr = logrank_test(g1["소요기간(개월)"], g0["소요기간(개월)"], event_observed_A=g1["__event_observed"], event_observed_B=g0["__event_observed"])
            lr_p = float(lr.p_value) if hasattr(lr, 'p_value') else float('nan')
            # KM 중앙값(도달 불가 시 NaN)
            km0 = KaplanMeierFitter(label='NonFast'); km0.fit(g0["소요기간(개월)"], event_observed=g0["__event_observed"])
            km1 = KaplanMeierFitter(label='Fast'); km1.fit(g1["소요기간(개월)"], event_observed=g1["__event_observed"])
            med0_raw = km0.median_survival_time_ if getattr(km0, 'median_survival_time_', None) is not None else np.nan
            med1_raw = km1.median_survival_time_ if getattr(km1, 'median_survival_time_', None) is not None else np.nan
            def fmt_med(v):
                try:
                    v = float(v)
                    if not np.isfinite(v):
                        return 'NR'
                    return f"{v:.2f}"
                except Exception:
                    return '-'
            med0 = fmt_med(med0_raw)
            med1 = fmt_med(med1_raw)
            n0 = int(len(g0)); e0 = int(g0["__event_observed"].sum())
            n1 = int(len(g1)); e1 = int(g1["__event_observed"].sum())
            surv_md = (
                "## 소요기간(생존분석 비교)\n\n"
                f"- 표본: Fast n={n1} (사건 {e1}), NonFast n={n0} (사건 {e0})\n"
                f"- KM 중앙값[개월](NR=도달 안 함): Fast={med1}, NonFast={med0}\n"
                f"- 로그랭크 p값: {lr_p:.6g}\n\n"
            )
    except Exception:
        surv_md = ""

    # 간단 마크다운 요약
    md = outdir / "baseline_fast_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Fast vs NonFast 베이스라인 요약 (" + str(args.min_start_year) + "+)\n\n")
        f.write(f"표본 크기: Fast={n_fast}, NonFast={n_nonfast}, 총합={n_fast + n_nonfast}\n\n")
        if not cont_summary.empty:
            if args.no_median:
                f.write("## 연속형 요약 (평균±표준편차, 최솟값–최댓값) 및 Welch t-검정 p값\n\n")
                cols = [
                    "variable",
                    "fast_n","fast_mean","fast_std","fast_min","fast_max",
                    "nonfast_n","nonfast_mean","nonfast_std","nonfast_min","nonfast_max",
                    "p_value",
                ]
            else:
                f.write("## 연속형 요약 (평균±표준편차, 중앙값[IQR], 최솟값–최댓값) 및 Welch t-검정 p값\n\n")
                cols = [
                    "variable",
                    "fast_n","fast_mean","fast_std","fast_p50","fast_p25","fast_p75","fast_min","fast_max",
                    "nonfast_n","nonfast_mean","nonfast_std","nonfast_p50","nonfast_p25","nonfast_p75","nonfast_min","nonfast_max",
                    "p_value",
                ]
            show = [c for c in cols if c in cont_summary.columns]
            try:
                f.write(cont_summary[show].to_markdown(index=False))
            except Exception:
                f.write(cont_summary[show].to_csv(index=False))
            f.write("\n\n")
            if zero_as_missing_cols or args.min_exclude_zero:
                note = []
                if zero_as_missing_cols:
                    note.append(f"다음 컬럼은 0을 결측으로 처리: {', '.join(zero_as_missing_cols)}")
                if args.min_exclude_zero:
                    note.append("최소값(및 범위) 계산에서 0을 제외")
                f.write("참고: " + "; ".join(note) + "\n\n")
        if not cat_summary.empty:
            f.write("## 범주형 분포 및 카이제곱 p값\n\n")
            # 각 변수별 상위 10개 카테고리만 미리보기
            preview = (
                cat_summary
                .sort_values(["variable", "n_fast"], ascending=[True, False])
                .groupby("variable")
                .head(10)
            )
            try:
                f.write(preview[["variable", "category", "n_fast", "pct_fast", "n_nonfast", "pct_nonfast", "p_value"]].to_markdown(index=False))
            except Exception:
                f.write(preview[["variable", "category", "n_fast", "pct_fast", "n_nonfast", "pct_nonfast", "p_value"]].to_csv(index=False))

        # 논문용 연속형 포맷 미리보기
        if not cont_formatted.empty:
            f.write("\n\n## 연속형(논문용 포맷) 미리보기\n\n")
            preview2 = cont_formatted.copy()
            # 포함 우선 변수 재정렬
            include_vars = [v.strip() for v in args.preview_include.split(',') if v.strip()]
            if include_vars:
                order_map = {v: i for i, v in enumerate(include_vars)}
                preview2 = preview2.copy()
                preview2['__ord_inc'] = preview2['variable'].map(lambda v: order_map.get(v, len(include_vars)))
                preview2['__ord_csv'] = preview2['variable'].map(lambda v: var_order.get(v, 10**9))
                preview2 = preview2.sort_values(['__ord_inc','__ord_csv']).drop(columns=['__ord_inc','__ord_csv'])
            else:
                # CSV 순서 정렬
                preview2['__ord_csv'] = preview2['variable'].map(lambda v: var_order.get(v, 10**9))
                preview2 = preview2.sort_values(['__ord_csv']).drop(columns=['__ord_csv'])
            try:
                f.write(preview2.head(args.preview_rows).to_markdown(index=False))
            except Exception:
                f.write(preview2.head(args.preview_rows).to_csv(index=False))

        # 생존비교(KM/로그랭크) 섹션 추가
        if surv_md:
            f.write("\n\n" + surv_md)

    print(f"저장: {out_cont}, {out_cat}, {out_meta}, {out_cont_fmt}, {md}")


if __name__ == "__main__":
    main()
