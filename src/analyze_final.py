from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
from typing import Optional, Tuple, List

import pandas as pd


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"CSV 읽기 실패: {path}")


def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_phase_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        # 기준/변형 일반 케이스
        "사업추진단계", "사업단계", "추진단계", "진행단계", "상태", "진행상태", "사업상태",
        # 외부 데이터 접미어 케이스
        "진행단계_열린데이터", "상태_열린데이터", "사업단계_열린데이터",
        "사업단계_정비몽땅", "진행단계_정비몽땅", "상태_정비몽땅",
    ]
    return pick_col(df, candidates)


def counts_by_gu_type(df: pd.DataFrame, gu_col: str, type_col: str) -> pd.DataFrame:
    return df.groupby([gu_col, type_col], dropna=False).size().reset_index(name="count").sort_values([gu_col, type_col])


def counts_by_gu_type_phase(df: pd.DataFrame, gu_col: str, type_col: str, phase_col: str) -> pd.DataFrame:
    return (
        df.groupby([gu_col, type_col, phase_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([gu_col, type_col, phase_col])
    )


# ---------------- 단계 라벨 군집화 -----------------
PHASE_CLUSTER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"추진위|추진위원"), "추진위"),
    (re.compile(r"조합설립"), "조합설립"),  # 조합설립인가 포함
    (re.compile(r"사업시행"), "사업시행"),  # 사업시행인가 포함
    (re.compile(r"관리처분"), "관리처분"),  # 관리처분인가 포함
    (re.compile(r"건축심의|건축.?위원"), "건축심의"),
    (re.compile(r"구역지정|정비구역|지정"), "구역지정"),
    (re.compile(r"착공"), "착공"),
    (re.compile(r"준공|완료|사용승인"), "준공"),
]


def normalize_phase_label(val: object) -> object:
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return s
    for pattern, label in PHASE_CLUSTER_PATTERNS:
        if pattern.search(s):
            return label
    return "기타"


def add_phase_cluster_column(df: pd.DataFrame, phase_col: str) -> tuple[pd.DataFrame, str]:
    """주어진 단계 컬럼을 군집화하여 신규 컬럼(<원본>_군집) 추가 후 반환.

    Returns: (새 DF, 신규 컬럼명)
    """
    new_col = f"{phase_col}_군집"
    if new_col in df.columns:
        return df, new_col
    df2 = df.copy()
    df2[new_col] = df2[phase_col].map(normalize_phase_label)
    return df2, new_col


def save_excel_summary(out_path: Path, tbl1: pd.DataFrame, tbl2: Optional[pd.DataFrame], gu_col: str, type_col: str, phase_col: Optional[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xl:
        tbl1.to_excel(xl, sheet_name="by_gu_type", index=False)
        # 피벗(자치구 x 사업유형)
        try:
            pv1 = tbl1.pivot(index=gu_col, columns=type_col, values="count").fillna(0).astype(int)
            pv1.loc["합계"] = pv1.sum(axis=0)
            pv1["소계"] = pv1.sum(axis=1)
            pv1.to_excel(xl, sheet_name="pivot_gu_type")
        except Exception:
            pass
        if tbl2 is not None and phase_col:
            tbl2.to_excel(xl, sheet_name="by_gu_type_phase", index=False)
            # 간단 피벗(자치구 x 사업유형, 값=합계) - 단계는 필터링용 원본 시트 참조
            try:
                pv2 = (
                    tbl2.groupby([gu_col, type_col])["count"].sum().unstack(fill_value=0).astype(int)
                )
                pv2.loc["합계"] = pv2.sum(axis=0)
                pv2["소계"] = pv2.sum(axis=1)
                pv2.to_excel(xl, sheet_name="pivot_gu_type_total")
            except Exception:
                pass


NOISE_TOKENS = ["일대", "일원", "주변", "부근", "인근", "일부", "전역", "전체"]


def _clean_raw_location(raw: str, gu: Optional[str]) -> str:
    # 괄호 내용 제거, 콤마/중점 등 구분자는 공백으로
    s = re.sub(r"[()\[\]]", " ", raw)
    s = re.sub(r"[·ㆍ/,]", " ", s)
    # 노이즈 토큰 제거
    for t in NOISE_TOKENS:
        s = s.replace(t, " ")
    s = re.sub(r"\s+", " ", s).strip()
    if gu:
        s = f"서울특별시 {gu} {s}"
    # 다중 번지 → 첫 번째 번지로 축약
    # 1) 123-45 형태 우선
    m = re.search(r"산?\d+(?:-\d+)?", s)
    if m:
        head = s[: m.end()]
        return head
    # 2) 숫자만(예: 199 203 206) → 첫 숫자만 유지
    m2 = re.search(r"\b\d+\b", s)
    if m2:
        head = s[: m2.end()]
        return head
    return s


def build_address(row: pd.Series, gu_col: str, addr_col: Optional[str]) -> Optional[str]:
    # 표준 파생 키로 주소 구성(가급적 일관된 형식 유지)
    parts = ["서울특별시"]
    gu = row.get(gu_col)
    if isinstance(gu, str) and gu:
        parts.append(gu)
    dong = row.get("동_std")
    if isinstance(dong, str) and dong:
        parts.append(dong)
    bunji = row.get("번지_std")
    if isinstance(bunji, str) and bunji:
        parts.append(bunji)
    built = " ".join(parts) if len(parts) >= 2 else None

    # 원본 위치가 있다면 정제하여 보조로 사용(노이즈 제거 + 번지 축약)
    if addr_col and isinstance(row.get(addr_col), str) and row.get(addr_col).strip():
        raw = row.get(addr_col)
        raw = _clean_raw_location(raw, gu if isinstance(gu, str) else None)
        # 원본이 더 풍부할 수 있으니, 표준 조합이 없을 때는 원본 정리본 사용
        if not built:
            built = raw
    return built


def _generate_query_candidates(row: pd.Series, gu_col: str, addr_col: Optional[str]) -> List[str]:
    cands: List[str] = []
    gu = row.get(gu_col)
    gu = gu if isinstance(gu, str) else None
    dong = row.get("동_std") if isinstance(row.get("동_std"), str) else None
    bunji = row.get("번지_std") if isinstance(row.get("번지_std"), str) else None
    # 1) 표준 주소(동+번지)
    parts = ["서울특별시"]
    if gu: parts.append(gu)
    if dong: parts.append(dong)
    if bunji: parts.append(bunji)
    if len(parts) >= 2:
        cands.append(" ".join(parts))
    # 2) 표준(동만)
    parts2 = ["서울특별시"]
    if gu: parts2.append(gu)
    if dong: parts2.append(dong)
    if len(parts2) >= 2:
        cands.append(" ".join(parts2))
    # 3) 원본 위치 정제판
    if addr_col and isinstance(row.get(addr_col), str) and row.get(addr_col).strip():
        cands.append(_clean_raw_location(row.get(addr_col), gu))
    # 4) 자치구만
    parts3 = ["서울특별시"]
    if gu: parts3.append(gu)
    cands.append(" ".join(parts3))
    # 중복 제거, 빈 문자열 제외
    out: List[str] = []
    for q in cands:
        q2 = re.sub(r"\s+", " ", q or "").strip()
        if q2 and (q2 not in out):
            out.append(q2)
    return out


def geocode_addresses(
    df: pd.DataFrame,
    gu_col: str,
    addr_col: Optional[str],
    cache_path: Path,
    provider: str = "nominatim",
    rate_limit_sec: float = 1.1,
) -> pd.DataFrame:
    """간단 지오코딩(캐시 지원). 기본은 OSM Nominatim 사용.

    네트워크 사용이 불가한 환경에서는 캐시가 있는 경우에만 좌표가 채워집니다.
    """
    # 지오코딩 비활성화 환경변수 지원
    if os.environ.get("DISABLE_GEOCODING", "").lower() in {"1", "true", "yes"}:
        print("지오코딩 비활성화: DISABLE_GEOCODING 환경변수 설정됨")
        return df

    cache: pd.DataFrame
    if cache_path.exists():
        cache = read_csv_smart(cache_path)
    else:
        cache = pd.DataFrame(columns=["query", "lat", "lon"])  # empty

    cache = cache.drop_duplicates(subset=["query"]).set_index("query")

    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
    except Exception:
        print("경고: geopy가 설치되어 있지 않아 지오코딩을 건너뜁니다. pip install geopy 필요")
        return df

    geolocator = Nominatim(user_agent="urban-refurb-analysis")
    geocode = RateLimiter(lambda q: geolocator.geocode(q, timeout=10), min_delay_seconds=rate_limit_sec, max_retries=2)

    lats, lons = [], []
    statuses: list[str] = []
    queries: list[Optional[str]] = []
    new_cache_rows = []
    for _, row in df.iterrows():
        tried = False
        success = False
        used_q: Optional[str] = None
        lat = pd.NA
        lon = pd.NA
        status = "no_address"
        for i, q in enumerate(_generate_query_candidates(row, gu_col=gu_col, addr_col=addr_col)):
            tried = True
            used_q = q
            # 캐시 먼저
            if q in cache.index:
                r = cache.loc[q]
                lat, lon = r.get("lat"), r.get("lon")
                if pd.notna(lat) and pd.notna(lon):
                    status = "cache_ok" if i == 0 else f"cache_ok_fallback{i}"
                    success = True
                    break
                else:
                    status = "cache_na"
                    # 계속 다음 후보 시도
                    continue
            # 호출 시도
            try:
                loc = geocode(q)
                if loc is not None:
                    lat, lon = loc.latitude, loc.longitude
                    status = "ok" if i == 0 else f"ok_fallback{i}"
                    success = True
                else:
                    lat, lon = (pd.NA, pd.NA)
                    status = "no_result"
            except Exception:
                lat, lon = (pd.NA, pd.NA)
                status = "error"
            # 캐시에 기록
            new_cache_rows.append({"query": q, "lat": lat, "lon": lon})
            if success:
                break
        # 최종 기록
        queries.append(used_q)
        lats.append(lat)
        lons.append(lon)
        if not tried:
            status = "no_address"
        statuses.append(status)

    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    df["geocode_query"] = queries
    df["geocode_status"] = statuses

    # 캐시 저장
    if new_cache_rows:
        updated = pd.concat([cache.reset_index(), pd.DataFrame(new_cache_rows)], ignore_index=True)
        updated = updated.drop_duplicates(subset=["query"], keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"지오코딩 캐시 저장: {cache_path} (+{len(new_cache_rows)}건)")

    return df


def _overlay_gu_boundaries(m, geojson_path: Path, name_field: Optional[str] = None) -> None:
    import json
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # 후보 필드 중 존재하는 이름 필드를 선택
    candidates = [name_field] if name_field else []
    candidates += ["SIG_KOR_NM", "ADM_NM", "gu_name", "name", "자치구"]
    def pick_name(props: dict) -> str:
        for k in candidates:
            if k and k in props and props[k]:
                return str(props[k])
        # 아무 것도 없으면 첫 키의 값을 표시
        return str(next(iter(props.values()))) if props else ""

    import folium
    def style_fn(_):
        return {"color": "#333333", "weight": 2, "fill": False}

    folium.GeoJson(
        gj,
        name="gu-boundaries",
        show=True,
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(fields=[], aliases=[], labels=False),
        highlight_function=lambda x: {"weight": 4, "color": "#000"},
        popup=folium.features.GeoJsonPopup(fields=[], aliases=[], labels=False),
    ).add_to(m)

    # 라벨을 별도 레이어로 추가(중심점에 텍스트 마커 배치)
    try:
        from shapely.geometry import shape  # type: ignore
    except Exception:
        shape = None  # type: ignore

    if shape is not None:  # type: ignore
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry")
            try:
                g = shape(geom)
                c = g.representative_point().coords[0]
                folium.map.Marker(
                    location=[c[1], c[0]],
                    icon=folium.DivIcon(html=f'<div style="font-size:10px;color:#444;background:rgba(255,255,255,0.6);padding:1px 3px;border-radius:3px;">{pick_name(props)}</div>'),
                ).add_to(m)
            except Exception:
                continue


def _overlay_city_boundary(m, geojson_path: Path, name_field: Optional[str] = None) -> None:
    import json
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    import folium
    def style_fn(_):
        return {"color": "#0055cc", "weight": 3, "fill": False}
    folium.GeoJson(
        gj,
        name="seoul-boundary",
        show=True,
        style_function=style_fn,
    ).add_to(m)


def save_map(df: pd.DataFrame, out_html: Path, type_col: str, gu_geojson: Optional[Path] = None, gu_name_field: Optional[str] = None, city_geojson: Optional[Path] = None, city_name_field: Optional[str] = None) -> None:
    try:
        import folium
    except Exception:
        print("경고: folium이 설치되어 있지 않아 지도를 건너뜁니다. pip install folium 필요")
        return

    # 중심을 서울 시청 좌표로 설정
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")
    projects_fg = folium.FeatureGroup(name="projects", show=True)

    color_by_type = {
        "재개발": "#1f77b4",  # blue
        "재건축": "#d62728",  # red
    }

    added = 0
    for _, r in df.iterrows():
        lat, lon = r.get("lat"), r.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        t = r.get(type_col, "")
        color = color_by_type.get(str(t), "#2ca02c")
        popup = []
        for key in ("자치구", "구역명", type_col, "사업단계", "위치"):
            if key in df.columns and pd.notna(r.get(key)):
                popup.append(f"{key}: {r.get(key)}")
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup("<br>".join(popup) if popup else None, max_width=400),
        ).add_to(projects_fg)
        added += 1

    projects_fg.add_to(m)
    # 경계 오버레이(선택)
    if gu_geojson and Path(gu_geojson).exists():
        try:
            _overlay_gu_boundaries(m, Path(gu_geojson), gu_name_field)
            print(f"자치구 경계 오버레이: {gu_geojson}")
        except Exception as e:
            print(f"경고: 자치구 경계 오버레이 실패: {e}")

    if city_geojson and Path(city_geojson).exists():
        try:
            _overlay_city_boundary(m, Path(city_geojson), city_name_field)
            print(f"서울시 경계 오버레이: {city_geojson}")
        except Exception as e:
            print(f"경고: 서울시 경계 오버레이 실패: {e}")

    try:
        import folium
        folium.LayerControl(collapsed=True).add_to(m)
    except Exception:
        pass

    m.save(str(out_html))
    print(f"지도 저장: {out_html} (마커 {added}개)")


def main():
    ap = argparse.ArgumentParser(description="merged_final.csv 분석 및 시각화")
    ap.add_argument("--input", default=str(Path("outputs") / "merged_final.csv"))
    ap.add_argument("--outdir", default=str(Path("outputs") / "analysis"))
    ap.add_argument("--geocode", action="store_true", help="지오코딩 수행(캐시 사용)")
    ap.add_argument("--gu-geojson", help="자치구 경계 GeoJSON 경로(지도로 오버레이)")
    ap.add_argument("--gu-name-field", help="GeoJSON 속성 중 자치구명 필드명(예: SIG_KOR_NM)")
    ap.add_argument("--city-geojson", help="서울시(시계) 경계 GeoJSON 경로(지도로 오버레이)")
    ap.add_argument("--city-name-field", help="GeoJSON 속성 중 시명 필드명(선택)")
    ap.add_argument("--no-phase-cluster", action="store_true", help="사업추진단계 군집화(정규화) 비활성화")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = read_csv_smart(input_path)

    # 기준 컬럼 선택(원본 우선, 없으면 표준)
    gu_col = pick_col(df, ["자치구", "자치구_std"]) or "자치구"
    type_col = pick_col(df, ["사업유형", "사업유형_std"]) or "사업유형"
    # 입력에 사업유형 컬럼이 전혀 없는 경우(예: 단일 유형 표본) 안전하게 대체 컬럼 생성
    if type_col not in df.columns:
        fallback_type_col = "사업유형"
        if fallback_type_col not in df.columns:
            df[fallback_type_col] = "전체"
        type_col = fallback_type_col
    phase_col = find_phase_column(df)
    addr_col = pick_col(df, ["위치"])  # 원본 위치 컬럼 있으면 사용

    # 1) 자치구, 사업유형 별 건수
    tbl1 = counts_by_gu_type(df, gu_col, type_col)
    tbl1_path = outdir / "counts_by_gu_type.csv"
    tbl1.to_csv(tbl1_path, index=False, encoding="utf-8-sig")
    print(f"저장: {tbl1_path} → {tbl1.shape}")

    # 지도: 선택적으로 지오코딩 실시
    if args.geocode:
        cache_path = Path("outputs") / "geocoding_cache.csv"
        df_geo = geocode_addresses(df, gu_col=gu_col, addr_col=addr_col, cache_path=cache_path)
        # 실패 사례 진단 저장
        fail_mask = df_geo["lat"].isna() | df_geo["lon"].isna()
        failures = df_geo.loc[fail_mask, [
            c for c in ["자치구", "구역명", "위치", gu_col, "동_std", "번지_std", "geocode_query", "geocode_status"] if c in df_geo.columns
        ]]
        fail_path = outdir / "geocoding_failures.csv"
        failures.to_csv(fail_path, index=False, encoding="utf-8-sig")
        n_total = len(df_geo)
        n_fail = int(fail_mask.sum())
        n_ok = n_total - n_fail
        print(f"지오코딩 요약: 총 {n_total}건, 성공 {n_ok}건, 실패 {n_fail}건 → {fail_path}")

        save_map(
            df_geo,
            outdir / "map_seoul_projects.html",
            type_col=type_col,
            gu_geojson=Path(args.gu_geojson) if args.gu_geojson else None,
            gu_name_field=args.gu_name_field,
            city_geojson=Path(args.city_geojson) if args.city_geojson else None,
            city_name_field=args.city_name_field,
        )
    else:
        print("지오코딩 스킵: --geocode 플래그 미설정")

    # 2) 자치구, 사업유형, 사업단계 별 건수 (원본 단계)
    tbl2 = None
    phase_cluster_col = None
    if phase_col is not None:
        tbl2 = counts_by_gu_type_phase(df, gu_col, type_col, phase_col)
        tbl2_path = outdir / "counts_by_gu_type_phase.csv"
        tbl2.to_csv(tbl2_path, index=False, encoding="utf-8-sig")
        print(f"저장: {tbl2_path} → {tbl2.shape}")

        # 2-1) 군집화 컬럼 추가(기본 활성)
        if not args.no_phase_cluster:
            df, phase_cluster_col = add_phase_cluster_column(df, phase_col)
            tbl2_cluster = counts_by_gu_type_phase(df, gu_col, type_col, phase_cluster_col)
            tbl2c_path = outdir / "counts_by_gu_type_phase_cluster.csv"
            tbl2_cluster.to_csv(tbl2c_path, index=False, encoding="utf-8-sig")
            print(f"저장: {tbl2c_path} → {tbl2_cluster.shape}")
    else:
        print("주의: 사업단계(혹은 유사) 컬럼을 찾지 못했습니다. 후보 컬럼명을 확인하세요.")

    # ---- Pivot / Summary Tables ----
    try:
        # 1) 자치구 x 사업유형 피벗
        pivot1 = tbl1.pivot_table(index=gu_col, columns=type_col, values="count", aggfunc="sum", fill_value=0)
        pivot1["합계"] = pivot1.sum(axis=1)
        total_row = pivot1.sum().to_frame().T
        total_row.index = ["총합"]
        pivot1_full = pd.concat([pivot1, total_row])
        print("\n[자치구 x 사업유형 표]\n", pivot1_full)

        # 2) 자치구 x (사업유형, 단계) 다중 피벗 (가능한 경우)
        pivot2_full = None
        pivot2c_full = None
        if phase_col is not None and tbl2 is not None:
            pivot2 = tbl2.pivot_table(index=gu_col, columns=[type_col, phase_col], values="count", aggfunc="sum", fill_value=0)
            pivot2["합계"] = pivot2.sum(axis=1)
            total2 = pivot2.sum().to_frame().T
            total2.index = ["총합"]
            pivot2_full = pd.concat([pivot2, total2])
            print("\n[자치구 x 사업유형 x 단계 표]\n", pivot2_full)
        if phase_cluster_col is not None:
            # 군집화 단계 피벗
            tbl2c = counts_by_gu_type_phase(df, gu_col, type_col, phase_cluster_col)
            pivot2c = tbl2c.pivot_table(index=gu_col, columns=[type_col, phase_cluster_col], values="count", aggfunc="sum", fill_value=0)
            pivot2c["합계"] = pivot2c.sum(axis=1)
            total2c = pivot2c.sum().to_frame().T
            total2c.index = ["총합"]
            pivot2c_full = pd.concat([pivot2c, total2c])
            print("\n[자치구 x 사업유형 x 단계(군집) 표]\n", pivot2c_full)

        # 3) 마크다운 저장
        md_path = outdir / "summary_tables.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 분석 요약 표\n\n")
            f.write("## 자치구 x 사업유형\n\n")
            try:
                f.write(pivot1_full.to_markdown())
            except Exception:
                f.write(pivot1_full.to_csv())
            if pivot2_full is not None:
                f.write("\n\n## 자치구 x 사업유형 x 사업단계(원본)\n\n")
                try:
                    f.write(pivot2_full.to_markdown())
                except Exception:
                    f.write(pivot2_full.to_csv())
            if pivot2c_full is not None:
                f.write("\n\n## 자치구 x 사업유형 x 사업단계(군집)\n\n")
                try:
                    f.write(pivot2c_full.to_markdown())
                except Exception:
                    f.write(pivot2c_full.to_csv())
        print(f"요약 표 저장: {md_path}")
    except Exception as e:
        print(f"경고: 요약 표 생성 실패 - {e}")

    # ---- 그래프 시각화 ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.switch_backend("Agg")  # 서버/헤드리스 환경 안전

        # ---- 한글 폰트 설정 ----
        try:
            from matplotlib import font_manager, rcParams
            preferred = ["Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR"]
            available = {f.name for f in font_manager.fontManager.ttflist}
            chosen = None
            for name in preferred:
                if name in available:
                    rcParams["font.family"] = name
                    chosen = name
                    break
            rcParams["axes.unicode_minus"] = False
            if chosen:
                print(f"한글 폰트 적용: {chosen}")
            else:
                print("경고: 한글 폰트를 찾지 못해 기본 폰트 사용(글리프 경고 가능)")
        except Exception as fe:
            print(f"경고: 한글 폰트 설정 실패 - {fe}")

        charts_dir = outdir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # 1) 자치구별 총합 막대 그래프 (세로형: 자치구를 하단 X축에)
        import numpy as np
        total_by_gu = tbl1.groupby(gu_col)["count"].sum().sort_values(ascending=False)
        gu_order = list(total_by_gu.index)
        plt.figure(figsize=(12, 6))
        bar_colors = sns.color_palette("viridis", n_colors=len(total_by_gu))
        x_pos = np.arange(len(total_by_gu))
        plt.bar(x_pos, total_by_gu.values, color=bar_colors)
        plt.title("자치구별 사업 건수")
        plt.ylabel("건수")
        plt.xlabel("자치구")
        plt.xticks(x_pos, total_by_gu.index, rotation=45, ha="right")
        for i, v in enumerate(total_by_gu.values):
            plt.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        bar_path = charts_dir / "gu_totals_bar.png"
        plt.savefig(bar_path, dpi=150)
        plt.close()

        # 2) 자치구 x 사업유형 스택형 비율 막대 (세로형)
        pivot1_pct = tbl1.pivot_table(index=gu_col, columns=type_col, values="count", aggfunc="sum", fill_value=0)
        pivot1_pct = pivot1_pct.reindex(gu_order)
        pivot1_pct_ratio = pivot1_pct.div(pivot1_pct.sum(axis=1), axis=0)
        plt.figure(figsize=(12, 6))
        type_palette = sns.color_palette("Paired", n_colors=len(pivot1_pct_ratio.columns))
        col_colors = {c: type_palette[i] for i, c in enumerate(pivot1_pct_ratio.columns)}
        x_pos = np.arange(len(pivot1_pct_ratio.index))
        bottom = np.zeros(len(pivot1_pct_ratio.index))
        for col in pivot1_pct_ratio.columns:
            vals = pivot1_pct_ratio[col].values  # 비율
            raw_counts = pivot1_pct[col].values  # 실제 건수
            # 현재 스택 기준으로 바 추가
            bars = plt.bar(x_pos, vals, bottom=bottom, label=col, color=col_colors[col])
            # 라벨(건수) 표시: 너무 얇은(<4% or count 0) 구간은 생략
            for i, (v_ratio, v_count) in enumerate(zip(vals, raw_counts)):
                if v_count <= 0 or v_ratio < 0.04:
                    continue
                y_center = bottom[i] + v_ratio / 2
                # 대비를 위해 배경 밝기 추정 (간단히 RGB 평균) 후 글자색 결정
                r, g, b = col_colors[col]
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_color = 'black' if luminance > 0.6 else 'white'
                plt.text(x_pos[i], y_center, str(int(v_count)), ha='center', va='center', fontsize=7, color=text_color)
            bottom += vals
        plt.xticks(x_pos, pivot1_pct_ratio.index, rotation=45, ha="right")
        plt.ylabel("비율")
        plt.xlabel("자치구")
        plt.title("자치구별 사업유형 구성비")
        plt.legend(title="사업유형", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        stacked_path = charts_dir / "gu_type_ratio_stacked.png"
        plt.savefig(stacked_path, dpi=150)
        plt.close()

        # 3) (선택) 단계 군집 Heatmap (군집화가 존재할 경우)
        if 'phase_cluster_col' in locals() and phase_cluster_col is not None:
            # 사용자 지정 단계 순서
            user_phase_order = ["구역지정", "조합설립", "추진위", "사업시행", "건축심의", "관리처분", "착공"]
            tbl_cluster = counts_by_gu_type_phase(df, gu_col, type_col, phase_cluster_col)
            # (단계, 자치구) 피벗 → 자치구를 X축(열)로 배치, 단계는 행
            mat = tbl_cluster.groupby([phase_cluster_col, gu_col])["count"].sum().unstack(fill_value=0)
            # 자치구 총합 기준 내림차순 순서 적용 (열)
            mat = mat.reindex(gu_order, axis=1)
            # 단계 순서 재배열 (지정 외 단계는 뒤에 추가)
            extra_phases = [p for p in mat.index if p not in user_phase_order]
            full_phase_order = user_phase_order + [p for p in extra_phases if p not in user_phase_order]
            # 행 재배열 시 결측(없던 단계)이 생길 수 있어 0으로 채우고 정수형으로 유지
            mat = mat.reindex(full_phase_order).fillna(0).astype(int)
            plt.figure(figsize=(14, 6))
            sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd")
            plt.title("사업추진단계(군집) × 자치구 건수 Heatmap")
            plt.xlabel("자치구")
            plt.ylabel("사업추진단계(군집)")
            plt.tight_layout()
            heat_path = charts_dir / "gu_phase_cluster_heatmap.png"
            plt.savefig(heat_path, dpi=150)
            plt.close()

        print(f"차트 저장: {charts_dir}")
    except Exception as e:
        print(f"경고: 그래프 생성 실패 - {e}")


if __name__ == "__main__":
    main()
