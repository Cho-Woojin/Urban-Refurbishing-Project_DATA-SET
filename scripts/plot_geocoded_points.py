#!/usr/bin/env python
"""지오코딩 결과 CSV를 기반으로 matplotlib / (선택) geopandas + contextily 로 포인트 지도 PNG 출력

기능:
  1) 기본: 단순 산점도 (경위도) WGS84 그대로
  2) --with-boundary: 행정동/구 경계 Shapefile 덮어 그림 (geopandas 필요)
  3) --web-tiles: Web Mercator 로 투영 후 배경 타일(contextily) 추가
  4) --hue-col: 색상 구분 컬럼 지정 (예: 진행단계)
  5) --filter-success: success==True 행만 표시
  6) --alpha, --size 로 시각적 조정

설치 필요 라이브러리: matplotlib (필수), geopandas/contextily (옵션)

예시:
  python scripts/plot_geocoded_points.py \
    --input outputs/주택정비형_geocoded_full.csv \
    --output outputs/plots/geocoded_points.png \
    --hue-col 진행단계 --filter-success --with-boundary DATA/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp --web-tiles

제한:
  - 타일 사용 시 인터넷 연결 필요
  - 대량 포인트(>5만) 시 scatter 렌더링 속도 저하 가능 → hexbin 옵션 추후 확장 가능
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None  # type: ignore

# contextily는 선택
try:
    import contextily as cx  # type: ignore
except Exception:
    cx = None  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description='지오코딩 포인트 지도 시각화')
    ap.add_argument('--input', required=True, type=Path, help='지오코딩된 CSV (lat, lon 필수)')
    ap.add_argument('--output', required=True, type=Path, help='저장할 PNG 경로')
    ap.add_argument('--lat-col', default='lat')
    ap.add_argument('--lon-col', default='lon')
    ap.add_argument('--hue-col', help='색상 구분 컬럼')
    ap.add_argument('--filter-success', action='store_true', help='success==True 행만 사용')
    ap.add_argument('--with-boundary', type=Path, help='경계 Shapefile (EPSG:4326 혹은 투영정보 포함)')
    ap.add_argument('--web-tiles', action='store_true', help='Web Mercator + 배경 타일 추가 (contextily 필요)')
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--size', type=float, default=20.0)
    ap.add_argument('--dpi', type=int, default=160)
    ap.add_argument('--title', default='Geocoded Points')
    return ap.parse_args()


def load_data(path: Path, lat_col: str, lon_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f'필수 컬럼 누락: {lat_col}, {lon_col}')
    # numeric 변환
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    return df


def main():
    args = parse_args()
    df = load_data(args.input, args.lat_col, args.lon_col)

    if args.filter_success:
        if 'success' in df.columns:
            before = len(df)
            df = df[df['success']].copy()
            print(f'[INFO] success 필터 적용: {before} -> {len(df)}')
        else:
            print('[WARN] success 컬럼 없음 - 필터 생략')

    df_plot = df.dropna(subset=[args.lat_col, args.lon_col]).copy()
    if df_plot.empty:
        print('[ERROR] 유효한 좌표 행이 없습니다.'); return 2

    boundary_gdf = None
    if args.with_boundary:
        if gpd is None:
            print('[WARN] geopandas 미설치 - 경계 레이어 생략')
        else:
            boundary_gdf = gpd.read_file(args.with_boundary)
            # 좌표계 정렬: Web 타일 모드면 3857, 아니면 4326 유지
            if args.web_tiles:
                if boundary_gdf.crs is None:
                    boundary_gdf.set_crs(epsg=4326, inplace=True)
                boundary_gdf = boundary_gdf.to_crs(epsg=3857)

    use_web = args.web_tiles and cx is not None
    if args.web_tiles and cx is None:
        print('[WARN] contextily 미설치 - web 타일 비활성')
        use_web = False

    # 포인트 GeoDataFrame (옵션)
    gdf_points = None
    if gpd is not None:
        gdf_points = gpd.GeoDataFrame(df_plot, geometry=gpd.points_from_xy(df_plot[args.lon_col], df_plot[args.lat_col]), crs='EPSG:4326')
        if use_web:
            gdf_points = gdf_points.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(7,7))

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='lightgray', linewidth=0.5)

    if gdf_points is not None:
        # 색상 구분
        if args.hue_col and args.hue_col in gdf_points.columns:
            unique_vals = gdf_points[args.hue_col].astype(str).unique()
            cmap = plt.get_cmap('tab20')
            for i,val in enumerate(unique_vals):
                sub = gdf_points[gdf_points[args.hue_col].astype(str)==val]
                sub.plot(ax=ax, markersize=args.size, alpha=args.alpha, color=cmap(i % 20), label=val)
        else:
            gdf_points.plot(ax=ax, markersize=args.size, alpha=args.alpha, color='crimson')
    else:
        # geopandas 없는 경우 matplotlib scatter 직접
        if args.hue_col and args.hue_col in df_plot.columns:
            cats = df_plot[args.hue_col].astype(str)
            unique_vals = cats.unique()
            cmap = plt.get_cmap('tab20')
            for i,val in enumerate(unique_vals):
                mask = cats==val
                plt.scatter(df_plot.loc[mask, args.lon_col], df_plot.loc[mask, args.lat_col], s=args.size, alpha=args.alpha, color=cmap(i % 20), label=val)
        else:
            plt.scatter(df_plot[args.lon_col], df_plot[args.lat_col], s=args.size, alpha=args.alpha, color='crimson')

    if use_web and cx is not None:
        cx.add_basemap(ax, crs='EPSG:3857', attribution_size=6)

    ax.set_title(args.title)
    if args.hue_col and ((gdf_points is not None and args.hue_col in gdf_points.columns) or (gdf_points is None and args.hue_col in df_plot.columns)):
        ax.legend(loc='best', fontsize=8, frameon=False)

    if use_web:
        ax.set_axis_off()
    else:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f'[DONE] 저장: {args.output}')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
