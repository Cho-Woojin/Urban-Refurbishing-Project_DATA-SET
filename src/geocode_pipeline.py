"""최소 구성 지오코딩 + 공간 분석 파이프라인 유틸리티.

모듈 목적 (노트북에 흩어진 로직을 분리):
- 느리고 무거운 지오코딩 단계는 한 번만 실행(스크립트/자동화 가능)
- 노트북은 시각화/탐색에 집중

외부 의존성 (최소 유지):
    pandas, geopandas, shapely, geopy (folium 은 노트북에서 선택적으로 사용)

공개 함수:
    load_csv_multi, normalize_addr, simplify_addr,
    load_cache, save_cache, geocode_primary, geocode_retry,
    resolve_repo_root, find_shape_auto, get_shape_path,
    load_seoul_boundary, build_points, spatial_join

설계 메모:
- 지오코딩/캐시 I/O 를 제외한 모든 함수는 부작용(side-effect) 없음 (in-place 갱신 명시)
- 지오코딩은 Nominatim + RateLimiter 사용 (USER_AGENT 프로젝트별 구분 권장)
- 공간 조인은 within 기준 (포인트는 WGS84 가정)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json, re, time, contextlib, os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------------- 설정 (외부에서 재정의 가능) ----------------
CITY_DEFAULT = "서울특별시"
USER_AGENT_DEFAULT = "seoul-redev-geocode"
PROGRESS_INTERVAL = 50
DEBUG = True

# 서울 행정동 이름 세트 (경계 필터용 기본 빈 set)
# 대규모 전체 목록은 노트북/스크립트에서 주입하여 메모리/로드 시점 분리 가능
SEOUL_DONG_NAMES: set[str] = set()

# ---------------- 디버그 / 실행시간 측정 유틸 ----------------

def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")

@contextlib.contextmanager
def stage(name: str):
    t0 = time.time()
    print(f"\n=== [{name}] 시작 ===")
    try:
        yield
    finally:
        print(f"=== [{name}] 종료 ({time.time() - t0:.2f}s) ===")

# ---------------- 경로 관련 헬퍼 ----------------

def resolve_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'DATA').exists() or (p / '.git').exists():
            debug(f'resolve_repo_root -> {p}')
            return p
    debug(f'resolve_repo_root fallback -> {start}')
    return start

def find_shape_auto(repo_root: Path) -> Optional[Path]:
    data_dir = repo_root / 'DATA'
    if not data_dir.exists():
        debug('DATA 디렉터리 없음')
        return None
    matches = list(data_dir.rglob('BND_ADM_DONG_PG*.shp'))
    debug(f'find_shape_auto matches: {len(matches)}')
    return matches[0] if matches else None

def get_shape_path(override: Optional[Path] = None) -> Path:
    if override:
        if override.exists():
            print(f'[SHAPE] 사용자 지정 경로 사용: {override}')
            return override
        raise FileNotFoundError(f'OVERRIDE_SHAPE_PATH 지정 경로를 찾을 수 없습니다: {override}')
    repo_root = resolve_repo_root(Path.cwd())
    shp = find_shape_auto(repo_root)
    if shp and shp.exists():
        print(f'[SHAPE] 자동 탐색 성공: {shp}')
        return shp
    raise FileNotFoundError(
        '행정동 Shapefile을 찾지 못했습니다. 수행 옵션:\n'
        '1) <repo_root>/DATA/ 하위에 BND_ADM_DONG_PG* 세트(.shp .shx .dbf .prj) 복사\n'
        '2) get_shape_path(override=Path("...")) 로 절대경로 지정\n'
        f'현재 작업 디렉터리: {Path.cwd()}'
    )

# ---------------- CSV & 주소 처리 ----------------

def load_csv_multi(path: Path, encodings: List[str] | None = None) -> pd.DataFrame:
    with stage('CSV 로드'):
        if encodings is None:
            encodings = ['utf-8-sig','utf-8','cp949','euc-kr']
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                print(f'[CSV 로드] {enc} -> {df.shape}')
                debug(f'컬럼: {list(df.columns)[:10]} ...')
                return df
            except UnicodeDecodeError as e:
                last_err = e
                debug(f'인코딩 실패: {enc}')
                continue
        raise RuntimeError(f'인코딩 판별 실패: {last_err}')

def normalize_addr(raw: str, city: str = CITY_DEFAULT) -> Optional[str]:
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = re.sub(r'\s+', ' ', str(raw).strip())
    if not s.startswith(('서울특별시','서울시','서울 ')) and city:
        s = f'{city} {s}'
    return s.replace('서울시','서울특별시')

def simplify_addr(addr: str) -> Optional[str]:
    if not addr or pd.isna(addr):
        return None
    s = re.sub(r'\(.*?\)', '', str(addr))
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.split(r'\s\d', s)[0].strip()
    tokens = s.split()
    if len(tokens) >= 3:
        s = ' '.join(tokens[:3])
    return s

# ---------------- 지오코딩 ----------------

def load_cache(path: Path) -> Dict[str, Tuple[float,float]]:
    with stage('캐시 로드'):
        if path.exists():
            try:
                with open(path,'r',encoding='utf-8') as f:
                    data = json.load(f)
                debug(f'캐시 로드 성공 entries={len(data)}')
                return {k: tuple(v) for k,v in data.items()}
            except Exception as e:
                print('[경고] 캐시 파싱 실패 → 새로 생성', e)
        return {}

def save_cache(cache: Dict[str, Tuple[float,float]], path: Path):
    with stage('캐시 저장'):
        with open(path,'w',encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        debug(f'저장 entries={len(cache)} path={path}')

def geocode_primary(df: pd.DataFrame, cache: Dict[str, Tuple[float,float]], delay: float, user_agent: str = USER_AGENT_DEFAULT) -> None:
    with stage('지오코딩 1차'):
        geolocator = Nominatim(user_agent=user_agent)
        geocode_func = RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=True)
        miss = 0
        lat_list, lon_list = [], []
        try:
            from tqdm import tqdm  # lazy import
            iterator = tqdm(df['정규화주소'], desc='Geocoding 1차')
        except ImportError:
            iterator = df['정규화주소']
        for i, addr in enumerate(iterator):
            if pd.isna(addr):
                lat_list.append(None); lon_list.append(None); continue
            if addr in cache:
                lat, lon = cache[addr]
                lat_list.append(lat); lon_list.append(lon); continue
            loc = geocode_func(addr)
            if loc is None:
                lat_list.append(None); lon_list.append(None); miss += 1
            else:
                cache[addr] = (loc.latitude, loc.longitude)
                lat_list.append(loc.latitude); lon_list.append(loc.longitude)
            if DEBUG and (i+1) % PROGRESS_INTERVAL == 0:
                debug(f'진행 {i+1}/{len(df)} 실패누적={miss}')
        df['lat'] = lat_list; df['lon'] = lon_list
        print(f'[지오코딩 1차] 실패 {miss} / 총 {len(df)} (실패율 {(miss/len(df)) if len(df) else 0:.2%})')

def geocode_retry(df: pd.DataFrame, cache: Dict[str, Tuple[float,float]], delay: float, user_agent: str = USER_AGENT_DEFAULT) -> None:
    with stage('지오코딩 재시도'):
        remaining = df['lat'].isna().sum() if 'lat' in df.columns else len(df)
        if remaining == 0:
            print('[재시도] 실패 없음 → 건너뜀'); return
        geolocator = Nominatim(user_agent=user_agent)
        geocode_func = RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=True)
        subset = df[df['lat'].isna()].copy(); subset['재시도주소'] = subset['정규화주소'].apply(simplify_addr)
        new_success = 0
        for i, (idx, row) in enumerate(subset.iterrows()):
            addr2 = row['재시도주소']
            if not addr2: continue
            if addr2 in cache:
                lat, lon = cache[addr2]
                df.at[idx,'lat'] = lat; df.at[idx,'lon'] = lon; new_success += 1; continue
            loc = geocode_func(addr2)
            if loc is not None:
                cache[addr2] = (loc.latitude, loc.longitude)
                df.at[idx,'lat'] = loc.latitude; df.at[idx,'lon'] = loc.longitude; new_success += 1
            if DEBUG and (i+1) % 10 == 0:
                debug(f'재시도 진행 {i+1}/{len(subset)} 추가성공={new_success}')
        still = df['lat'].isna().sum()
        print(f'[재시도] 추가 성공 {new_success}, 잔여 실패 {still}')

# ---------------- 경계 / 공간조인 ----------------

def load_seoul_boundary(shape_path: Path, dong_names: Optional[set[str]] = None) -> gpd.GeoDataFrame:
    """서울 행정동 Shapefile 로드 후 행정동 컬럼 자동 선택 → 서울만 필터 → 자치구 단위 dissolve.

    매개변수:
      shape_path: Shapefile 경로
      dong_names: 허용할 행정동 이름 집합 (None 인 경우 전역 SEOUL_DONG_NAMES 사용)

    반환:
      자치구 단위로 dissolve 된 GeoDataFrame (EPSG:4326 보장)
    """
    dong_names = dong_names if dong_names is not None else SEOUL_DONG_NAMES
    if not dong_names:
        raise ValueError('dong_names set 비어있음 (SEOUL_DONG_NAMES 주입 필요).')
    with stage('경계 로드/필터(행정동)'):
        all_emd = gpd.read_file(shape_path)
        if all_emd.empty:
            raise ValueError('경계 Shapefile 비어있음')
        lower_map = {c.lower(): c for c in all_emd.columns}
        dong_key_candidates = ['adm_dr_nm','adm_nm','emd_kor_nm','emd_nm','행정동','동명','법정동명','법정동']
        dong_cols = [lower_map[k] for k in dong_key_candidates if k in lower_map]
        if not dong_cols:
            raise KeyError('행정동 후보 컬럼 없음')
        best_col = None; best_rate = -1.0; best_series = None
        for c in dong_cols:
            series = all_emd[c].astype(str).str.strip()
            mask = series.isin(dong_names)
            rate = mask.mean()
            if rate > best_rate:
                best_rate = rate; best_col = c; best_series = series
        if best_rate <= 0:
            raise ValueError('서울 행정동 교집합이 0%')
        seoul_emd = all_emd[best_series.isin(dong_names)].copy()
        # 자치구 컬럼 탐색 (이름 우선 → 코드)
        lower_map_seoul = {c.lower(): c for c in seoul_emd.columns}
        sig_name_keys = ['sigungu_nm','sig_kor_nm','sig_nm','시군구명','sig_eng_nm']
        sig_code_keys = ['sig_cd','시군구코드']
        sig_col = None
        for k in sig_name_keys:
            if k in lower_map_seoul:
                sig_col = lower_map_seoul[k]; break
        if not sig_col:
            for k in sig_code_keys:
                if k in lower_map_seoul:
                    sig_col = lower_map_seoul[k]; break
        if not sig_col:
            if 'EMD_CD' in seoul_emd.columns:
                seoul_emd['SIG_CD_AUTOGEN'] = seoul_emd['EMD_CD'].astype(str).str.slice(0,5)
                sig_col = 'SIG_CD_AUTOGEN'
            else:
                # (폴백) 임의 문자열 컬럼에서 5자리 숫자 패턴 비율이 높으면 코드로 간주
                for c in seoul_emd.columns:
                    if seoul_emd[c].dtype == object:
                        sample = seoul_emd[c].astype(str).str[:5]
                        if sample.str.isdigit().mean() > 0.9 and sample.str.len().eq(5).mean() > 0.9:
                            seoul_emd['SIG_CD_FALLBACK'] = sample
                            sig_col = 'SIG_CD_FALLBACK'
                            break
        if not sig_col:
            raise KeyError('자치구 컬럼을 찾거나 생성할 수 없습니다.')
        seoul_gu = seoul_emd.dissolve(by=sig_col, as_index=False)
        if seoul_gu.crs is None:
            seoul_gu.set_crs(epsg=4326, inplace=True)
        elif seoul_gu.crs.to_string().lower() not in ('epsg:4326','wgs84'):
            seoul_gu = seoul_gu.to_crs(epsg=4326)
        return seoul_gu

def build_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    with stage('포인트 생성'):
        if 'lat' not in df.columns or 'lon' not in df.columns:
            raise KeyError('lat/lon 컬럼이 필요합니다.')
        pts = df.dropna(subset=['lat','lon']).copy()
        gdf = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts['lon'], pts['lat'])], crs='EPSG:4326')
        return gdf

def spatial_join(pts_gdf: gpd.GeoDataFrame, seoul_gu: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, Optional[str]]:
    with stage('공간조인'):
        priority = ['SIGUNGU_NM','SIG_KOR_NM','SIG_NM','시군구명','SIG_CD','시군구코드','SIG_CD_AUTOGEN','SIG_CD_FALLBACK']
        cand = [c for c in priority if c in seoul_gu.columns]
        admin_col = cand[0] if cand else None
        if admin_col:
            joined = gpd.sjoin(pts_gdf, seoul_gu[[admin_col,'geometry']], how='left', predicate='within')
            return joined, admin_col
        return pts_gdf, None

__all__ = [
    'load_csv_multi','normalize_addr','simplify_addr',
    'load_cache','save_cache','geocode_primary','geocode_retry',
    'resolve_repo_root','find_shape_auto','get_shape_path',
    'load_seoul_boundary','build_points','spatial_join',
    'SEOUL_DONG_NAMES'
]
