#!/usr/bin/env python
"""두 컬럼(대표지번, 정비구역위치) 기반 다단계 지오코딩 스크립트

용도:
  DATA/핵심단계_구간포함_corrected.csv 처럼 주소 관련 컬럼이 2개 있을 때
  -> 후보 주소 문자열을 여러 개 생성하여 순차적으로 Nominatim 지오코딩
  -> 첫 성공 결과를 채우고 상태코드 기록

특징:
  - 캐시(JSON) 사용 (query -> (lat, lon))
  - 주소 후보 우선순위: (대표지번 전체) > (정비구역위치 전체) > (대표지번 동/번지 추출) > (정비구역위치 동/번지 추출)
  - --city-prefix 로 서울특별시 접두 자동 보강 (없으면 붙임)
  - --retry-simplify 로 1차 실패행에 대해 괄호/길이 단순화 후 재시도
  - --max-rows 로 상위 N행 테스트

출력:
  - 결과 CSV/Parquet (lat, lon, geocode_query, geocode_status 컬럼 포함)
  - 캐시 JSON (기존 + 신규)
  - 실패행 CSV(옵션)

예시:
  python scripts/geocode_dual_columns.py \
      --input DATA/핵심단계_구간포함_corrected.csv \
      --out outputs/핵심단계_geocoded.csv \
      --cache outputs/geocode_cache_dual.json \
      --retry-simplify --delay 1.2
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

try:
    from geopy.geocoders import Nominatim  # type: ignore
    from geopy.extra.rate_limiter import RateLimiter  # type: ignore
except Exception:  # pragma: no cover
    Nominatim = None  # type: ignore
    RateLimiter = None  # type: ignore

# ---------------- 공통 유틸 ----------------
ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

def read_csv_multi(path: Path) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (last={last})")

def load_cache(path: Path) -> Dict[str, Tuple[float, float]]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: tuple(v) for k, v in data.items()}  # type: ignore
        except Exception:
            print(f"[WARN] 캐시 파싱 실패 → 무시: {path}")
    return {}

def save_cache(cache: Dict[str, Tuple[float,float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ---------------- 주소 전처리 ----------------
SPACE_RE = re.compile(r"\s+")
PAREN_RE = re.compile(r"\(.*?\)")
HYPHEN_SPLIT_RE = re.compile(r"\s\d")

def normalize_base(s: str | float | int | None) -> str:
    if s is None:
        return ""
    v = str(s).replace("\ufeff", "").strip()
    v = SPACE_RE.sub(" ", v)
    return v

def add_city_prefix(addr: str, city: str) -> str:
    if not addr:
        return addr
    if addr.startswith((city, city.replace("특별시", "시"), "서울 ")):
        return addr.replace("서울시", city)
    return f"{city} {addr}".strip()

def simplify_addr(addr: str) -> str:
    if not addr:
        return addr
    s = PAREN_RE.sub("", addr)
    s = SPACE_RE.sub(" ", s).strip()
    # 숫자 뒤 토큰 잘라 과도한 세부 제거 (예: "...동 123-45 외 3필지" → 앞부분만)
    m = HYPHEN_SPLIT_RE.split(s)
    if m:
        head = m[0].strip()
        if len(head.split()) >= 3:
            head = " ".join(head.split()[:3])
        return head
    return s

def extract_dong_bunji(raw: str) -> Tuple[str, str]:
    s = re.sub(r"\s+", "", raw or "")
    if not s:
        return "", ""
    m_d = re.search(r"([가-힣0-9]+동)", s)
    dong = m_d.group(1) if m_d else ""
    bunji = ""
    if dong:
        tail = s.split(dong, 1)[1]
        m_b = re.search(r"산?([0-9]+(?:-[0-9]+)?)", tail)
        bunji = m_b.group(1) if m_b else ""
    return dong, bunji

# ---------------- 주소 후보 생성 ----------------

def generate_candidates(row: pd.Series, col_a: str, col_b: str, city: str) -> List[str]:
    vals = []
    a = normalize_base(row.get(col_a))
    b = normalize_base(row.get(col_b))
    if a:
        vals.append(add_city_prefix(a, city))
    if b and b != a:
        vals.append(add_city_prefix(b, city))
    # 파생 (동+번지)
    for src in [a, b]:
        if not src:
            continue
        dong, bunji = extract_dong_bunji(src)
        if dong and bunji:
            vals.append(add_city_prefix(f"{dong} {bunji}", city))
        if dong:
            vals.append(add_city_prefix(dong, city))
    # 중복 제거 순서 유지
    out: List[str] = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

# ---------------- 지오코딩 ----------------

def geocode_df(df: pd.DataFrame, col_a: str, col_b: str, city: str, delay: float, cache: Dict[str, Tuple[float,float]], user_agent: str) -> pd.DataFrame:
    if Nominatim is None or RateLimiter is None:
        print("[ERROR] geopy 미설치: pip install geopy"); return df
    geolocator = Nominatim(user_agent=user_agent)
    geocode_func = RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=True)

    queries: list[Optional[str]] = []
    statuses: list[str] = []
    lats: list[Optional[float]] = []
    lons: list[Optional[float]] = []

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(df.iterrows(), total=len(df), desc="Geocoding")
    except Exception:
        iterator = df.iterrows()

    for idx, row in iterator:
        cands = generate_candidates(row, col_a, col_b, city)
        used_q: Optional[str] = None
        status = "no_candidate"
        lat = None; lon = None
        for i, q in enumerate(cands):
            used_q = q
            if q in cache:
                lat, lon = cache[q]
                if lat is not None and lon is not None:
                    status = "cache_ok" if i == 0 else f"cache_ok_fallback{i}"
                    break
                else:
                    status = "cache_na"  # continue
            else:
                loc = geocode_func(q)
                if loc is not None:
                    lat = loc.latitude; lon = loc.longitude
                    cache[q] = (lat, lon)
                    status = "ok" if i == 0 else f"ok_fallback{i}"
                    break
                else:
                    cache[q] = (None, None)  # type: ignore
                    status = "no_result"
        queries.append(used_q)
        statuses.append(status)
        lats.append(lat)
        lons.append(lon)

    df = df.copy()
    df['lat'] = lats
    df['lon'] = lons
    df['geocode_query'] = queries
    df['geocode_status'] = statuses
    return df

# ---------------- 재시도 ----------------

def retry_simplify(df: pd.DataFrame, col_a: str, col_b: str, city: str, delay: float, cache: Dict[str, Tuple[float,float]], user_agent: str) -> int:
    if Nominatim is None or RateLimiter is None:
        return 0
    geolocator = Nominatim(user_agent=user_agent)
    geocode_func = RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=True)
    target = df[(df['lat'].isna()) | (df['lon'].isna())].copy()
    filled = 0
    for idx, row in target.iterrows():
        sims = []
        for src_col in [col_a, col_b]:
            raw = row.get(src_col)
            if isinstance(raw, str) and raw.strip():
                sims.append(add_city_prefix(simplify_addr(raw), city))
        # 중복 제거
        uniq = []
        seen = set()
        for s in sims:
            if s and s not in seen:
                seen.add(s); uniq.append(s)
        for i, q in enumerate(uniq):
            if q in cache and cache[q][0] is not None and cache[q][1] is not None:
                df.at[idx, 'lat'] = cache[q][0]; df.at[idx, 'lon'] = cache[q][1]; filled += 1; break
            loc = geocode_func(q)
            if loc is not None:
                cache[q] = (loc.latitude, loc.longitude)
                df.at[idx, 'lat'] = loc.latitude; df.at[idx, 'lon'] = loc.longitude; filled += 1
                break
    return filled

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="두 컬럼 기반 다단계 지오코딩")
    ap.add_argument('--input', required=True, type=Path, help='입력 CSV 경로')
    ap.add_argument('--col-a', default='대표지번', help='주소/지번 후보 1차 컬럼명')
    ap.add_argument('--col-b', default='정비구역위치', help='주소/지번 후보 2차 컬럼명')
    ap.add_argument('--city-prefix', default='서울특별시', help='주소 맨 앞 도시명 접두 보강')
    ap.add_argument('--cache', type=Path, default=Path('outputs/geocode_cache_dual.json'), help='지오코드 캐시 JSON')
    ap.add_argument('--delay', type=float, default=1.1, help='Nominatim 질의간 최소 지연(초)')
    ap.add_argument('--out', required=True, type=Path, help='결과 출력 경로(.csv/.parquet)')
    ap.add_argument('--retry-simplify', action='store_true', help='1차 실패행 단순화 주소 재시도')
    ap.add_argument('--max-rows', type=int, help='상위 N행만 처리(테스트용)')
    ap.add_argument('--failures-out', type=Path, help='지오코딩 실패행 CSV 출력 경로')
    ap.add_argument('--user-agent', default='dual-geocoder-cli', help='Nominatim User-Agent')
    return ap.parse_args()

# ---------------- main ----------------

def main():
    args = parse_args()
    if not args.input.exists():
        print('[ERROR] 입력 파일 없음:', args.input); return 2
    df = read_csv_multi(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    for col in [args.col_a, args.col_b]:
        if col not in df.columns:
            print(f'[ERROR] 컬럼 없음: {col}. 실제 컬럼: {list(df.columns)[:20]}')
            return 3

    cache = load_cache(args.cache)
    result = geocode_df(df, args.col_a, args.col_b, args.city_prefix, args.delay, cache, args.user_agent)

    if args.retry_simplify:
        added = retry_simplify(result, args.col_a, args.col_b, args.city_prefix, args.delay, cache, args.user_agent)
        print(f'[INFO] 단순화 재시도 추가 성공: {added}')

    # 실패행 저장(옵션)
    if args.failures_out:
        fail_mask = result['lat'].isna() | result['lon'].isna()
        failures = result.loc[fail_mask].copy()
        args.failures_out.parent.mkdir(parents=True, exist_ok=True)
        failures.to_csv(args.failures_out, index=False, encoding='utf-8-sig')
        print(f'[INFO] 실패행 {len(failures)}건 저장 → {args.failures_out}')

    # 캐시 저장
    save_cache(cache, args.cache)

    # 출력
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == '.parquet':
        result.to_parquet(args.out, index=False)
    else:
        result.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'[DONE] 저장: {args.out} (rows={len(result)}) 실패={(result["lat"].isna() | result["lon"].isna()).sum()}')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
