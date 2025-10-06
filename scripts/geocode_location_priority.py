#!/usr/bin/env python
"""정비구역위치 1차, (자치구+법정동+대표지번) 2차 조합 기반 지오코딩 스크립트

요구사항 반영:
  1) 첫 주소 후보: 정비구역위치 원문 (도시 접두 자동 보강)
  2) 1차 실패 시 주소 단순화(괄호/노이즈 제거, 번지 첫 항)
  3) 2차: 자치구 + 법정동 + 대표지번 조합 (번지/동 부분추출 포함)
  4) 최종 fallback: 자치구 + 법정동, 자치구 단독

상태코드 예:
  ok / ok_fallbackN, cache_ok*, simplified_ok, combo_ok, dong_ok, gu_only_ok, no_result, ssl_error

캐시(JSON) 구조: { query: [lat, lon] }

사용 예:
  python scripts/geocode_location_priority.py \
    --input outputs/주택정비형_정비구역지정이후.csv \
    --out outputs/주택정비형_geocoded.csv \
    --cache outputs/geocode_cache_location.json \
    --delay 1.2 --retry-simplify
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd

try:
    from geopy.geocoders import Nominatim  # type: ignore
    from geopy.extra.rate_limiter import RateLimiter  # type: ignore
    from geopy.exc import GeocoderUnavailable  # type: ignore
except Exception:
    Nominatim = None  # type: ignore
    RateLimiter = None  # type: ignore
    GeocoderUnavailable = Exception  # type: ignore

ENCODINGS = ["utf-8-sig","utf-8","cp949","euc-kr"]
SPACE_RE = re.compile(r"\s+")
PAREN_RE = re.compile(r"\(.*?\)")
MULTI_BUNJI_RE = re.compile(r"[,·]\s*산?\d+(?:-\d+)?")
BUNJI_HEAD_RE = re.compile(r"산?\d+(?:-\d+)?")

CORE_COLS = ["정비구역위치","자치구","법정동","대표지번"]

# ---------------- I/O ----------------

def read_csv_multi(path: Path) -> pd.DataFrame:
    last=None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last=e
    raise RuntimeError(f"CSV 읽기 실패: {path} (last={last})")

def load_cache(path: Path) -> Dict[str, Tuple[float,float]]:
    if path.exists():
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            return {k: tuple(v) for k,v in data.items()}  # type: ignore
        except Exception:
            print('[WARN] 캐시 파싱 실패 → 새로 생성')
    return {}

def save_cache(cache: Dict[str, Tuple[float,float]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(cache,f,ensure_ascii=False,indent=2)

# ---------------- 주소 처리 ----------------

def normalize_city_prefix(s: str, city: str) -> str:
    if not s:
        return s
    if s.startswith((city, city.replace('특별시','시'),'서울 ')):
        return s.replace('서울시', city)
    return f"{city} {s}".strip()

def normalize_base(s: str | float | int | None) -> str:
    if s is None:
        return ''
    v=str(s).replace('\ufeff','').strip()
    v=SPACE_RE.sub(' ',v)
    return v

def simplify_location(s: str) -> str:
    # 괄호 제거, 다중 번지 첫 번지 유지, 기타 노이즈 토큰 제거(일대, 외n필지 등)
    if not s:
        return s
    s=PAREN_RE.sub('',s)
    s=re.sub(r"외\s*\d+필지",'',s)
    s=re.sub(r"일대|주변|일부|인근","",s)
    s=SPACE_RE.sub(' ',s).strip()
    # 다중 번지: 콤마/중점 이후 번지 패턴 제거
    s=MULTI_BUNJI_RE.split(s)[0].strip()
    # 번지 패턴 첫 것까지만 남기고 뒤 절단
    m=BUNJI_HEAD_RE.search(s)
    if m:
        # 번지 이후 공백 기준으로 너무 긴 설명이면 잘라냄
        head=s[:m.end()]
        tail=s[m.end():]
        if len(tail.split())>2:
            s=head
    return s

def extract_dong_bunji(s: str) -> Tuple[str,str]:
    raw=re.sub(r"\s+","",s or '')
    if not raw:
        return '', ''
    m_d=re.search(r"([가-힣0-9]+동)", raw)
    dong=m_d.group(1) if m_d else ''
    bunji=''
    if dong:
        tail=raw.split(dong,1)[1]
        m_b=re.search(r"산?([0-9]+(?:-[0-9]+)?)", tail)
        bunji=m_b.group(1) if m_b else ''
    return dong, bunji

# ---------------- 후보 생성 ----------------

def generate_candidates(row: pd.Series, city: str) -> List[Tuple[str,str]]:
    """후보 (query, tag) 리스트.
    tag: 어떤 전략으로 나온 후보인지(우선순위/출처 표시)
    우선순위 순으로 append
    """
    out: List[Tuple[str,str]] = []
    loc = normalize_base(row.get('정비구역위치'))
    gu = normalize_base(row.get('자치구'))
    dong = normalize_base(row.get('법정동'))
    main = normalize_base(row.get('대표지번'))

    if loc:
        out.append((normalize_city_prefix(loc, city),'loc_raw'))
        simp = simplify_location(loc)
        if simp and simp != loc:
            out.append((normalize_city_prefix(simp, city),'loc_simplified'))
    # 조합 (자치구 + 법정동 + 대표지번)
    if gu and dong and main:
        out.append((normalize_city_prefix(f"{gu} {dong} {main}", city),'combo_full'))
    # 동+번지 추출
    if loc:
        d2,b2=extract_dong_bunji(loc)
        if d2 and b2:
            out.append((normalize_city_prefix(f"{d2} {b2}", city),'loc_dong_bunji'))
        if d2:
            out.append((normalize_city_prefix(d2, city),'loc_dong'))
    # fallback: 자치구+법정동, 자치구 단독
    if gu and dong:
        out.append((normalize_city_prefix(f"{gu} {dong}", city),'gu_dong'))
    if gu:
        out.append((normalize_city_prefix(gu, city),'gu_only'))

    # 중복 제거 (첫 등장 우선)
    seen=set()
    uniq=[]
    for q,t in out:
        if q and q not in seen:
            seen.add(q); uniq.append((q,t))
    return uniq

# ---------------- 지오코딩 ----------------

def geocode_frame(df: pd.DataFrame, city: str, delay: float, cache: Dict[str,Tuple[float,float]], user_agent: str, network: bool, insecure: bool) -> pd.DataFrame:
    if Nominatim is None or RateLimiter is None:
        print('[ERROR] geopy 미설치: pip install geopy'); return df
    geocode_func=None
    if network:
        if insecure:
            try:
                import requests, urllib3  # type: ignore
                from geopy.adapters import RequestsAdapter  # type: ignore
                urllib3.disable_warnings()
                session=requests.Session(); session.verify=False
                geolocator=Nominatim(user_agent=user_agent, adapter_factory=lambda **kw: RequestsAdapter(session=session, **kw))
            except Exception as e:
                print(f"[WARN] insecure 초기화 실패 → 일반 모드: {e}")
                geolocator=Nominatim(user_agent=user_agent)
        else:
            geolocator=Nominatim(user_agent=user_agent)
        geocode_func=RateLimiter(geolocator.geocode, min_delay_seconds=delay, swallow_exceptions=False)
    else:
        print('[INFO] 네트워크 비활성 모드 - 캐시 조회만')

    queries=[]; statuses=[]; lats=[]; lons=[]; tags=[]
    try:
        from tqdm import tqdm  # type: ignore
        iterator=tqdm(df.iterrows(), total=len(df), desc='Geocoding')
    except Exception:
        iterator=df.iterrows()
    for idx,row in iterator:
        candidates=generate_candidates(row, city)
        used_q=None; status='no_candidate'; lat=None; lon=None; used_tag=''
        for i,(q,tag) in enumerate(candidates):
            used_q=q; used_tag=tag
            if q in cache:
                lat,lon=cache[q]
                if lat is not None and lon is not None:
                    status=f'cache_ok' if i==0 else f'cache_ok_fallback{i}'
                    break
                else:
                    status='cache_na'; continue
            if not network:
                status='cache_miss_network_off'; continue
            try:
                loc=geocode_func(q) if geocode_func else None
            except GeocoderUnavailable as ge:
                msg=str(ge)
                if 'CERTIFICATE_VERIFY_FAILED' in msg:
                    status='ssl_error'
                else:
                    status='unavailable'
                cache[q]=(None,None)  # type: ignore
                continue
            except Exception:
                status='error'; cache[q]=(None,None)  # type: ignore
                continue
            if loc is not None:
                lat=loc.latitude; lon=loc.longitude
                cache[q]=(lat,lon)
                # 태그별 성공 코드 세분화
                if tag=='loc_raw':
                    status='ok'
                elif tag=='loc_simplified':
                    status='simplified_ok'
                elif tag=='combo_full':
                    status='combo_ok'
                elif tag=='loc_dong_bunji':
                    status='dong_bunji_ok'
                elif tag=='loc_dong':
                    status='dong_ok'
                elif tag=='gu_dong':
                    status='gu_dong_ok'
                elif tag=='gu_only':
                    status='gu_only_ok'
                else:
                    status='ok'
                break
            else:
                cache[q]=(None,None)  # type: ignore
                status='no_result'
        queries.append(used_q); statuses.append(status); lats.append(lat); lons.append(lon); tags.append(used_tag)
    out=df.copy()
    out['lat']=lats; out['lon']=lons; out['geocode_query']=queries; out['geocode_status']=statuses; out['geocode_tag']=tags
    return out

# ---------------- CLI ----------------

def parse_args():
    ap=argparse.ArgumentParser(description='정비구역위치 우선 지오코딩')
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cache', type=Path, default=Path('outputs/geocode_cache_location.json'))
    ap.add_argument('--city-prefix', default='서울특별시')
    ap.add_argument('--delay', type=float, default=1.2)
    ap.add_argument('--user-agent', default='loc-priority-geocode')
    ap.add_argument('--disable-network', action='store_true')
    ap.add_argument('--insecure', action='store_true')
    ap.add_argument('--max-rows', type=int)
    return ap.parse_args()

# ---------------- main ----------------

def main():
    args=parse_args()
    if not args.input.exists():
        print('[ERROR] 입력 없음', args.input); return 2
    df=read_csv_multi(args.input)
    # 필수 컬럼 존재 여부
    missing=[c for c in CORE_COLS if c not in df.columns]
    if missing:
        print('[WARN] 일부 컬럼 누락:', missing)
    if args.max_rows:
        df=df.head(args.max_rows).copy()
    cache=load_cache(args.cache)
    result=geocode_frame(df, args.city_prefix, args.delay, cache, args.user_agent, network=not args.disable_network, insecure=args.insecure)
    save_cache(cache, args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower()=='.parquet':
        result.to_parquet(args.out, index=False)
    else:
        result.to_csv(args.out, index=False, encoding='utf-8-sig')
    # 간단 통계
    stat=result['geocode_status'].value_counts(dropna=False).to_dict()
    success=int(result['lat'].notna().sum())
    fail=int((result['lat'].isna()|result['lon'].isna()).sum())
    print(f"[DONE] rows={len(result)} success={success} fail={fail} status_dist={stat}")
    return 0

if __name__=='__main__':  # pragma: no cover
    sys.exit(main())
