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
        --delay 1.2

구버전 호환: 과거 예시에 있었던 --retry-simplify 옵션은 자동 단순화 로직으로 대체되었으며 지금은 의미 없는(no-op) 호환 플래그입니다.
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

# 정교한 번지(산/본번/부번) 추출용 정규식
DONG_RE = re.compile(r"([가-힣0-9]+동)")
BUNJI_TOKEN_RE = re.compile(r"(산)?(\d+)(?:-(\d+))?번?지?")  # 산25-3 / 25-3 / 25 형태

# 다중 번지 구분자 (콤마, 중점, 슬래시 등) → 첫 번지만 활용
MULTI_SEPARATOR_RE = re.compile(r"[,·/]")

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

def parse_bunji(raw: str) -> Optional[Dict[str,str]]:
    """번지 문자열에서 산 여부 / 본번 / 부번 추출.
    반환: { 'san': 'Y'|'N', 'main': 본번(str), 'sub': 부번(str|''), 'repr': 표시용원문 }
    여러 번지가 포함될 경우 첫 번지(정렬순)만 사용.
    """
    if not raw:
        return None
    txt = raw.strip()
    # 다중 분리 → 첫 토큰만
    parts = MULTI_SEPARATOR_RE.split(txt)
    if parts:
        txt = parts[0].strip()
    # '외 2필지' 제거
    txt = re.sub(r"외\s*\d+필지", "", txt)
    m = BUNJI_TOKEN_RE.search(txt)
    if not m:
        return None
    san_flag, main, sub = m.group(1) or '', m.group(2), m.group(3) or ''
    return {
        'san': 'Y' if san_flag else 'N',
        'main': main,
        'sub': sub,
        'repr': (san_flag or '') + main + (('-' + sub) if sub else '')
    }

def extract_dong_bunji_detailed(s: str) -> Tuple[str, Optional[Dict[str,str]]]:
    """문장에서 동 + 번지(산/본/부) 조합 추출. 번지는 dict 형태.
    예: '종로구 사직동 12-3 일대' -> ('사직동', {'san':'N','main':'12','sub':'3','repr':'12-3'})
    """
    if not s:
        return '', None
    s_norm = re.sub(r"\s+", "", s)
    m_d = DONG_RE.search(s_norm)
    if not m_d:
        return '', None
    dong = m_d.group(1)
    tail = s_norm.split(dong, 1)[1]
    m_b = BUNJI_TOKEN_RE.search(tail)
    if not m_b:
        return dong, None
    san_flag, main, sub = m_b.group(1) or '', m_b.group(2), m_b.group(3) or ''
    return dong, {
        'san': 'Y' if san_flag else 'N',
        'main': main,
        'sub': sub,
        'repr': (san_flag or '') + main + (('-' + sub) if sub else '')
    }

def extract_dong_bunji(s: str) -> Tuple[str,str]:  # 유지: 기존 인터페이스 (하위호환)
    dong, info = extract_dong_bunji_detailed(s)
    if not info:
        return dong, ''
    return dong, info['repr']

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
    main_raw = normalize_base(row.get('대표지번'))
    bunji_info = parse_bunji(main_raw) if main_raw else None

    if loc:
        out.append((normalize_city_prefix(loc, city),'loc_raw'))
        simp = simplify_location(loc)
        if simp and simp != loc:
            out.append((normalize_city_prefix(simp, city),'loc_simplified'))
    # 조합 (자치구 + 법정동 + 대표지번)
    if gu and dong and main_raw:
        # 대표지번 전체 (원문 그대로)
        out.append((normalize_city_prefix(f"{gu} {dong} {main_raw}", city),'combo_full'))
        # 파싱 성공 시 본번/본번-부번 변형 후보 추가
        if bunji_info:
            # 산 여부 유지/비유지 버전
            san_prefix = '산' if bunji_info['san']=='Y' else ''
            if bunji_info['sub']:
                # full (산여부+본-부)
                out.append((normalize_city_prefix(f"{gu} {dong} {san_prefix}{bunji_info['main']}-{bunji_info['sub']}", city),'combo_parsed_full'))
                # 본번만 (산여부 유지)
                out.append((normalize_city_prefix(f"{gu} {dong} {san_prefix}{bunji_info['main']}", city),'combo_main_only'))
                # 산 제거 변형(산 번지일 경우) - 일부 데이터 소스가 산 누락
                if bunji_info['san']=='Y':
                    out.append((normalize_city_prefix(f"{gu} {dong} {bunji_info['main']}-{bunji_info['sub']}", city),'combo_full_no_san'))
                    out.append((normalize_city_prefix(f"{gu} {dong} {bunji_info['main']}", city),'combo_main_no_san'))
            else:
                # 부번 없음 → 산 있는/없는 2종
                if bunji_info['san']=='Y':
                    out.append((normalize_city_prefix(f"{gu} {dong} {bunji_info['main']}", city),'combo_main_no_san'))
    # 동+번지 추출
    if loc:
        d2_info, bunji2 = extract_dong_bunji(loc)
        if d2_info and bunji2:
            # 상세 파싱 재시도
            _, detailed = extract_dong_bunji_detailed(loc)
            if detailed:
                san_prefix = '산' if detailed['san']=='Y' else ''
                if detailed['sub']:
                    out.append((normalize_city_prefix(f"{d2_info} {san_prefix}{detailed['main']}-{detailed['sub']}", city),'loc_dong_bunji_full'))
                    out.append((normalize_city_prefix(f"{d2_info} {san_prefix}{detailed['main']}", city),'loc_dong_main'))
                    if detailed['san']=='Y':
                        out.append((normalize_city_prefix(f"{d2_info} {detailed['main']}-{detailed['sub']}", city),'loc_dong_bunji_full_no_san'))
                else:
                    out.append((normalize_city_prefix(f"{d2_info} {san_prefix}{detailed['main']}", city),'loc_dong_main'))
                    if detailed['san']=='Y':
                        out.append((normalize_city_prefix(f"{d2_info} {detailed['main']}", city),'loc_dong_main_no_san'))
            # 원래 추출 문자열 (하위호환 태그)
            out.append((normalize_city_prefix(f"{d2_info} {bunji2}", city),'loc_dong_bunji'))
        if d2_info:
            out.append((normalize_city_prefix(d2_info, city),'loc_dong'))
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
                elif tag in ('combo_full','combo_parsed_full','combo_full_no_san'):
                    status='combo_ok'
                elif tag in ('combo_main_only','combo_main_no_san'):
                    status='combo_main_ok'
                elif tag in ('loc_dong_bunji','loc_dong_bunji_full','loc_dong_bunji_full_no_san'):
                    status='dong_bunji_ok'
                elif tag in ('loc_dong_main','loc_dong_main_no_san'):
                    status='dong_main_ok'
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
    # Deprecated / backward compatibility: used previously in examples
    ap.add_argument('--retry-simplify', action='store_true', help='(Deprecated) 단순화 재시도 플래그 - 현재 자동 처리되어 무시됨')
    ap.add_argument('--full-output', action='store_true', help='전체 원본 컬럼 + geocode_* 상세 컬럼까지 포함 (기본은 lat,lon,success 최소)')
    return ap.parse_args()

# ---------------- main ----------------

def main():
    args=parse_args()
    if getattr(args, 'retry_simplify', False):
        print('[INFO] --retry-simplify 플래그는 더 이상 필요하지 않아 무시됩니다 (자동 단순화 내장).')
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
    # 성공여부 컬럼 생성
    result['success']=result['lat'].notna() & result['lon'].notna()
    if not args.full_output:
        minimal_df=result[['lat','lon','success']].copy()
        if args.out.suffix.lower()=='.parquet':
            minimal_df.to_parquet(args.out, index=False)
        else:
            minimal_df.to_csv(args.out, index=False, encoding='utf-8-sig')
    else:
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
