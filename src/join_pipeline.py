"""
요구사항 기반 CSV 단계별 조인 파이프라인

구조 개요:
- 기준 테이블: "(25년 3월기준) 서울시 정비사업 추진현황.csv"
- 조인 순서: 기준 → "서울열린데이터광장_642건.csv" → "정비몽땅_447건.csv"
- 표준 조인 키: (자치구, 구역명, 위치, 사업유형)
    - 위치는 내부적으로 (행정동=동_std, 번지수=번지_std)로 파생하여 매칭
    - 자치구는 완전 일치, 사업유형은 "재개발"/"재건축"만 허용
- 기준 테이블의 원본 컬럼은 모두 유지(Left Join), 외부 데이터는 필요한 컬럼만 선택

핵심 함수:
- read_csv_smart: 인코딩을 자동 시도하여 CSV 로드
- extract_dong_and_bunji: 위치 문자열에서 (동, 번지) 추출
- add_std_keys: 각 데이터셋에서 표준화 키 컬럼 생성
- select_columns_for_join: 조인에 필요한 컬럼만 선별(충돌 방지용 접미어 포함)
- left_join_with_fallback: 키 결측을 고려해 우선순위별로 점진적 Left Join

실행(파워셸):
    python .\\src\\join_pipeline.py
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml

ENCODINGS_TRY = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


# ---------- I/O 유틸 ----------
def read_csv_smart(path: Path) -> pd.DataFrame:
    """여러 인코딩 후보를 순차적으로 시도하여 CSV를 DataFrame으로 읽습니다.

    Args:
        path: 파일 경로(Path)
    Returns:
        pandas.DataFrame: 로드된 데이터프레임
    Raises:
        RuntimeError: 모든 인코딩 시도가 실패한 경우
    """
    last_err: Exception | None = None
    for enc in ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (마지막 에러: {last_err})")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """DataFrame을 UTF-8-SIG로 저장합니다. 부모 폴더가 없으면 생성합니다.

    Args:
        df: 저장할 데이터프레임
        path: 저장 경로
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------- 정규화/파생 ----------
def norm_text(s: str | float | int | None) -> str:
    """텍스트 표준화: BOM/공백 제거 및 단순화.

    - 선행/후행 공백 제거
    - 내부 공백 제거(연속 공백 포함)
    - 하이픈은 유지
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "").strip()
    # 공백/특수문자 단순화
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "-")  # 하이픈 유지
    return s

# 매칭에서 의미 없는 단어(사업명 토큰화 시 제외)
STOPWORDS = {
    "재정비", "주택", "아파트", "재개발", "정비", "사업", "구역", "조합",
    "설립", "추진위원회", "공공", "테스트"
}

# 명칭 토큰의 불필요 접미어(말꼬리) 제거 대상
SUFFIXES = {"구역", "지구", "아파트", "빌라", "타운", "주공", "시범", "맨션", "동"}


def first_token_meaningful(s: str | float | int | None) -> str:
    """문자열에서 불용어가 아닌 첫 의미 단어 추출(공백 보존, 접미어 제거)."""
    if s is None:
        return ""
    v = str(s).replace("\ufeff", "").strip()
    if not v:
        return ""
    import re as _re
    cleaned = _re.sub(r"[^가-힣0-9]", " ", v)
    for tok in cleaned.split():
        if not tok:
            continue
        # 접미어 제거
        for suf in sorted(SUFFIXES, key=len, reverse=True):
            if tok.endswith(suf) and len(tok) > len(suf):
                tok = tok[: -len(suf)]
                break
        if tok and tok not in STOPWORDS:
            return tok
    return ""


def name_core_token(s: str | float | int | None) -> str:
    """명칭에서 핵심 토큰(한글+선택적 숫자)을 추출. 예: '신당10구역' → '신당10'"""
    if s is None:
        return ""
    import re as _re
    v = str(s).replace("\ufeff", "").strip()
    if not v:
        return ""
    cleaned = _re.sub(r"[^가-힣0-9]", " ", v)
    # 가장 먼저 나오는 한글+선택적 숫자 블록
    for tok in cleaned.split():
        # 접미어 제거
        for suf in sorted(SUFFIXES, key=len, reverse=True):
            if tok.endswith(suf) and len(tok) > len(suf):
                tok = tok[: -len(suf)]
                break
        m = _re.match(r"^[가-힣]+[0-9]*$", tok)
        if m:
            return m.group(0)
    return ""


def std_gu(s: str | float | int | None) -> str:
    """자치구 표준화(현재는 공백 제거 정도). 필요시 약어/오타 교정 로직 확장 가능."""
    return norm_text(s)


def std_project_type(raw: str | float | int | None) -> str:
    """사업유형 표준화: 재개발/재건축만 허용.

    Returns:
        "재개발" | "재건축" | ""(허용되지 않는 값)
    """
    v = norm_text(raw)
    if not v:
        return ""
    # 키워드 포함 판정
    if "재개발" in v:
        return "재개발"
    if "재건축" in v:
        return "재건축"
    return ""  # 그 외는 매칭에서 제외


def extract_dong_and_bunji(addr: str | float | int | None) -> tuple[str, str]:
    """주소/위치 문자열에서 동, 번지 추출.
    - 동: 마지막 '동' 토큰 기준으로 추출 (예: '가락동', '상계1동')
    - 번지: 동 이후 첫 숫자 블록(예: '123-45', '123')
    - 데이터마다 편차가 크므로 보수적으로 추출하고 없으면 빈 문자열 반환
    """
    s = norm_text(addr)
    if not s:
        return "", ""

    # 동 추출: '...동' 또는 '...동n가'까지 포함해서 포착
    m_dong = re.search(r"([가-힣]+(?:[0-9]+)?동(?:[0-9]+가)?)", s)
    dong = m_dong.group(1) if m_dong else ""

    bunji = ""
    if dong:
        # 동 이후 부분에서 번지 패턴 찾기 (선행 'n가' 토큰, 구분자 제거 후 탐색)
        tail = s.split(dong, 1)[1]
        tail = re.sub(r"^\s*[,·]*\s*", "", tail)
        m_bunji = re.search(r"산?([0-9]+(?:-[0-9]+)?)", tail)
        bunji = m_bunji.group(1) if m_bunji else ""
    else:
        # 동이 없으면 전체에서 숫자 블록을 한 번 시도
        m_bunji = re.search(r"([0-9]+(?:-[0-9]+)?)", s)
        bunji = m_bunji.group(1) if m_bunji else ""

    return dong, bunji


@dataclass
class KeyMapping:
    gu: list[str]
    area: list[str]  # 구역명(= 타 데이터의 사업명/정비구역명칭)
    loc: list[str]   # 위치(주소계 열)
    ptype: list[str] # 사업유형 후보 컬럼들


# 데이터셋별 컬럼 후보 설정
BASE_MAP = KeyMapping(
    gu=["자치구"],
    area=["구역명"],
    loc=["위치"],
    ptype=["사업유형", "사업유형(대분류)", "사업유형_대분류", "사업유형_중분류", "사업유형_소분류"],
)
OPEN_MAP = KeyMapping(
    gu=["자치구"],
    area=["정비구역명칭", "구역명", "사업명"],
    loc=["위치", "법정동"],  # 위치가 없으면 법정동 기반으로 동만 추출
    ptype=["사업유형", "사업구분", "사업분류", "유형", "사업대분류", "사업세부분류"],
)
JBM_MAP = KeyMapping(
    gu=["자치구"],
    area=["사업명", "정비구역명칭", "구역명"],
    loc=["위치"],
    ptype=["사업유형", "유형", "사업구분", "사업대분류", "사업세부분류"],
)


def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 리스트 중 DataFrame에 존재하는 첫 번째 컬럼명을 반환."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_std_keys(df: pd.DataFrame, km: KeyMapping) -> pd.DataFrame:
    """주어진 KeyMapping을 바탕으로 표준 키 컬럼 생성.

    생성 컬럼:
      - 자치구_std, 구역명_std, 동_std, 번지_std, 사업유형_std
    """
    df = df.copy()
    gu_col = pick_first_col(df, km.gu)
    area_col = pick_first_col(df, km.area)
    loc_col = pick_first_col(df, km.loc)
    ptype_col = pick_first_col(df, km.ptype)

    df["자치구_std"] = df[gu_col].map(std_gu) if gu_col else ""
    df["구역명_std"] = df[area_col].map(norm_text) if area_col else ""

    # 위치 → 동/번지 (여러 후보 소스 결합)
    # 우선순위: 위치 > 대표지번 > 정비구역명칭 > 법정동
    loc_sources = []
    if loc_col:
        loc_sources.append(df[loc_col].astype(str))
    for extra in ["대표지번", "정비구역명칭", "법정동"]:
        if extra in df.columns:
            loc_sources.append(df[extra].astype(str))
    if loc_sources:
        loc_text = loc_sources[0]
        for s in loc_sources[1:]:
            loc_text = loc_text.fillna("") + " " + s.fillna("")
        dongs, bunjis = zip(*loc_text.map(extract_dong_and_bunji))
        df["동_std"] = list(dongs)
        df["번지_std"] = list(bunjis)
        # 전체 위치 텍스트 정규화도 보존(부분문자열 매칭용)
        df["위치정규_std"] = pd.Series(loc_text).map(norm_text).astype(str)
    else:
        df["동_std"] = ""
        df["번지_std"] = ""
        df["위치정규_std"] = ""

    # 동 뿌리(…동2가 → …동), 번지 본번(123-45 → 123) 파생
    def _dong_root(v: str) -> str:
        if not v:
            return ""
        # 중곡4동 → 중곡동, 성수동1가 → 성수동
        m = re.match(r"^([가-힣]+)(?:\d+)?동$", v)
        if m:
            return m.group(1) + "동"
        m2 = re.match(r"^([가-힣]+동)(?:\d+가)?$", v)
        return m2.group(1) if m2 else v
    df["동뿌리_std"] = df["동_std"].map(_dong_root)
    df["번지본번_std"] = df["번지_std"].astype(str).str.split("-").str[0].fillna("")

    # 사업유형
    if ptype_col:
        df["사업유형_std"] = df[ptype_col].map(std_project_type)
    else:
        df["사업유형_std"] = ""

    # 명칭 첫단어 표준화(구역명/정비구역명칭/사업명 순 우선)
    name_source = None
    if "구역명" in df.columns:
        name_source = df["구역명"]
    elif "정비구역명칭" in df.columns:
        name_source = df["정비구역명칭"]
    elif "사업명" in df.columns:
        name_source = df["사업명"]
    if name_source is not None:
        df["명칭첫단어_std"] = name_source.map(first_token_meaningful)
    else:
        df["명칭첫단어_std"] = ""

    # 명칭 핵심 키
    if name_source is not None:
        df["명칭핵심_std"] = name_source.map(name_core_token)
    else:
        df["명칭핵심_std"] = ""

    # 명칭 정규 키(완전 정규화 텍스트)
    if name_source is not None:
        df["명칭정규_std"] = name_source.map(norm_text)
    else:
        df["명칭정규_std"] = ""

    return df


# ---------- 필요 컬럼 선별 ----------
def select_columns_for_join(df: pd.DataFrame, extra_cols: list[str], suffix: str) -> pd.DataFrame:
    """조인에 필요한 표준 키 + 추가 컬럼만 유지.

    - 추가 컬럼은 기준 테이블과의 이름 충돌을 피하기 위해 접미어를 부여합니다.
    """
    keep_keys = [
        "자치구_std", "구역명_std", "동_std", "번지_std", "사업유형_std", "명칭첫단어_std",
        "동뿌리_std", "번지본번_std", "명칭핵심_std", "명칭정규_std", "위치정규_std",
    ]
    cols = [c for c in keep_keys if c in df.columns]

    extras_kept = []
    for c in extra_cols:
        if c in df.columns and c not in cols:
            newc = f"{c}{suffix}"
            df = df.rename(columns={c: newc})
            extras_kept.append(newc)
            cols.append(newc)
    return df[cols]


# ---------- 조인 로직 ----------
def exclude_urban_redevelopment(df: pd.DataFrame, km: KeyMapping) -> pd.DataFrame:
    """'도시정비형 재개발' 문자열이 사업유형 관련 컬럼에 포함된 행을 제외합니다.

    - 대상 컬럼: km.ptype 후보 중 존재하는 모든 컬럼 검사
    - 공백 변형을 허용(예: '도시정비형   재개발')
    """
    import re as _re
    cols = [c for c in km.ptype if c in df.columns]
    if not cols:
        return df
    pattern = _re.compile(r"도시정비형\s*재개발")
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | df[c].astype(str).str.contains(pattern, na=False)
    return df.loc[~mask].copy()
def left_join_with_fallback(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    """우선순위 키를 바꿔가며 첫 성공 시점에 Left Join을 수행합니다.

    우선순위: 위치(동/번지) → 자치구+위치 → 자치구+동뿌리+본번 → 자치구+동 → 명칭첫단어+자치구 → 기존 조합

    Args:
        base: 기준 테이블(모든 원 컬럼 유지)
        other: 조인 대상(표준 키 및 선택 컬럼만 포함)
        label: 로그용 라벨
    Returns:
        병합된 DataFrame
    """
    join_plans = [
        ["동_std", "번지_std"],
        ["자치구_std", "동_std", "번지_std"],
        ["자치구_std", "동뿌리_std", "번지본번_std"],
        ["자치구_std", "동_std"],
        ["자치구_std", "명칭핵심_std"],
        ["명칭첫단어_std", "자치구_std"],
        ["자치구_std", "구역명_std", "동_std", "번지_std", "사업유형_std"],
        ["자치구_std", "구역명_std", "동_std", "사업유형_std"],
        ["자치구_std", "구역명_std", "사업유형_std"],
    ]

    merged = base.copy()
    # 진단 저장 경로 준비
    repo = Path(__file__).resolve().parents[1]
    diag_dir = repo / "outputs" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    any_plan_used = False
    for keys in join_plans:
        if not all(k in merged.columns and k in other.columns for k in keys):
            continue

    # 우측 데이터 표준키는 선택된 조인키만 유지하고 나머지 표준키는 제거
        std_cols = {"자치구_std", "구역명_std", "동_std", "번지_std", "사업유형_std", "명칭첫단어_std", "동뿌리_std", "번지본번_std", "명칭핵심_std", "명칭정규_std", "위치정규_std"}
        non_std_cols = [c for c in other.columns if c not in std_cols]
        keep_cols = list(dict.fromkeys(list(keys) + non_std_cols))
        other_use = other[keep_cols].copy()
        any_plan_used = True

        key_counts = other_use[keys].value_counts()
        dup_counts = key_counts[key_counts > 1]
        n_dup_keys = int(dup_counts.shape[0])
        n_dup_rows = int(dup_counts.sum())

        base_keys_df = merged[keys].copy()
        if isinstance(dup_counts.index, pd.MultiIndex):
            dup_keys_df = dup_counts.index.to_frame(index=False)
        else:
            dup_keys_df = pd.DataFrame({keys[0]: dup_counts.index})
        base_match_with_dups = base_keys_df.merge(dup_keys_df.drop_duplicates(), on=keys, how="inner").shape[0]

        dup_rows = other_use.merge(dup_keys_df, on=keys, how="inner")
        dup_rows_path = diag_dir / f"{label}_dups_on_{'-'.join(keys)}.csv"
        write_csv(dup_rows.head(1000), dup_rows_path)

        if n_dup_keys > 0:
            print(f"진단[{label}] 키 {keys} 기준 중복키 {n_dup_keys}개, 중복행 {n_dup_rows}개, 기준 교집합 {base_match_with_dups}행 → 중복 제거 후 조인")
        other_dedup = other_use.drop_duplicates(subset=keys, keep="first")

        # 조인 수행: 이번 플랜으로 보강될 수 있는 '추가 컬럼'만 결측 채우기
        extras = [c for c in other_use.columns if c not in keys]
        before_vals = None
        if extras:
            before_vals = merged[extras].isna().any(axis=1) if all(c in merged.columns for c in extras) else None

        tmp = merged.merge(other_dedup, on=keys, how="left", suffixes=("", "_y"))

        new_filled_rows = 0
        # 각 추가 컬럼에 대해: 기존 값이 없으면 y에서 채우고, y 컬럼 제거
        for col in extras:
            ycol = f"{col}_y"
            if ycol in tmp.columns:
                if col in tmp.columns:
                    # 기존 값이 NaN인 곳만 채움
                    before_na = tmp[col].isna()
                    tmp.loc[before_na, col] = tmp.loc[before_na, ycol]
                else:
                    # 기존에 없던 컬럼이면 새로 만듦
                    tmp[col] = tmp[ycol]
                tmp = tmp.drop(columns=[ycol])

        if extras:
            # 새로 채워진 행 수 추정
            after_vals = tmp[extras].isna().any(axis=1) if all(c in tmp.columns for c in extras) else None
            if before_vals is not None and after_vals is not None:
                new_filled_rows = int((before_vals & ~after_vals).sum())

        merged = tmp
        print(f"조인 시도[{label}] on {keys} → 보강컬럼 {len(extras)}개, 신규매칭행 {new_filled_rows}")
    if not any_plan_used:
        print(f"경고: [{label}] 공통 키 미존재로 조인 스킵")

    # 보강: 열린데이터 전용 부분문자열 기반 매칭(최우선: 위치/명칭 포함)
    if label == "열린데이터":
        # other 쪽 검색 코퍼스 구성(명칭정규_std + 위치정규_std + 기타 텍스트형 추가컬럼)
        std_cols = {"자치구_std", "구역명_std", "동_std", "번지_std", "사업유형_std", "명칭첫단어_std", "동뿌리_std", "번지본번_std", "명칭핵심_std", "명칭정규_std", "위치정규_std"}
        other_extras = [c for c in other.columns if c not in std_cols]
        if other_extras:
            def _row_corpus(r: pd.Series) -> str:
                parts = []
                if "명칭정규_std" in other.columns:
                    parts.append(str(r.get("명칭정규_std", "")))
                if "위치정규_std" in other.columns:
                    parts.append(str(r.get("위치정규_std", "")))
                for c in other_extras:
                    v = r.get(c, None)
                    if pd.notna(v):
                        parts.append(str(v))
                return norm_text(" ".join(parts))

            other_corpus = other.apply(_row_corpus, axis=1)

            # 보강할 대상(아직 추가 컬럼 중 결측이 있는 행)
            need_cols = [c for c in other_extras if c in merged.columns] or other_extras
            if need_cols:
                before_any_na = merged[need_cols].isna().any(axis=1) if all(c in merged.columns for c in need_cols) else pd.Series(True, index=merged.index)
                filled = 0

                for idx, row in merged[before_any_na].iterrows():
                    # 이 행에서 채울 수 있는 컬럼 후보
                    to_fill_cols = [c for c in other_extras if (c not in merged.columns) or pd.isna(row.get(c, pd.NA))]
                    if not to_fill_cols:
                        continue

                    needles = []
                    # 기준 위치/명칭 기반 탐색어 구성
                    v_loc = str(row.get("위치정규_std", ""))
                    v_name = str(row.get("명칭정규_std", ""))
                    v_dongbunji = f"{row.get('동_std','')}{row.get('번지_std','')}"
                    v_rootbun = f"{row.get('동뿌리_std','')}{row.get('번지본번_std','')}"
                    for v in [v_loc, v_dongbunji, v_rootbun, v_name]:
                        nv = norm_text(v)
                        if nv:
                            needles.append(nv)

                    match_idx = None
                    for nd in needles:
                        mask = other_corpus.str.contains(nd, na=False, regex=False)
                        if not mask.any():
                            continue
                        # 동일 자치구 우선
                        if "자치구_std" in other.columns and "자치구_std" in merged.columns:
                            same_gu = other["자치구_std"].astype(str) == str(row.get("자치구_std", ""))
                            mask2 = mask & same_gu
                            if mask2.any():
                                mask = mask2
                        # 첫 매칭 1건 사용
                        try:
                            match_idx = mask[mask].index[0]
                        except Exception:
                            match_idx = None
                        if match_idx is not None:
                            break

                    if match_idx is None:
                        continue

                    cand = other.loc[match_idx]
                    changed = False
                    for c in other_extras:
                        yv = cand.get(c, pd.NA)
                        if c in merged.columns:
                            if pd.isna(merged.at[idx, c]) and pd.notna(yv):
                                merged.at[idx, c] = yv
                                changed = True
                        else:
                            merged[c] = pd.NA
                            if pd.notna(yv):
                                merged.at[idx, c] = yv
                                changed = True
                    if changed:
                        filled += 1

                print(f"부분문자열 보강[{label}] → 신규매칭행 {filled}")

    return merged


def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / "DATA"
    out = repo / "outputs"
    cfg_cols = repo / "config" / "join_keep_columns.yaml"

    base_path = data / "(25년 3월기준) 서울시 정비사업 추진현황.csv"
    open_path = data / "서울열린데이터광장_642건.csv"
    jbm_path = data / "정비몽땅_447건.csv"

    # 1) 로드
    base = read_csv_smart(base_path)
    original_cols = list(base.columns)  # 기준 CSV의 원본 컬럼 보존용
    open_df = read_csv_smart(open_path)
    jbm_df = read_csv_smart(jbm_path)

    # 2) 표준키 생성
    base = add_std_keys(base, BASE_MAP)
    open_df = add_std_keys(open_df, OPEN_MAP)
    jbm_df = add_std_keys(jbm_df, JBM_MAP)

    # 2-1) 기준 테이블에서 '도시정비형 재개발' 제외
    base = exclude_urban_redevelopment(base, BASE_MAP)

    # 3) 사업유형 필터: 재개발/재건축만
    allow_types = {"재개발", "재건축"}
    base = base[base["사업유형_std"].isin(allow_types)].copy()

    # 3-1) 필터링된 기준 테이블만 별도 CSV로 저장(원본 컬럼만 유지)
    base_filtered_original = base.loc[:, [c for c in original_cols if c in base.columns]].copy()
    # '도시정비형 재개발' 제외 + 허용 사업유형 필터 적용된 기준 CSV 저장
    write_csv(base_filtered_original, out / "base_filtered_no_urban_redev.csv")
    # 외부 데이터는 후보를 넓히기 위해 사업유형으로 필터링하지 않음

    # 4) 불필요 컬럼 제거(조인키 + 추가 추출만 유지)
    #    config/join_keep_columns.yaml 에서 보존 컬럼 설정을 읽습니다.
    extras_open: list[str] = []
    extras_jbm: list[str] = []
    if cfg_cols.exists():
        with open(cfg_cols, "r", encoding="utf-8") as f:
            keep_cfg = yaml.safe_load(f) or {}
        extras_open = keep_cfg.get("서울열린데이터광장_642건.csv", []) or []
        extras_jbm = keep_cfg.get("정비몽땅_447건.csv", []) or []

    open_for_join = select_columns_for_join(open_df, extras_open, suffix="_열린데이터")
    jbm_for_join = select_columns_for_join(jbm_df, extras_jbm, suffix="_정비몽땅")

    # 5) 단계별 조인
    step1 = left_join_with_fallback(base, open_for_join, label="열린데이터")
    write_csv(step1, out / "merged_step1.csv")

    final = left_join_with_fallback(step1, jbm_for_join, label="정비몽땅")
    write_csv(final, out / "merged_final.csv")

    print(f"완료: {out / 'merged_step1.csv'} → {step1.shape}, {out / 'merged_final.csv'} → {final.shape}")


if __name__ == "__main__":
    main()
