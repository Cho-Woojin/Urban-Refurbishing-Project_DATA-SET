"""
CSV 조인 스크립트
- DATA 폴더의 여러 CSV를 불러와 공통 키로 병합하여 outputs 폴더에 저장합니다.
- 컬럼 한국어가 포함되어 있으므로 UTF-8-SIG로 입출력 처리합니다.

빠른 실행 (PowerShell):
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python .\src\join_csv.py

환경 변수(optional):
- OUTPUT_DIR: 결과 저장 경로 (기본: outputs)
- DATA_DIR: 원본 데이터 경로 (기본: DATA)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import yaml

ENCODINGS_TRY = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def read_csv_smart(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (마지막 에러: {last_err})")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 공백 제거, 소문자, 특수문자 단순화
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("__+", "_", regex=True)
        .str.lower()
    )
    return df


def extract_dong_and_bunji(addr: str | float | int | None) -> tuple[str, str]:
    """주소 문자열에서 동/번지 추출(보수적).
    - 동: ([가-힣0-9]+동)
    - 번지: 동 이후 산?숫자(-숫자) 패턴
    """
    import re
    if addr is None:
        return "", ""
    s = str(addr).replace("\ufeff", "").strip()
    s = re.sub(r"\s+", "", s)
    m_d = re.search(r"([가-힣0-9]+동)", s)
    dong = m_d.group(1) if m_d else ""
    bunji = ""
    if dong:
        tail = s.split(dong, 1)[1]
        m_b = re.search(r"산?([0-9]+(?:-[0-9]+)?)", tail)
        bunji = m_b.group(1) if m_b else ""
    else:
        m_b = re.search(r"([0-9]+(?:-[0-9]+)?)", s)
        bunji = m_b.group(1) if m_b else ""
    return dong, bunji


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv("DATA_DIR", repo_root / "DATA"))
    out_dir = Path(os.getenv("OUTPUT_DIR", repo_root / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 입력 파일들
    files = [
        "base_filtered_no_urban_redev.csv",
        "서울열린데이터광장_642건.csv",
        "정비몽땅_447건.csv",
    ]

    dfs: dict[str, pd.DataFrame] = {}
    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"경고: 파일 없음 → {fpath}")
            continue
        df = read_csv_smart(fpath)
        df = normalize_columns(df)
        dfs[fname] = df
        print(f"로드 완료: {fname} (shape={df.shape})")

    if not dfs:
        raise SystemExit("읽은 파일이 없습니다. DATA 경로와 파일명을 확인하세요.")

    # 구성 파일에서 표준 키 목록과 파일별 실제 컬럼 매핑을 읽기
    cfg_path = repo_root / "config" / "schema_keys.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        standard_keys: list[str] = [s.strip() for s in cfg.get("standard_keys", [])]
        file_key_map: dict[str, dict[str, str]] = cfg.get("file_key_map", {})
    else:
        standard_keys = ["사업코드", "정비구역", "사업명", "구", "동"]
        file_key_map = {}

    # 각 DF별로 표준 키 컬럼 생성 (설정 기반)
    def add_standard_keys_for_file(df: pd.DataFrame, fname: str) -> pd.DataFrame:
        df = df.copy()
        mapping = file_key_map.get(fname, {})
        # 정규화된 컬럼명 -> 원래 컬럼명 사전
        norm_to_orig = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
        for std in standard_keys:
            actual = mapping.get(std)
            if not actual:
                continue
            key = actual.lower().strip().replace(" ", "_")
            src = norm_to_orig.get(key)
            if src and (src != std):
                df[std] = df[src]
        # 문자열 표준화
        for c in standard_keys:
            if c in df.columns:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.replace("-", "", regex=False)
                    .str.replace("·", "", regex=False)
                )
        return df

    for fname in list(dfs.keys()):
        dfs[fname] = add_standard_keys_for_file(dfs[fname], fname)
        # 위치 관련 파생 키 생성: 동_std, 번지_std (가능하면)
        # 위치/법정동/행정동 등의 컬럼 후보를 찾아서 파생
        cand_loc = [
            "위치", "법정동", "행정동", "주소", "지번", "법정주소"
        ]
        col_loc = next((c for c in cand_loc if c in dfs[fname].columns), None)
        if col_loc:
            dongs, bunjis = zip(*dfs[fname][col_loc].map(extract_dong_and_bunji))
            dfs[fname]["동_std"] = list(dongs)
            dfs[fname]["번지_std"] = list(bunjis)

    # 병합 전략: 우선순위가 높은 테이블을 기준으로 left join 단계적 수행
    base_name = "base_filtered_no_urban_redev.csv" if "base_filtered_no_urban_redev.csv" in dfs else next(iter(dfs))
    base = dfs[base_name]
    print(f"기준 테이블: {base_name} (shape={base.shape})")

    # 더 세분화된 조인 키 우선순위(상단일수록 엄격)
    join_keys_priority = [
        ["구역명", "자치구", "동_std", "번지_std"],  # 명칭+행정동+번지
        ["구역명", "자치구", "동_std"],
        ["구역명", "자치구"],
        ["사업명", "자치구", "동_std", "번지_std"],
        ["사업명", "자치구", "동_std"],
        ["사업명", "자치구"],                       # 가장 느슨
    ]

    # 기준 테이블에 조인키가 있는지 빠르게 점검
    for keys in join_keys_priority:
        missing = [k for k in keys if k not in base.columns]
        if missing:
            print(f"참고: 기준 테이블에 없음 -> {missing}")

    merged = base.copy()
    for name, df in dfs.items():
        if name == base_name:
            continue
        joined = None
        for keys in join_keys_priority:
            if all(k in merged.columns and k in df.columns for k in keys):
                try:
                    joined = merged.merge(
                        df,
                        on=keys,
                        how="left",
                        suffixes=(None, f"_{Path(name).stem}")
                    )
                    print(f"조인 성공: {name} on {keys} → shape={joined.shape}")
                    break
                except Exception as e:
                    print(f"조인 실패: {name} on {keys} → {e}")
        if joined is None:
            print(f"스킵: {name} (공통 키 없음)")
        else:
            merged = joined

    out_path = out_dir / "merged.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {out_path} (shape={merged.shape})")


if __name__ == "__main__":
    main()
