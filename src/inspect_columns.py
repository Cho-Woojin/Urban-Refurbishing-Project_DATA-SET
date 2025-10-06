"""
각 CSV의 컬럼 목록과 간단한 통계를 출력합니다.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ENCODINGS_TRY = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {path} (마지막 에러: {last_err})")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "DATA"
    files = sorted([p for p in data_dir.glob("*.csv")])
    for p in files:
        df = read_csv_smart(p)
        cols = [c.strip().replace("\ufeff", "") for c in df.columns]
        print("\n==>", p.name, f"rows={len(df)}, cols={len(cols)}")
        for c in cols:
            print(" -", c)


if __name__ == "__main__":
    main()
