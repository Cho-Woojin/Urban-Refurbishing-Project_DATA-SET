"""[Deprecated]
이 스크립트는 더 이상 유지보수하지 않습니다.

대체 스크립트: `join_pipeline.py`
사유:
  - `join_pipeline.py` 가 표준화 키 생성/다단계 우선순위 조인/부분문자열 보강/진단 출력 등
    고도화된 기능을 모두 포함하고 있으며, `join_csv.py` 내용은 그 초기 단순 버전입니다.
  - 중복 유지 시 혼동을 야기하므로 최소 표면 API만 남기고 즉시 종료하도록 변경.

사용자는 이 파일 대신 아래 명령을 사용하세요:
  python -m src.join_pipeline

필요 시 git 기록을 통해 과거 구현을 확인할 수 있습니다.
"""
from __future__ import annotations
import sys

def main():  # pragma: no cover - 단순 안내용
    print(
        "[join_csv.py Deprecated] 이 스크립트는 제거 예정입니다. 대신 'python -m src.join_pipeline' 을 사용하세요.",
        file=sys.stderr,
    )
    return 1

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
