"""
Phase 1 データ収集をまとめて実行するエントリーポイント。

実行: py -3.13 run_phase1.py
オプション:
  --statcast-years  2024 2025   取得する年（デフォルト: 2024 2025）
  --schedule-days   30          スケジュール取得日数
  --skip-statcast               Statcast をスキップ
  --skip-schedule               スケジュールをスキップ
  --skip-opponents              対戦相手をスキップ
  --force                       既存キャッシュを上書き
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PYTHON = sys.executable


def run(script: str, *args: str) -> int:
    cmd = [PYTHON, str(HERE / script), *args]
    print(f"\n{'='*60}")
    print(f"実行: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 一括データ収集")
    parser.add_argument("--statcast-years", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--schedule-days", type=int, default=30)
    parser.add_argument("--skip-statcast", action="store_true")
    parser.add_argument("--skip-schedule", action="store_true")
    parser.add_argument("--skip-opponents", action="store_true")
    parser.add_argument("--force", action="store_true")
    a = parser.parse_args()

    force_flag = ["--force"] if a.force else []

    # ① 大谷 Statcast
    if not a.skip_statcast:
        years_args = ["--years"] + [str(y) for y in a.statcast_years]
        rc = run("collect_statcast.py", *years_args, *force_flag)
        if rc != 0:
            print(f"!! collect_statcast.py 失敗 (exit {rc})")

    # ② ドジャーススケジュール
    if not a.skip_schedule:
        rc = run("collect_schedule.py", "--days", str(a.schedule_days), "--roster")
        if rc != 0:
            print(f"!! collect_schedule.py 失敗 (exit {rc})")

    # ③ 対戦相手データ（スケジュールに依存）
    if not a.skip_opponents:
        rc = run("collect_opponents.py", "--year", str(max(a.statcast_years)), *force_flag)
        if rc != 0:
            print(f"!! collect_opponents.py 失敗 (exit {rc})")

    print("\n=== Phase 1 完了 ===")


if __name__ == "__main__":
    main()
