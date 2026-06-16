"""
大谷翔平 Statcast データ収集スクリプト
- 投手データ (pitcher=660271)
- 打者データ (batter=660271)
を年ごとに取得して parquet キャッシュに保存する。

実行: py -3.13 collect_statcast.py [--years 2021 2022 2023 2024 2025]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pybaseball

OHTANI_ID = 660271
DATA_DIR = Path(__file__).parent / "data" / "statcast"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASON_DATES = {
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-10-01"),
    2025: ("2025-03-27", "2025-10-01"),
    2026: ("2026-03-26", "2026-10-01"),
}

STATCAST_COLS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter", "stand", "p_throws",
    "pitch_type", "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension",
    "plate_x", "plate_z",
    "zone", "type", "description", "events", "bb_type",
    "balls", "strikes",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "delta_home_win_exp", "delta_run_exp",
    "home_team", "away_team", "inning", "inning_topbot",
]


def _cache_path(role: str, year: int) -> Path:
    return DATA_DIR / f"ohtani_{role}_{year}.parquet"


def _fetch_pitcher(year: int) -> pd.DataFrame:
    start, end = SEASON_DATES[year]
    print(f"  [投手] {year}: {start} → {end} 取得中...")
    pybaseball.cache.enable()
    df = pybaseball.statcast_pitcher(start, end, player_id=OHTANI_ID)
    return df


def _fetch_batter(year: int) -> pd.DataFrame:
    start, end = SEASON_DATES[year]
    print(f"  [打者] {year}: {start} → {end} 取得中...")
    pybaseball.cache.enable()
    df = pybaseball.statcast_batter(start, end, player_id=OHTANI_ID)
    return df


def _save(df: pd.DataFrame, path: Path, role: str, year: int) -> None:
    if df is None or df.empty:
        print(f"    !! データなし ({role} {year})")
        return
    cols = [c for c in STATCAST_COLS if c in df.columns]
    df[cols].to_parquet(path, index=False)
    print(f"    -> {len(df):,} 行保存: {path.name}")


def collect(years: list[int], force: bool = False) -> None:
    for year in years:
        if year not in SEASON_DATES:
            print(f"  !! {year} は未対応 (2021-2025)")
            continue

        for role, fetch_fn in [("pitcher", _fetch_pitcher), ("batter", _fetch_batter)]:
            path = _cache_path(role, year)
            if path.exists() and not force:
                rows = pd.read_parquet(path).shape[0]
                print(f"  [SKIP] {path.name} 既存 ({rows:,} 行)")
                continue
            try:
                df = fetch_fn(year)
                _save(df, path, role, year)
            except Exception as e:
                print(f"    !! エラー ({role} {year}): {e}")


def summarize() -> None:
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        print("キャッシュファイルなし")
        return
    print("\n=== キャッシュ一覧 ===")
    for f in files:
        df = pd.read_parquet(f)
        print(f"  {f.name:40s}  {len(df):>6,} 行  {f.stat().st_size/1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="大谷 Statcast データ収集")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(SEASON_DATES.keys()),
        help="取得する年（デフォルト: 2021-2025）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存キャッシュを上書き",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="キャッシュ一覧を表示して終了",
    )
    args = parser.parse_args()

    if args.summary:
        summarize()
        return

    print(f"対象年: {args.years}")
    collect(args.years, force=args.force)
    summarize()


if __name__ == "__main__":
    main()
