"""
対戦相手（投手・打者）の Statcast データ取得スクリプト
スケジュールファイルから先発投手 ID を読み取り、Statcast を取得する。
打者は相手チームのロスターから取得。

実行: py -3.13 collect_opponents.py [--year 2025] [--pitcher-only]
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pybaseball
import requests

DATA_DIR_SCHEDULE = Path(__file__).parent / "data" / "schedule"
DATA_DIR_OPP = Path(__file__).parent / "data" / "opponents"
DATA_DIR_OPP.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://statsapi.mlb.com/api/v1"
OHTANI_ID = 660271

SEASON_DATES = {
    2024: ("2024-03-20", "2024-10-01"),
    2025: ("2025-03-27", "2025-10-01"),
    2026: ("2026-03-26", "2026-10-01"),
}


def _get(endpoint: str, params: dict | None = None) -> dict:
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_team_batters(team_id: int) -> list[int]:
    """チームのアクティブ打者 player_id リストを返す"""
    data = _get(f"teams/{team_id}/roster", {"rosterType": "active"})
    batters = []
    for p in data.get("roster", []):
        pos = p.get("position", {}).get("type", "")
        if pos not in ("Pitcher",):
            batters.append(p["person"]["id"])
    return batters


def _cache_path(role: str, player_id: int, year: int) -> Path:
    return DATA_DIR_OPP / f"{role}_{player_id}_{year}.parquet"


def _fetch_and_save_pitcher(player_id: int, year: int, force: bool = False) -> pd.DataFrame | None:
    path = _cache_path("pitcher", player_id, year)
    if path.exists() and not force:
        return pd.read_parquet(path)
    if year not in SEASON_DATES:
        return None
    start, end = SEASON_DATES[year]
    try:
        pybaseball.cache.enable()
        df = pybaseball.statcast_pitcher(start, end, player_id=player_id)
        if df is not None and not df.empty:
            df.to_parquet(path, index=False)
            return df
    except Exception as e:
        print(f"    !! pitcher {player_id}: {e}")
    return None


def _fetch_and_save_batter(player_id: int, year: int, force: bool = False) -> pd.DataFrame | None:
    path = _cache_path("batter", player_id, year)
    if path.exists() and not force:
        return pd.read_parquet(path)
    if year not in SEASON_DATES:
        return None
    start, end = SEASON_DATES[year]
    try:
        pybaseball.cache.enable()
        df = pybaseball.statcast_batter(start, end, player_id=player_id)
        if df is not None and not df.empty:
            df.to_parquet(path, index=False)
            return df
    except Exception as e:
        print(f"    !! batter {player_id}: {e}")
    return None


def collect_upcoming_pitchers(year: int, force: bool = False) -> None:
    """スケジュールから今後の相手先発投手を収集"""
    schedule_path = DATA_DIR_SCHEDULE / "dodgers_schedule.parquet"
    if not schedule_path.exists():
        print("スケジュールファイルがありません。先に collect_schedule.py を実行してください。")
        return

    df_sched = pd.read_parquet(schedule_path)
    pitcher_ids = df_sched["opp_starter_id"].dropna().astype(int).unique()
    print(f"相手先発投手 {len(pitcher_ids)} 人分の Statcast データを取得")

    for i, pid in enumerate(pitcher_ids, 1):
        path = _cache_path("pitcher", pid, year)
        if path.exists() and not force:
            rows = pd.read_parquet(path).shape[0]
            print(f"  [{i:2d}/{len(pitcher_ids)}] pitcher {pid}: SKIP ({rows} 行既存)")
            continue
        print(f"  [{i:2d}/{len(pitcher_ids)}] pitcher {pid}: 取得中...")
        df = _fetch_and_save_pitcher(pid, year, force=force)
        rows = len(df) if df is not None else 0
        print(f"    -> {rows} 行保存")


def collect_upcoming_batters(year: int, force: bool = False) -> None:
    """スケジュールから今後の相手チーム打者を収集"""
    schedule_path = DATA_DIR_SCHEDULE / "dodgers_schedule.parquet"
    if not schedule_path.exists():
        print("スケジュールファイルがありません。先に collect_schedule.py を実行してください。")
        return

    df_sched = pd.read_parquet(schedule_path)
    team_ids = df_sched["opponent_team_id"].dropna().astype(int).unique()
    print(f"相手チーム {len(team_ids)} チームの打者データを取得")

    all_batter_ids: set[int] = set()
    for tid in team_ids:
        try:
            batter_ids = fetch_team_batters(tid)
            all_batter_ids.update(batter_ids)
            print(f"  チーム {tid}: 打者 {len(batter_ids)} 人")
        except Exception as e:
            print(f"  !! チーム {tid} ロスター取得失敗: {e}")

    # 大谷は除外（別途 ohtani_batter_*.parquet で管理）
    all_batter_ids.discard(OHTANI_ID)
    batter_ids = sorted(all_batter_ids)
    print(f"合計打者 {len(batter_ids)} 人")

    for i, pid in enumerate(batter_ids, 1):
        path = _cache_path("batter", pid, year)
        if path.exists() and not force:
            print(f"  [{i:3d}/{len(batter_ids)}] batter {pid}: SKIP")
            continue
        print(f"  [{i:3d}/{len(batter_ids)}] batter {pid}: 取得中...")
        df = _fetch_and_save_batter(pid, year, force=force)
        rows = len(df) if df is not None else 0
        print(f"    -> {rows} 行保存")


def summarize() -> None:
    files = sorted(DATA_DIR_OPP.glob("*.parquet"))
    pitchers = [f for f in files if f.name.startswith("pitcher_")]
    batters = [f for f in files if f.name.startswith("batter_")]
    print(f"\n=== 対戦相手キャッシュ ===")
    print(f"  投手: {len(pitchers)} ファイル")
    print(f"  打者: {len(batters)} ファイル")
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  合計: {total_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="対戦相手データ収集")
    parser.add_argument("--year", type=int, default=datetime.today().year)
    parser.add_argument("--pitcher-only", action="store_true")
    parser.add_argument("--batter-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.batter_only:
        collect_upcoming_pitchers(args.year, force=args.force)
    if not args.pitcher_only:
        collect_upcoming_batters(args.year, force=args.force)

    summarize()


if __name__ == "__main__":
    main()
