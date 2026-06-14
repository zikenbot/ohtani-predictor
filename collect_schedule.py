"""
ドジャース 今後の試合スケジュール・先発投手取得スクリプト
MLB Stats API (statsapi.mlb.com) を使用。

実行: py -3.13 collect_schedule.py [--days 30]
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

DODGERS_TEAM_ID = 119
BASE_URL = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path(__file__).parent / "data" / "schedule"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_schedule(days_ahead: int = 30) -> list[dict]:
    today = date.today()
    end = today + timedelta(days=days_ahead)
    data = _get("schedule", {
        "teamId": DODGERS_TEAM_ID,
        "startDate": today.isoformat(),
        "endDate": end.isoformat(),
        "sportId": 1,
        "hydrate": "probablePitcher(note),team",
    })

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            game_date = date_entry["date"]
            game_pk = g.get("gamePk")
            status = g.get("status", {}).get("abstractGameState", "")

            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team = home.get("team", {}).get("name", "")
            away_team = away.get("team", {}).get("name", "")
            home_id = home.get("team", {}).get("id")
            away_id = away.get("team", {}).get("id")

            # 相手チーム
            if home_id == DODGERS_TEAM_ID:
                opp_team = away_team
                opp_team_id = away_id
                home_away = "home"
            else:
                opp_team = home_team
                opp_team_id = home_id
                home_away = "away"

            # 先発投手（相手）
            if home_id == DODGERS_TEAM_ID:
                opp_pitcher_node = away.get("probablePitcher")
            else:
                opp_pitcher_node = home.get("probablePitcher")

            # 自チーム先発（大谷が先発かどうか確認用）
            if home_id == DODGERS_TEAM_ID:
                own_pitcher_node = home.get("probablePitcher")
            else:
                own_pitcher_node = away.get("probablePitcher")

            opp_pitcher_id = opp_pitcher_node.get("id") if opp_pitcher_node else None
            opp_pitcher_name = opp_pitcher_node.get("fullName") if opp_pitcher_node else "TBD"
            own_pitcher_id = own_pitcher_node.get("id") if own_pitcher_node else None
            own_pitcher_name = own_pitcher_node.get("fullName") if own_pitcher_node else "TBD"

            # 大谷(660271)が先発投手として登録されているか
            ohtani_starting = own_pitcher_id == 660271

            games.append({
                "game_date": game_date,
                "game_pk": game_pk,
                "status": status,
                "home_away": home_away,
                "home_team": home_team,
                "away_team": away_team,
                "opponent_team": opp_team,
                "opponent_team_id": opp_team_id,
                "opp_starter_id": opp_pitcher_id,
                "opp_starter_name": opp_pitcher_name,
                "dodgers_starter_id": own_pitcher_id,
                "dodgers_starter_name": own_pitcher_name,
                "ohtani_starting_pitcher": ohtani_starting,
            })

    return games


def fetch_roster(team_id: int) -> list[dict]:
    """チームの現在のアクティブロスター（40人枠含む）"""
    data = _get(f"teams/{team_id}/roster", {"rosterType": "active"})
    players = []
    for p in data.get("roster", []):
        person = p.get("person", {})
        players.append({
            "player_id": person.get("id"),
            "name": person.get("fullName"),
            "position": p.get("position", {}).get("abbreviation"),
            "jersey_number": p.get("jerseyNumber"),
        })
    return players


def save_schedule(games: list[dict]) -> Path:
    df = pd.DataFrame(games)
    path = DATA_DIR / "dodgers_schedule.parquet"
    df.to_parquet(path, index=False)

    json_path = DATA_DIR / "dodgers_schedule.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)

    return path


def print_upcoming(games: list[dict], n: int = 10) -> None:
    print(f"\n=== 今後のドジャース試合 (最大{n}件) ===")
    header = f"{'日付':12} {'H/A':5} {'相手':30} {'相手先発':30} {'大谷先発':8}"
    print(header)
    print("-" * len(header))
    for g in games[:n]:
        ohtani_flag = "★先発" if g["ohtani_starting_pitcher"] else ""
        print(
            f"{g['game_date']:12} {g['home_away']:5} "
            f"{g['opponent_team']:30} {g['opp_starter_name']:30} {ohtani_flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ドジャース試合スケジュール取得")
    parser.add_argument("--days", type=int, default=30, help="今日から何日先まで取得するか")
    parser.add_argument("--roster", action="store_true", help="ドジャースロスターも取得")
    args = parser.parse_args()

    print(f"スケジュール取得中 (今後{args.days}日)...")
    games = fetch_schedule(days_ahead=args.days)
    print(f"{len(games)} 試合取得")

    path = save_schedule(games)
    print(f"保存: {path}")

    print_upcoming(games)

    if args.roster:
        print("\nドジャースロスター取得中...")
        roster = fetch_roster(DODGERS_TEAM_ID)
        df_r = pd.DataFrame(roster)
        rpath = DATA_DIR / "dodgers_roster.parquet"
        df_r.to_parquet(rpath, index=False)
        print(f"{len(roster)} 選手保存: {rpath}")


if __name__ == "__main__":
    main()
