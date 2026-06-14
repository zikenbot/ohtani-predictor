"""
大谷翔平 対戦予想エンジン
─ スケジュールから次試合の相手を特定し、投打両面の予測を出力する

実行: py -3.13 predict.py [--game-date 2026-06-14] [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd

from model_pitcher_cluster import (
    build_pitcher_feature,
    build_pitcher_feature_db,
    find_similar_pitchers,
    compute_ohtani_stats_vs_pitcher,
    predict_vs_pitcher,
)
from model_batter_profile import (
    build_batter_profile,
    build_ohtani_pitcher_profile,
    predict_ohtani_vs_batter,
)

DATA_DIR = Path(__file__).parent / "data"
OHTANI_ID = 660271
MIN_DIRECT_PITCHES = 10  # 戦歴充分判定の閾値


def _load_ohtani(role: str) -> pd.DataFrame:
    """大谷の Statcast データを全年分結合して返す"""
    files = sorted((DATA_DIR / "statcast").glob(f"ohtani_{role}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"ohtani_{role}_*.parquet が見つかりません。collect_statcast.py を実行してください。")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_opponent(role: str, player_id: int) -> pd.DataFrame | None:
    """対戦相手の Statcast データを読み込む（キャッシュなければ None）"""
    files = sorted((DATA_DIR / "opponents").glob(f"{role}_{player_id}_*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_schedule() -> pd.DataFrame:
    path = DATA_DIR / "schedule" / "dodgers_schedule.parquet"
    if not path.exists():
        raise FileNotFoundError("スケジュールファイルがありません。collect_schedule.py を実行してください。")
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _find_game(schedule: pd.DataFrame, target_date: str | None) -> pd.Series:
    """予測対象の試合を返す（日付指定なければ直近試合）"""
    today = pd.Timestamp(date.today())
    if target_date:
        dt = pd.Timestamp(target_date)
        row = schedule[schedule["game_date"] == dt]
        if row.empty:
            raise ValueError(f"{target_date} の試合が見つかりません")
        return row.iloc[0]
    # 今日以降の最初の試合
    future = schedule[schedule["game_date"] >= today].sort_values("game_date")
    if future.empty:
        raise ValueError("今後の試合が見つかりません")
    return future.iloc[0]


# ── 打者予測（大谷 vs 相手投手） ────────────────────────────


def predict_ohtani_batter(
    ohtani_batter_df: pd.DataFrame,
    opp_pitcher_id: int | None,
    opp_pitcher_name: str,
    opp_pitcher_df: pd.DataFrame | None,
) -> dict:
    """大谷（打者）の対戦予測を返す"""
    result: dict = {
        "role": "batter",
        "opponent_pitcher": opp_pitcher_name,
        "opponent_pitcher_id": opp_pitcher_id,
    }

    # 直接対面データ確認
    if opp_pitcher_id:
        direct = ohtani_batter_df[ohtani_batter_df["pitcher"] == opp_pitcher_id]
        n_direct = len(direct)
        result["direct_pitches"] = int(n_direct)
    else:
        direct = pd.DataFrame()
        n_direct = 0

    if n_direct >= MIN_DIRECT_PITCHES:
        # 戦歴充分 → 実績直接使用
        from model_pitcher_cluster import _compute_stats
        stats = _compute_stats(direct)
        result["method"] = "direct_history"
        result["prediction"] = stats
        result["note"] = f"直接対面 {n_direct} 球の実績ベース"
    else:
        # 戦歴不足 → 類似投手で補完
        if opp_pitcher_df is None or len(opp_pitcher_df) < 10:
            result["method"] = "insufficient_data"
            result["prediction"] = None
            result["note"] = "相手投手データ不足（collect_opponents.py を実行してください）"
            return result

        opp_feat = build_pitcher_feature(opp_pitcher_df)

        # 大谷打者データに登場した投手の特性 DB を構築
        feat_db, keys = build_pitcher_feature_db(ohtani_batter_df, {opp_pitcher_id: opp_pitcher_df} if opp_pitcher_id else {})

        exclude = {opp_pitcher_id} if opp_pitcher_id else set()
        similar = find_similar_pitchers(opp_feat, feat_db, keys, k=15, exclude_ids=exclude)
        prediction = predict_vs_pitcher(similar, ohtani_batter_df)

        result["method"] = prediction.get("method", "similarity_weighted")
        result["prediction"] = prediction
        result["note"] = f"直接対面 {n_direct} 球 → 類似投手 {prediction.get('n_similar', 0)} 人の加重平均"

    return result


# ── 投手予測（大谷 vs 相手打者群） ──────────────────────────


def predict_ohtani_pitcher(
    ohtani_pitcher_df: pd.DataFrame,
    opp_team_id: int,
    opp_team_name: str,
) -> dict:
    """大谷（投手）vs 相手チーム打者陣の対戦予測を返す"""
    result: dict = {
        "role": "pitcher",
        "opponent_team": opp_team_name,
        "opponent_team_id": opp_team_id,
    }

    ohtani_prof = build_ohtani_pitcher_profile(ohtani_pitcher_df)
    result["ohtani_pitch_mix"] = ohtani_prof.get("pitch_mix", {})

    # 相手チームの打者データを読み込み
    batter_files = list((DATA_DIR / "opponents").glob("batter_*_*.parquet"))
    if not batter_files:
        result["method"] = "insufficient_data"
        result["note"] = "相手打者データなし（collect_opponents.py を実行してください）"
        return result

    batter_predictions: list[dict] = []
    loaded = 0
    for f in batter_files:
        batter_df = pd.read_parquet(f)
        if batter_df.empty:
            continue
        profile = build_batter_profile(batter_df)
        pred = predict_ohtani_vs_batter(profile, ohtani_prof)
        # player_id をファイル名から取得
        parts = f.stem.split("_")
        pid = int(parts[1]) if len(parts) >= 2 else -1
        batter_predictions.append({"player_id": pid, **pred})
        loaded += 1

    if not batter_predictions:
        result["method"] = "insufficient_data"
        result["note"] = "有効な打者データなし"
        return result

    # チーム全体の平均
    xwobas = [p["predicted_xwoba_allowed"] for p in batter_predictions if p.get("predicted_xwoba_allowed") is not None]
    whiffs = [p["predicted_whiff_rate"] for p in batter_predictions if p.get("predicted_whiff_rate") is not None]

    result["method"] = "batter_profile_weighted"
    result["n_batters_analyzed"] = loaded
    result["team_avg_xwoba_allowed"] = round(np.mean(xwobas), 3) if xwobas else None
    result["team_avg_whiff_rate"] = round(np.mean(whiffs), 3) if whiffs else None
    result["note"] = f"相手チーム {loaded} 打者の弱点マップ × 大谷球種傾向"
    result["per_batter"] = sorted(
        batter_predictions,
        key=lambda x: x.get("predicted_xwoba_allowed") or 1.0
    )[:10]  # 被 xwOBA が低い（= 抑えやすい）上位10人

    return result


# ── メイン ──────────────────────────────────────────────────


def run_prediction(game_date: str | None = None, as_json: bool = False) -> dict:
    print("データ読み込み中...")
    ohtani_batter_df = _load_ohtani("batter")
    ohtani_pitcher_df = _load_ohtani("pitcher")
    schedule = _load_schedule()

    game = _find_game(schedule, game_date)
    print(f"\n対象試合: {game['game_date'].strftime('%Y-%m-%d')}  "
          f"{game['home_away'].upper()}  vs {game['opponent_team']}")
    print(f"  相手先発: {game['opp_starter_name']} (ID: {game['opp_starter_id']})")
    print(f"  大谷先発投手: {game['ohtani_starting_pitcher']}")

    opp_pitcher_id = int(game["opp_starter_id"]) if pd.notna(game["opp_starter_id"]) else None
    opp_pitcher_name = game["opp_starter_name"]
    opp_team_id = int(game["opponent_team_id"]) if pd.notna(game["opponent_team_id"]) else None
    opp_team_name = game["opponent_team"]

    # 相手先発投手データ読み込み
    opp_pitcher_df = _load_opponent("pitcher", opp_pitcher_id) if opp_pitcher_id else None

    output: dict = {
        "game_date": game["game_date"].strftime("%Y-%m-%d"),
        "home_away": game["home_away"],
        "opponent_team": opp_team_name,
        "opp_starter": opp_pitcher_name,
        "ohtani_starting_pitcher": bool(game["ohtani_starting_pitcher"]),
    }

    # ① 大谷打者予測（常時）
    print("\n[大谷 打者] 予測中...")
    output["ohtani_batter"] = predict_ohtani_batter(
        ohtani_batter_df, opp_pitcher_id, opp_pitcher_name, opp_pitcher_df
    )

    # ② 大谷投手予測（大谷が先発の場合のみ）
    if game["ohtani_starting_pitcher"]:
        print("[大谷 投手] 予測中...")
        output["ohtani_pitcher"] = predict_ohtani_pitcher(
            ohtani_pitcher_df, opp_team_id, opp_team_name
        )
    else:
        output["ohtani_pitcher"] = {"role": "pitcher", "note": "この試合で大谷は先発投手ではありません"}

    return output


def _print_result(output: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  予測日: {output['game_date']}  {output['home_away'].upper()}  vs {output['opponent_team']}")
    print(f"  相手先発: {output['opp_starter']}")
    print("=" * 60)

    # 打者
    batter = output.get("ohtani_batter", {})
    pred = batter.get("prediction", {}) or {}
    print("\n▼ 大谷（打者）予測")
    print(f"  手法: {batter.get('method', '-')}  ({batter.get('note', '')})")
    for k in ["k_rate", "bb_rate", "woba", "xwoba", "xba", "swing_rate", "whiff_rate"]:
        v = pred.get(k)
        print(f"  {k:15s}: {v if v is not None else '---'}")

    # 投手
    pitcher = output.get("ohtani_pitcher", {})
    if pitcher.get("method") not in (None, "insufficient_data"):
        print("\n▼ 大谷（投手）予測")
        print(f"  手法: {pitcher.get('method', '-')}  ({pitcher.get('note', '')})")
        print(f"  球種: {pitcher.get('ohtani_pitch_mix', {})}")
        print(f"  チーム平均 被 xwOBA : {pitcher.get('team_avg_xwoba_allowed', '---')}")
        print(f"  チーム平均 Whiff 率 : {pitcher.get('team_avg_whiff_rate', '---')}")
    else:
        print(f"\n▼ 大谷（投手）: {pitcher.get('note', '非先発')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="大谷翔平 対戦予想")
    parser.add_argument("--game-date", type=str, help="予測対象日 (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    args = parser.parse_args()

    output = run_prediction(game_date=args.game_date, as_json=args.json)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        _print_result(output)


if __name__ == "__main__":
    main()
