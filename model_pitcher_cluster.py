"""
投手クラスタリングモジュール
─ 相手投手の球種特性ベクトルを生成し、大谷が過去に対戦した「類似投手」を探索
─ 大谷 vs 類似投手群の実績からバッター予測成績を算出する

主な公開関数:
  build_ohtani_batter_db(df_batter)  -> DataFrame   大谷打者 DB 構築
  build_pitcher_feature(df)          -> dict         1 投手の特性ベクトル
  find_similar_pitchers(target_feat, pitcher_feat_db, k) -> list[dict]
  predict_vs_pitcher(similar, ohtani_db) -> dict     予測成績
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 対象球種（大谷打席データに現れる主要球種）
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "KC", "SV", "CS"]

# 特性ベクトルの構成要素
# [球種割合 × n] + [球種別 avg_velo, pfx_x, pfx_z × n] + [arm_angle_proxy × 2]
FEATURE_PITCH_KEYS = PITCH_TYPES  # 割合特徴量


def _pitch_mix(df: pd.DataFrame) -> dict[str, float]:
    """球種割合（合計 1.0）"""
    total = len(df)
    if total == 0:
        return {pt: 0.0 for pt in PITCH_TYPES}
    counts = df["pitch_type"].value_counts()
    return {pt: counts.get(pt, 0) / total for pt in PITCH_TYPES}


def _pitch_metrics(df: pd.DataFrame) -> dict[str, float]:
    """球種別 平均球速・変化量（不足時は全体平均で補完）"""
    metrics: dict[str, float] = {}
    global_velo = df["release_speed"].mean()
    global_px = df["pfx_x"].mean()
    global_pz = df["pfx_z"].mean()

    for pt in PITCH_TYPES:
        sub = df[df["pitch_type"] == pt]
        if len(sub) >= 5:
            metrics[f"{pt}_velo"] = sub["release_speed"].mean()
            metrics[f"{pt}_pfx_x"] = sub["pfx_x"].mean()
            metrics[f"{pt}_pfx_z"] = sub["pfx_z"].mean()
        else:
            metrics[f"{pt}_velo"] = global_velo
            metrics[f"{pt}_pfx_x"] = global_px
            metrics[f"{pt}_pfx_z"] = global_pz

    return metrics


def _arm_angle(df: pd.DataFrame) -> dict[str, float]:
    """リリースポイントの中央値（アームアングルの代理変数）"""
    return {
        "release_pos_x_median": df["release_pos_x"].median(),
        "release_pos_z_median": df["release_pos_z"].median(),
    }


def build_pitcher_feature(df: pd.DataFrame) -> dict[str, float]:
    """
    1 投手の Statcast データから特性ベクトルを辞書で返す。
    NaN は 0 で補完済み。
    """
    feat: dict[str, float] = {}
    feat.update(_pitch_mix(df))
    feat.update(_pitch_metrics(df))
    feat.update(_arm_angle(df))
    return {k: (0.0 if (v != v) else float(v)) for k, v in feat.items()}


def _feature_keys(sample: dict) -> list[str]:
    return sorted(sample.keys())


def build_pitcher_feature_db(
    ohtani_batter_df: pd.DataFrame,
    opponent_dfs: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """
    大谷打者データに登場する投手 + opponent_dfs の特性 DB を構築する。

    Returns:
        feat_df : index=player_id, columns=特性キー
        keys    : 使用した特性キーのリスト
    """
    # 大谷打席に登場した投手 ID ごとに特性ベクトルを作成
    records = {}
    pitcher_ids_in_batter = ohtani_batter_df["pitcher"].dropna().astype(int).unique()

    for pid in pitcher_ids_in_batter:
        sub = ohtani_batter_df[ohtani_batter_df["pitcher"] == pid]
        if len(sub) >= 3:  # 最低 3 球
            records[pid] = build_pitcher_feature(sub)

    # opponent_dfs があれば追加（新たな対戦相手）
    for pid, df in opponent_dfs.items():
        if len(df) >= 3:
            records[pid] = build_pitcher_feature(df)

    if not records:
        raise ValueError("投手特性 DB が空です")

    feat_df = pd.DataFrame.from_dict(records, orient="index").fillna(0.0)
    feat_df.index.name = "player_id"
    keys = sorted(feat_df.columns.tolist())
    feat_df = feat_df[keys]
    return feat_df, keys


def find_similar_pitchers(
    target_feat: dict[str, float],
    feat_df: pd.DataFrame,
    keys: list[str],
    k: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[dict]:
    """
    target_feat に最も近い k 人の投手を返す。

    Returns: [{"player_id": int, "distance": float}, ...]
    """
    work_df = feat_df.copy()
    if exclude_ids:
        work_df = work_df[~work_df.index.isin(exclude_ids)]

    if len(work_df) == 0:
        return []

    target_vec = np.array([target_feat.get(k_, 0.0) for k_ in keys]).reshape(1, -1)
    X = work_df[keys].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform(target_vec)

    n_neighbors = min(k, len(work_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(target_scaled)

    results = []
    player_ids = work_df.index.tolist()
    for dist, idx in zip(distances[0], indices[0]):
        results.append({"player_id": player_ids[idx], "distance": float(dist)})
    return results


# ── 大谷（打者）実績計算 ─────────────────────────────────────


def _at_bat_events(df: pd.DataFrame) -> pd.DataFrame:
    """打席結果（1球単位データから打席終了イベントを抽出）"""
    terminal_events = {
        "strikeout", "walk", "hit_by_pitch",
        "single", "double", "triple", "home_run",
        "field_out", "force_out", "grounded_into_double_play",
        "double_play", "triple_play",
        "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_fly_error", "sac_bunt",
    }
    return df[df["events"].isin(terminal_events)].copy()


def compute_ohtani_stats_vs_pitcher(
    ohtani_batter_df: pd.DataFrame,
    pitcher_id: int,
) -> dict:
    """大谷 vs 特定投手の実績を返す"""
    sub = ohtani_batter_df[ohtani_batter_df["pitcher"] == pitcher_id]
    return _compute_stats(sub)


def _compute_stats(df: pd.DataFrame) -> dict:
    """球データから打者成績を計算"""
    n_pitches = len(df)
    if n_pitches == 0:
        return {"n_pitches": 0, "n_pa": 0}

    events_df = _at_bat_events(df)
    n_pa = len(events_df)

    swings = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    ])
    whiffs = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ])

    k_count = (events_df["events"] == "strikeout").sum() if n_pa > 0 else 0
    bb_count = (events_df["events"].isin(["walk", "hit_by_pitch"])).sum() if n_pa > 0 else 0
    woba_val = events_df["woba_value"].sum() if n_pa > 0 else 0.0
    woba_denom = events_df["woba_denom"].sum() if n_pa > 0 else 0.0
    xba = df["estimated_ba_using_speedangle"].mean()
    xwoba = df["estimated_woba_using_speedangle"].mean()

    return {
        "n_pitches": n_pitches,
        "n_pa": int(n_pa),
        "k_rate": round(k_count / n_pa, 3) if n_pa > 0 else None,
        "bb_rate": round(bb_count / n_pa, 3) if n_pa > 0 else None,
        "woba": round(woba_val / woba_denom, 3) if woba_denom > 0 else None,
        "xwoba": round(float(xwoba), 3) if not np.isnan(xwoba) else None,
        "xba": round(float(xba), 3) if not np.isnan(xba) else None,
        "swing_rate": round(swings.mean(), 3),
        "whiff_rate": round(whiffs.sum() / swings.sum(), 3) if swings.sum() > 0 else None,
    }


def predict_vs_pitcher(
    similar_pitchers: list[dict],
    ohtani_batter_df: pd.DataFrame,
    min_pitches: int = 5,
) -> dict:
    """
    類似投手群に対する大谷の加重平均成績を予測する。
    距離が小さいほど重みが大きい（inverse distance weighting）。
    """
    records = []
    for sim in similar_pitchers:
        pid = sim["player_id"]
        dist = sim["distance"]
        stats = compute_ohtani_stats_vs_pitcher(ohtani_batter_df, pid)
        if stats["n_pitches"] >= min_pitches:
            records.append({"player_id": pid, "dist": dist, **stats})

    if not records:
        return {"method": "insufficient_data", "n_similar": 0}

    # Inverse distance weight (distance=0 の場合は大きい重みを割り当て)
    weights = np.array([1.0 / (r["dist"] + 0.01) for r in records])
    weights /= weights.sum()

    def wavg(key: str) -> float | None:
        vals = [r.get(key) for r in records]
        valid = [(w, v) for w, v in zip(weights, vals) if v is not None]
        if not valid:
            return None
        ws, vs = zip(*valid)
        ws = np.array(ws); ws /= ws.sum()
        return round(float(np.dot(ws, vs)), 3)

    return {
        "method": "similarity_weighted",
        "n_similar": len(records),
        "k_rate": wavg("k_rate"),
        "bb_rate": wavg("bb_rate"),
        "woba": wavg("woba"),
        "xwoba": wavg("xwoba"),
        "xba": wavg("xba"),
        "swing_rate": wavg("swing_rate"),
        "whiff_rate": wavg("whiff_rate"),
        "similar_pitcher_ids": [r["player_id"] for r in records],
    }
