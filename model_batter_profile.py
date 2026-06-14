"""
打者プロファイルモジュール
─ 相手打者の球種×ゾーン別弱点マップを構築
─ 大谷（投手）の球種・コース傾向と照合して被打率・被 wOBA 予測を出力

主な公開関数:
  build_batter_profile(df_batter)         -> dict   打者弱点マップ
  build_ohtani_pitcher_profile(df)        -> dict   大谷投手の球種・コース傾向
  predict_ohtani_vs_batter(batter_profile, ohtani_pitcher_profile) -> dict
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "KC", "SV"]

# Statcast ゾーン定義
# 1-9: ストライクゾーン（3×3グリッド）
# 11-14: ボールゾーン（コーナー外側）
STRIKE_ZONES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BALL_ZONES = [11, 12, 13, 14]
ALL_ZONES = STRIKE_ZONES + BALL_ZONES

# ゾーンを内角・真ん中・外角に分類（右打者基準、left=内角側はゾーン 1,4,7）
INSIDE_ZONES  = [1, 4, 7]   # 打者内角側
MIDDLE_ZONES  = [2, 5, 8]
OUTSIDE_ZONES = [3, 6, 9]   # 打者外角側
HIGH_ZONES    = [1, 2, 3]
MID_ZONES     = [4, 5, 6]
LOW_ZONES     = [7, 8, 9]


def _swings(df: pd.DataFrame) -> pd.Series:
    return df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    ])


def _whiffs(df: pd.DataFrame) -> pd.Series:
    return df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ])


def _stats_for_subset(df: pd.DataFrame) -> dict:
    """部分データから swing/whiff/woba/xba を計算"""
    n = len(df)
    if n == 0:
        return {"n": 0}

    sw = _swings(df).sum()
    wh = _whiffs(df).sum()
    terminal = df[df["events"].notna() & (df["woba_denom"].fillna(0) > 0)]
    woba_v = terminal["woba_value"].sum()
    woba_d = terminal["woba_denom"].sum()
    xba = df["estimated_ba_using_speedangle"].mean()
    xwoba = df["estimated_woba_using_speedangle"].mean()

    return {
        "n": int(n),
        "n_pa": int(len(terminal)),
        "swing_rate": round(sw / n, 3) if n else None,
        "whiff_rate": round(wh / sw, 3) if sw > 0 else None,
        "woba": round(woba_v / woba_d, 3) if woba_d > 0 else None,
        "xba": round(float(xba), 3) if not np.isnan(xba) else None,
        "xwoba": round(float(xwoba), 3) if not np.isnan(xwoba) else None,
    }


def build_batter_profile(df: pd.DataFrame) -> dict:
    """
    打者の Statcast データから弱点マップを構築する。

    Returns:
      {
        "overall": {...},
        "by_pitch": { "FF": {...}, "SL": {...}, ... },
        "by_zone": { 1: {...}, 2: {...}, ... },
        "by_pitch_zone": { ("FF", 5): {...}, ... },
        "hand": "R"/"L",
        "n_total": int,
      }
    """
    n_total = len(df)
    if n_total == 0:
        return {"n_total": 0}

    hand = df["stand"].mode().iloc[0] if "stand" in df.columns else "R"

    # 全体
    overall = _stats_for_subset(df)

    # 球種別
    by_pitch: dict[str, dict] = {}
    for pt in PITCH_TYPES:
        sub = df[df["pitch_type"] == pt]
        if len(sub) >= 5:
            by_pitch[pt] = _stats_for_subset(sub)

    # ゾーン別
    by_zone: dict[int, dict] = {}
    for z in ALL_ZONES:
        sub = df[df["zone"] == z]
        if len(sub) >= 3:
            by_zone[z] = _stats_for_subset(sub)

    # 球種×ゾーン別（最低 3 球）
    by_pitch_zone: dict[tuple, dict] = {}
    for pt in PITCH_TYPES:
        for z in STRIKE_ZONES:
            sub = df[(df["pitch_type"] == pt) & (df["zone"] == z)]
            if len(sub) >= 3:
                by_pitch_zone[(pt, z)] = _stats_for_subset(sub)

    # コース帯別（内/中/外 × 高/中/低）
    zone_bands: dict[str, dict] = {}
    for label, zones in [
        ("inside", INSIDE_ZONES), ("middle_h", MIDDLE_ZONES), ("outside", OUTSIDE_ZONES),
        ("high", HIGH_ZONES), ("mid_v", MID_ZONES), ("low", LOW_ZONES),
    ]:
        sub = df[df["zone"].isin(zones)]
        if len(sub) >= 5:
            zone_bands[label] = _stats_for_subset(sub)

    return {
        "n_total": n_total,
        "hand": hand,
        "overall": overall,
        "by_pitch": by_pitch,
        "by_zone": by_zone,
        "by_pitch_zone": {f"{pt}_{z}": v for (pt, z), v in by_pitch_zone.items()},
        "zone_bands": zone_bands,
    }


def build_ohtani_pitcher_profile(df: pd.DataFrame) -> dict:
    """
    大谷（投手）の球種別 割合・コース傾向・被打率を集計する。
    """
    n_total = len(df)
    if n_total == 0:
        return {"n_total": 0}

    pitch_mix: dict[str, float] = {}
    pitch_location: dict[str, dict] = {}
    pitch_effectiveness: dict[str, dict] = {}

    present_types = df["pitch_type"].dropna().unique()
    for pt in present_types:
        sub = df[df["pitch_type"] == pt]
        pitch_mix[pt] = round(len(sub) / n_total, 3)
        pitch_location[pt] = {
            "avg_plate_x": round(sub["plate_x"].mean(), 3),
            "avg_plate_z": round(sub["plate_z"].mean(), 3),
            "avg_pfx_x": round(sub["pfx_x"].mean(), 3),
            "avg_pfx_z": round(sub["pfx_z"].mean(), 3),
            "avg_velo": round(sub["release_speed"].mean(), 1),
            "avg_spin": round(sub["release_spin_rate"].mean(), 0),
            "primary_zones": sub["zone"].value_counts().head(3).index.tolist(),
        }
        pitch_effectiveness[pt] = _stats_for_subset(sub)

    # 全体コース帯別
    zone_tendency: dict[str, float] = {}
    for label, zones in [
        ("inside", INSIDE_ZONES), ("outside", OUTSIDE_ZONES),
        ("high", HIGH_ZONES), ("low", LOW_ZONES),
    ]:
        zone_tendency[label] = round(
            df["zone"].isin(zones).sum() / n_total, 3
        )

    return {
        "n_total": n_total,
        "pitch_mix": pitch_mix,
        "pitch_location": pitch_location,
        "pitch_effectiveness": pitch_effectiveness,
        "zone_tendency": zone_tendency,
    }


def predict_ohtani_vs_batter(
    batter_profile: dict,
    ohtani_profile: dict,
) -> dict:
    """
    大谷（投手）vs 相手打者の対戦予測。
    大谷の球種割合 × 打者の球種別弱点（xwOBA・空振り率）を加重平均。

    Returns:
      {
        "predicted_xwoba": float,   # 大谷が許す推定 xwOBA
        "predicted_whiff_rate": float,
        "pitch_recommendation": [...],  # 攻め方ランキング
      }
    """
    pitch_mix = ohtani_profile.get("pitch_mix", {})
    batter_by_pitch = batter_profile.get("by_pitch", {})

    weighted_xwoba = 0.0
    weighted_whiff = 0.0
    total_weight = 0.0

    pitch_scores: list[dict] = []

    for pt, mix_rate in pitch_mix.items():
        if mix_rate < 0.02:  # 2% 未満は無視
            continue
        bp = batter_by_pitch.get(pt, {})
        if not bp or bp.get("n", 0) < 5:
            # データ不足は batter 全体値で代替
            xwoba = batter_profile.get("overall", {}).get("xwoba")
            whiff = batter_profile.get("overall", {}).get("whiff_rate")
        else:
            xwoba = bp.get("xwoba")
            whiff = bp.get("whiff_rate")

        if xwoba is not None:
            weighted_xwoba += mix_rate * xwoba
            total_weight += mix_rate
        if whiff is not None:
            weighted_whiff += mix_rate * (whiff or 0.0)

        # 攻め方スコア: 低 xwOBA かつ 高 whiff が良い
        eff_score = None
        if xwoba is not None and whiff is not None:
            # スコアが低いほど打者に有利（大谷に不利）→ 高い方が良い
            eff_score = round((1.0 - xwoba) * 0.5 + whiff * 0.5, 3)

        pitch_scores.append({
            "pitch_type": pt,
            "mix_rate": mix_rate,
            "batter_xwoba": xwoba,
            "batter_whiff_rate": whiff,
            "effectiveness_score": eff_score,
        })

    pred_xwoba = round(weighted_xwoba / total_weight, 3) if total_weight > 0 else None
    pred_whiff = round(weighted_whiff / total_weight, 3) if total_weight > 0 else None

    pitch_scores.sort(key=lambda x: (-(x["effectiveness_score"] or 0)))

    return {
        "predicted_xwoba_allowed": pred_xwoba,
        "predicted_whiff_rate": pred_whiff,
        "batter_hand": batter_profile.get("hand"),
        "n_batter_pitches": batter_profile.get("n_total", 0),
        "pitch_recommendation": pitch_scores[:5],
        "weak_zones": _find_weak_zones(batter_profile),
    }


def _find_weak_zones(batter_profile: dict) -> list[dict]:
    """打者の苦手ゾーン（低 xBA かつ 高 whiff）を返す"""
    by_zone = batter_profile.get("by_zone", {})
    zones = []
    for z_key, stats in by_zone.items():
        xba = stats.get("xba")
        whiff = stats.get("whiff_rate")
        if xba is not None and whiff is not None:
            weakness = round((1 - xba) * 0.4 + whiff * 0.6, 3)
            zones.append({"zone": z_key, "xba": xba, "whiff_rate": whiff, "weakness_score": weakness})
    zones.sort(key=lambda x: -x["weakness_score"])
    return zones[:5]
