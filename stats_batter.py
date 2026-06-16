"""
大谷翔平（打者）今季 Stats 集計モジュール
既存の model_batter_profile.py のロジック（_swings/_whiffs/zone定義）を再利用し、
シーズン単位の追加集計（サマリー・ゾーングリッド・月別トレンド・対左右投手・カウント別）を提供する。

主な公開関数:
  compute_summary(df)        -> dict   シーズン総合サマリー
  compute_zone_grid(df, metric) -> dict  3x3 グリッド（行列データ + 各セルのn）
  compute_pitch_split(df)    -> DataFrame  球種別スプリット
  compute_ev_stats(df)       -> DataFrame  打球（EV/LA）一覧
  compute_monthly_trend(df)  -> DataFrame  月別 wOBA/xwOBA トレンド
  compute_lr_split(df)       -> DataFrame  対左右投手スプリット
  compute_count_split(df)    -> DataFrame  カウント別スプリット
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_batter_profile import (
    _swings, _whiffs, _stats_for_subset,
    STRIKE_ZONES, build_batter_profile,
)

MIN_PA = 10  # 集計表示に必要な最低打席数（少数サンプルはデータ不足として隠す）

COUNT_GROUP_MAP = {
    "1-0": "先行", "2-0": "先行", "3-0": "先行", "2-1": "先行", "3-1": "先行",
    "0-0": "均衡", "1-1": "均衡", "2-2": "均衡",
    "0-1": "追い込み", "0-2": "追い込み", "1-2": "追い込み",
    "3-2": "フルカウント",
}
COUNT_GROUP_ORDER = ["先行", "均衡", "追い込み", "フルカウント"]


def _is_barrel(ev: pd.Series, la: pd.Series) -> pd.Series:
    """
    Statcast の生データに barrel フラグが含まれないため、公式定義を近似計算する。
    EV>=98 で LA 許容範囲が EV 上昇に応じて 26-30°(EV98) → 8-50°(EV116+) に広がる。
    """
    lower = 26 - (ev - 98) * (26 - 8) / (116 - 98)
    upper = 30 + (ev - 98) * (50 - 30) / (116 - 98)
    lower = lower.clip(upper=26)
    upper = upper.clip(lower=30)
    is_max = ev >= 116
    cond = (ev >= 98) & (
        is_max & (la >= 8) & (la <= 50)
        | (~is_max) & (la >= lower) & (la <= upper)
    )
    return cond.fillna(False)


def compute_summary(df: pd.DataFrame) -> dict:
    """シーズン総合サマリー（wOBA/xwOBA/K%/BB%/バレル率/ハードヒット率/EV）"""
    if df.empty:
        return {}

    pa_rows = df[df["events"].notna()]
    n_pa = len(pa_rows)
    k_rate  = (pa_rows["events"] == "strikeout").sum() / n_pa if n_pa else None
    bb_rate = pa_rows["events"].isin(["walk", "hit_by_pitch"]).sum() / n_pa if n_pa else None

    terminal = df[df["events"].notna() & (df["woba_denom"].fillna(0) > 0)]
    woba_d = terminal["woba_denom"].sum()
    woba   = terminal["woba_value"].sum() / woba_d if woba_d > 0 else None
    xwoba  = terminal["estimated_woba_using_speedangle"].mean()
    xwoba  = float(xwoba) if pd.notna(xwoba) else None

    bip = df[df["type"] == "X"]
    n_bip = len(bip)
    barrel_rate   = _is_barrel(bip["launch_speed"], bip["launch_angle"]).sum() / n_bip if n_bip else None
    hard_hit_rate = (bip["launch_speed"] >= 95).sum() / n_bip if n_bip else None
    avg_ev = bip["launch_speed"].mean()
    max_ev = bip["launch_speed"].max()

    return {
        "n_pa": int(n_pa),
        "n_bip": int(n_bip),
        "k_rate": round(k_rate, 3) if k_rate is not None else None,
        "bb_rate": round(bb_rate, 3) if bb_rate is not None else None,
        "woba": round(woba, 3) if woba is not None else None,
        "xwoba": round(xwoba, 3) if xwoba is not None else None,
        "barrel_rate": round(barrel_rate, 3) if barrel_rate is not None else None,
        "hard_hit_rate": round(hard_hit_rate, 3) if hard_hit_rate is not None else None,
        "avg_ev": round(float(avg_ev), 1) if pd.notna(avg_ev) else None,
        "max_ev": round(float(max_ev), 1) if pd.notna(max_ev) else None,
    }


def compute_zone_grid(df: pd.DataFrame, metric: str = "xwoba") -> dict | None:
    """
    ストライクゾーン9分割（zone列 1-9）を、各ゾーンの平均 plate_x/plate_z から
    3x3 グリッド位置を動的に決定して配置する。
    metric: 'xwoba' or 'whiff_rate'
    戻り値: {"grid": 3x3 ndarray(値, NaN埋め), "n_grid": 3x3 ndarray(件数)}
    """
    if df.empty or "zone" not in df.columns:
        return None

    profile = build_batter_profile(df)
    by_zone = profile.get("by_zone", {})

    rows = []
    for z in STRIKE_ZONES:
        sub = df[df["zone"] == z]
        if sub.empty:
            continue
        stats = by_zone.get(z, {})
        rows.append({
            "zone": z,
            "mean_x": sub["plate_x"].mean(),
            "mean_z": sub["plate_z"].mean(),
            "value": stats.get(metric),
            "n": stats.get("n", len(sub)),
        })

    if len(rows) < 9:
        return None

    zdf = pd.DataFrame(rows)
    # 横方向(x)・縦方向(z)それぞれを3グループに動的分割（実データの位置関係で決定）
    zdf["col"] = pd.qcut(zdf["mean_x"].rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    zdf["row"] = pd.qcut(zdf["mean_z"].rank(method="first"), 3, labels=[2, 1, 0]).astype(int)  # z大=上=row0

    grid = np.full((3, 3), np.nan)
    n_grid = np.zeros((3, 3), dtype=int)
    for _, r in zdf.iterrows():
        if r["value"] is not None:
            grid[int(r["row"]), int(r["col"])] = r["value"]
        n_grid[int(r["row"]), int(r["col"])] = r["n"]

    return {"grid": grid, "n_grid": n_grid}


def compute_pitch_split(df: pd.DataFrame) -> pd.DataFrame:
    """球種別スプリット（投球数/xwOBA/空振り率/平均EV）"""
    if df.empty:
        return pd.DataFrame()

    profile = build_batter_profile(df)
    by_pitch = profile.get("by_pitch", {})

    rows = []
    for pt, stats in by_pitch.items():
        sub = df[df["pitch_type"] == pt]
        bip = sub[sub["type"] == "X"]
        avg_ev = bip["launch_speed"].mean()
        rows.append({
            "pitch_type": pt,
            "n": stats.get("n", 0),
            "n_pa": stats.get("n_pa", 0),
            "xwoba": stats.get("xwoba"),
            "whiff_rate": stats.get("whiff_rate"),
            "avg_ev": round(float(avg_ev), 1) if pd.notna(avg_ev) else None,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False) if not out.empty else out


def compute_ev_stats(df: pd.DataFrame) -> pd.DataFrame:
    """打球（インプレー）の EV・打球角度・種別データ"""
    if df.empty:
        return pd.DataFrame()
    bip = df[df["type"] == "X"].dropna(subset=["launch_speed", "launch_angle"])
    cols = [c for c in ["launch_speed", "launch_angle", "bb_type", "events", "game_date"] if c in bip.columns]
    return bip[cols].copy()


def compute_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """月別 wOBA/xwOBA トレンド（サンプル不足の月は除外）"""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["month"] = work["game_date"].dt.to_period("M")

    rows = []
    for month, sub in work.groupby("month"):
        terminal = sub[sub["events"].notna() & (sub["woba_denom"].fillna(0) > 0)]
        n_pa = len(terminal)
        if n_pa < MIN_PA:
            continue
        woba_d = terminal["woba_denom"].sum()
        woba = terminal["woba_value"].sum() / woba_d if woba_d > 0 else None
        xwoba = terminal["estimated_woba_using_speedangle"].mean()
        rows.append({
            "month": str(month),
            "n_pa": n_pa,
            "woba": round(woba, 3) if woba is not None else None,
            "xwoba": round(float(xwoba), 3) if pd.notna(xwoba) else None,
        })
    return pd.DataFrame(rows)


def compute_lr_split(df: pd.DataFrame) -> pd.DataFrame:
    """対左右投手スプリット"""
    if df.empty or "p_throws" not in df.columns:
        return pd.DataFrame()

    rows = []
    for hand, label in [("L", "対左投手"), ("R", "対右投手")]:
        sub = df[df["p_throws"] == hand]
        if sub.empty:
            continue
        stats = _stats_for_subset(sub)
        if stats.get("n_pa", 0) < MIN_PA:
            continue
        rows.append({"hand": label, **stats})
    return pd.DataFrame(rows)


def compute_count_split(df: pd.DataFrame) -> pd.DataFrame:
    """カウント別スプリット（先行/均衡/追い込み/フルカウント）"""
    if df.empty or "balls" not in df.columns or "strikes" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["count_str"] = (
        work["balls"].fillna(-1).astype(int).astype(str) + "-" +
        work["strikes"].fillna(-1).astype(int).astype(str)
    )
    work["count_group"] = work["count_str"].map(COUNT_GROUP_MAP)

    rows = []
    for label in COUNT_GROUP_ORDER:
        sub = work[work["count_group"] == label]
        if sub.empty:
            continue
        stats = _stats_for_subset(sub)
        if stats.get("n_pa", 0) < MIN_PA:
            continue
        rows.append({"count_group": label, **stats})
    return pd.DataFrame(rows)
