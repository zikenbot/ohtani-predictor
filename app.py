"""
大谷翔平 対戦予想ダッシュボード  Streamlit アプリ
起動: streamlit run app.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from predict import run_prediction, _load_schedule
from stats_batter import (
    compute_summary, compute_zone_grid, compute_pitch_split,
    compute_ev_stats, compute_monthly_trend, compute_lr_split, compute_count_split,
    compute_pitcher_type_split,
)

DATA_DIR = Path(__file__).parent / "data"
OHTANI_ID = 660271

# ─── ページ設定 ────────────────────────────────────────────
st.set_page_config(
    page_title="大谷翔平 対戦予想",
    page_icon="⚾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="metric-container"] {
    background: #1e2a3a; border-radius: 12px; padding: 12px 16px; margin: 4px 0;
  }
  [data-testid="metric-container"] label { font-size: 0.75rem; color: #8899aa; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 700; color: #ffffff;
  }
  .pitch-badge {
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 0.75rem; font-weight: 600; margin: 2px;
  }
  .section-header {
    font-size: 1.1rem; font-weight: 700; margin: 16px 0 8px 0;
    padding-left: 8px; border-left: 4px solid #0078ff;
  }
  .method-badge {
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: 0.7rem; background: #243447; color: #8899aa; margin-bottom: 8px;
  }
  .warn-box {
    background: #2a2010; border: 1px solid #664400; border-radius: 8px;
    padding: 10px 14px; font-size: 0.85rem; color: #ccaa55;
  }
</style>
""", unsafe_allow_html=True)

# ─── 定数 ──────────────────────────────────────────────────
PITCH_COLORS = {
    "FF": "#e74c3c", "SI": "#e67e22", "FC": "#f39c12",
    "SL": "#2ecc71", "ST": "#27ae60", "CU": "#3498db",
    "CH": "#9b59b6", "FS": "#1abc9c", "KC": "#16a085",
    "SV": "#2980b9", "CS": "#8e44ad",
}
PITCH_LABELS = {
    "FF": "フォーシーム", "SI": "シンカー", "FC": "カッター",
    "SL": "スライダー", "ST": "スイーパー", "CU": "カーブ",
    "CH": "チェンジアップ", "FS": "スプリット", "KC": "ナックルカーブ",
    "SV": "スローカーブ", "CS": "スローカーブ",
}
# 結果別カラー（番号付き円マーカー用）
RESULT_COLORS = {
    "ball":                    "#43a047",  # 緑
    "called_strike":           "#e53935",  # 赤
    "swinging_strike":         "#fb8c00",  # オレンジ
    "swinging_strike_blocked": "#fb8c00",
    "foul":                    "#fdd835",  # 黄
    "foul_tip":                "#fdd835",
    "hit_into_play":           "#1e88e5",  # 青
    "blocked_ball":            "#78909c",  # グレー
    "pitchout":                "#78909c",
}
RESULT_TEXT_COLORS = {
    "ball": "#fff", "called_strike": "#fff",
    "swinging_strike": "#fff", "swinging_strike_blocked": "#fff",
    "foul": "#222", "foul_tip": "#222",
    "hit_into_play": "#fff", "blocked_ball": "#fff", "pitchout": "#fff",
}
DESC_LABELS = {
    "ball": "ボール", "called_strike": "見逃し",
    "swinging_strike": "空振り", "swinging_strike_blocked": "空振り(ブロック)",
    "foul": "ファウル", "foul_tip": "ファウルチップ",
    "hit_into_play": "インプレー", "blocked_ball": "ブロック", "pitchout": "ピッチアウト",
}
# インプレー結果ごとの Plotly マーカー記号
INPLAY_SYMBOLS = {"out": "x", "hit": "diamond", "hr": "star"}
# MLB Stats API description → Statcast style
MLB_DESC_MAP = {
    "ball": "ball", "called strike": "called_strike",
    "swinging strike": "swinging_strike", "swinging strike (blocked)": "swinging_strike_blocked",
    "foul": "foul", "foul tip": "foul_tip", "foul bunt": "foul",
    "in play, no out": "hit_into_play", "in play, out(s)": "hit_into_play",
    "in play, run(s)": "hit_into_play", "blocked ball": "blocked_ball",
    "pitchout": "pitchout", "intent ball": "ball",
    "automatic ball": "ball", "automatic strike": "called_strike",
    "hit by pitch": "ball",
}
METRIC_HELP = {
    "wOBA":  "加重出塁率（Weighted On-Base Average）。打席結果に打点価値の重みをつけた指標。リーグ平均は約0.320。高いほど優秀。",
    "xwOBA": "期待加重出塁率。打球の速度・角度から計算した「本来の」wOBA。守備の運不運を除いた真の打力を示す。",
    "xBA":   "期待打率。打球速度・角度から算出した理論上の打率。守備シフトや運を排除した実力値。",
    "K率":   "三振率（Strikeout Rate）。打席のうち三振になる割合。低いほど良い。MLBリーグ平均は約22〜23%。",
    "BB率":  "四球率（Walk Rate）。打席のうち四球になる割合。高いほど選球眼が良い。リーグ平均は約8〜9%。",
    "空振り率": "Whiff Rate。スイングのうち空振りになる割合。低いほどコンタクト能力が高い。リーグ平均は約25%。",
    "バレル率": "Barrel%。打球速度・角度の組み合わせが「理想的な強い打球」とされる条件を満たした割合。打球（インプレー）に対する割合。リーグ平均は約8%、優秀な打者は15%以上。",
    "ハードヒット率": "Hard-Hit%。打球速度95mph以上の打球の割合。打者の長打力・強さを示す。リーグ平均は約35〜40%。",
}

# ─── ヘルパー ──────────────────────────────────────────────
def _inplay_kind(event_val) -> str:
    """インプレー結果を 'out' / 'hit' / 'hr' に分類する。"""
    if event_val is None or (isinstance(event_val, float) and np.isnan(event_val)):
        return "out"
    ev = str(event_val).lower().replace(" ", "_")
    if ev == "home_run":
        return "hr"
    if ev in ("single", "double", "triple"):
        return "hit"
    return "out"

def fmt(v, digits=3):
    return f"{v:.{digits}f}" if v is not None else "---"

def pct(v):
    return f"{v*100:.1f}%" if v is not None else "---"

def grade(woba):
    if woba is None: return "---", "#666"
    if woba >= 0.400: return "大当たり ◎", "#27ae60"
    if woba >= 0.350: return "好調 ○", "#2ecc71"
    if woba >= 0.300: return "普通 △", "#f39c12"
    return "苦手 ×", "#e74c3c"

def pitcher_grade(xwoba):
    if xwoba is None: return "---", "#666"
    if xwoba <= 0.280: return "支配的 ◎", "#27ae60"
    if xwoba <= 0.330: return "好投 ○", "#2ecc71"
    if xwoba <= 0.370: return "普通 △", "#f39c12"
    return "苦戦 ×", "#e74c3c"

def pitch_badge(pt, pct_val=None):
    color = PITCH_COLORS.get(pt, "#555")
    label = PITCH_LABELS.get(pt, pt)
    extra = f" {pct_val*100:.0f}%" if pct_val else ""
    return f'<span class="pitch-badge" style="background:{color}22;color:{color};border:1px solid {color}">{pt} {label}{extra}</span>'

# ─── ストライクゾーン共通シェイプ ──────────────────────────
SZ_X   = 0.708   # 半幅（ft）
SZ_BOT = 1.55
SZ_TOP = 3.40

def _add_zone_shapes(fig: go.Figure) -> None:
    """ストライクゾーン + 3×3グリッド + ホームプレートを追加"""
    # ゾーン枠
    fig.add_shape(type="rect",
        x0=-SZ_X, x1=SZ_X, y0=SZ_BOT, y1=SZ_TOP,
        line=dict(color="white", width=2.5),
        fillcolor="rgba(255,255,255,0.07)",
    )
    # 縦 2本
    col_w = SZ_X * 2 / 3
    for x in (-SZ_X + col_w, -SZ_X + col_w * 2):
        fig.add_shape(type="line", x0=x, x1=x, y0=SZ_BOT, y1=SZ_TOP,
                      line=dict(color="rgba(255,255,255,0.45)", width=1))
    # 横 2本
    row_h = (SZ_TOP - SZ_BOT) / 3
    for y in (SZ_BOT + row_h, SZ_BOT + row_h * 2):
        fig.add_shape(type="line", x0=-SZ_X, x1=SZ_X, y0=y, y1=y,
                      line=dict(color="rgba(255,255,255,0.45)", width=1))
    # ホームプレート（五角形）
    fig.add_shape(type="path",
        path="M -0.708,0.42 L 0.708,0.42 L 0.708,0.16 L 0,0 L -0.708,0.16 Z",
        fillcolor="rgba(255,255,255,0.88)",
        line=dict(color="white", width=1.5),
    )

def _zone_layout(title: str) -> dict:
    # x: -1.5〜1.5 (3 ft)、y: 0.3〜4.3 (4 ft) → データ空間 3:4 でゾーンの縦横比が正確になる
    return dict(
        title=dict(text=title, font=dict(size=13, color="#ccc"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,50,15,1)",
        xaxis=dict(
            range=[-1.5, 1.5],
            tickfont=dict(color="#aaa"),
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            title=dict(text="← 外角（大谷左打ち）　内角 →", font=dict(color="#aaa", size=11)),
        ),
        yaxis=dict(
            range=[0.3, 4.3],
            tickfont=dict(color="#aaa"),
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            title=dict(text="高さ (ft)", font=dict(color="#aaa", size=11)),
            scaleanchor="x",   # x と等尺にしてゾーンの縦横比を正確に保つ
            scaleratio=1,
        ),
        legend=dict(font=dict(color="#ccc", size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=120),
        height=500,
        hoverlabel=dict(bgcolor="#1a2535", font_color="#fff", font_size=13),
    )

# ─── 過去対戦図（球種別カラー点群） ────────────────────────
def build_zone_fig_history(pitch_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    _add_zone_shapes(fig)

    for pt, color in PITCH_COLORS.items():
        sub = pitch_df[pitch_df["pitch_type"] == pt].dropna(subset=["plate_x", "plate_z"])
        if sub.empty:
            continue
        label_jp = PITCH_LABELS.get(pt, pt)
        hover = [
            f"<b>{pt} {label_jp}</b><br>"
            f"球速: {row.release_speed:.1f} mph<br>"
            f"結果: {DESC_LABELS.get(row.description, row.description)}<br>"
            f"日付: {row.game_date.strftime('%m/%d') if pd.notna(row.game_date) else ''}"
            for row in sub.itertuples()
        ]
        # インプレー結果で形状を分ける（色は球種色を維持）
        symbols = [
            INPLAY_SYMBOLS[_inplay_kind(getattr(r, "events", None))]
            if str(getattr(r, "description", "")) == "hit_into_play"
            else "circle"
            for r in sub.itertuples()
        ]
        fig.add_trace(go.Scatter(
            x=sub["plate_x"], y=sub["plate_z"],
            mode="markers",
            name=f"{pt} {label_jp}",
            marker=dict(color=color, size=10, opacity=0.82,
                        symbol=symbols,
                        line=dict(color="rgba(0,0,0,0.5)", width=0.8)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(**_zone_layout(title))
    return fig

# ─── 今試合図（結果色 × 投球順番号マーカー） ───────────────
def build_zone_fig_live(pitch_df: pd.DataFrame, title: str) -> go.Figure:
    """1球ずつ番号付き円マーカー。参照画像スタイル。"""
    fig = go.Figure()
    _add_zone_shapes(fig)

    if pitch_df.empty:
        fig.update_layout(**_zone_layout(title))
        return fig

    # 凡例トレース（ダミー、表示用）
    shown_descs: set[str] = set()
    for desc, color in RESULT_COLORS.items():
        if desc in ("swinging_strike_blocked", "foul_tip", "pitchout", "blocked_ball", "hit_into_play"):
            continue  # インプレーは下で3種に分けて表示
        shown_descs.add(desc)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=12, symbol="circle"),
            name=DESC_LABELS.get(desc, desc),
            legendgroup=desc,
        ))
    # インプレー3種の凡例
    ip_color = RESULT_COLORS["hit_into_play"]
    for kind, sym, label in [
        ("out", "x",       "インプレー（アウト）"),
        ("hit", "diamond", "インプレー（安打）"),
        ("hr",  "star",    "インプレー（本塁打）"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=ip_color, size=12, symbol=sym),
            name=label,
            legendgroup=f"inplay_{kind}",
        ))

    # 打席ごとに番号をリセット
    pitch_num = 0
    for _, row in pitch_df.sort_values(["at_bat_number", "pitch_number"]).iterrows():
        if pd.isna(row.get("plate_x")) or pd.isna(row.get("plate_z")):
            continue
        pitch_num += 1
        desc = str(row.get("description", "")).lower().replace(" ", "_")
        color = RESULT_COLORS.get(desc, "#78909c")
        txt_color = RESULT_TEXT_COLORS.get(desc, "#fff")
        pt = row.get("pitch_type", "")
        pt_label = PITCH_LABELS.get(pt, pt)
        velo = f"{row['release_speed']:.0f}" if pd.notna(row.get("release_speed")) else "?"
        desc_label = DESC_LABELS.get(desc, desc)
        ev = row.get("events", "")
        ev_str = f" → {ev}" if pd.notna(ev) and ev else ""
        ab_num = int(row.get("at_bat_number", 0))

        hover = (
            f"<b>#{pitch_num}　第{ab_num}打席</b><br>"
            f"{pt} {pt_label}　{velo} mph<br>"
            f"{desc_label}{ev_str}"
        )

        # インプレーは形状で打席結果を表現、それ以外は番号付き円
        if desc == "hit_into_play":
            kind   = _inplay_kind(ev)
            symbol = INPLAY_SYMBOLS[kind]
            fig.add_trace(go.Scatter(
                x=[row["plate_x"]], y=[row["plate_z"]],
                mode="markers+text",
                marker=dict(color=color, size=24, symbol=symbol,
                            line=dict(color="white", width=2.0)),
                text=[str(pitch_num)],
                textfont=dict(color="#fff", size=9, family="Arial Black"),
                textposition="top center",
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[row["plate_x"]], y=[row["plate_z"]],
                mode="markers+text",
                marker=dict(color=color, size=26, symbol="circle",
                            line=dict(color="white", width=1.8)),
                text=[str(pitch_num)],
                textfont=dict(color=txt_color, size=10, family="Arial Black"),
                textposition="middle center",
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(**_zone_layout(title))
    return fig

# ─── コース別ヒートマップ（今季Stats） ──────────────────────
def build_zone_heatmap_fig(grid_data: dict, metric: str) -> go.Figure:
    grid   = grid_data["grid"]
    n_grid = grid_data["n_grid"]

    col_w = SZ_X * 2 / 3
    row_h = (SZ_TOP - SZ_BOT) / 3
    x_centers = [-SZ_X + col_w * (i + 0.5) for i in range(3)]
    y_centers = [SZ_BOT + row_h * (i + 0.5) for i in range(3)]   # 下→上

    z_display = grid[::-1]      # grid[0]=上段 なので反転して下→上の y_centers に合わせる
    n_display = n_grid[::-1]

    if metric == "xwoba":
        colorscale = [[0, "#1565c0"], [0.5, "#37474f"], [1, "#c62828"]]
        zmin, zmax = 0.150, 0.550
        cell_fmt = lambda v: f"{v:.3f}"
    else:  # whiff_rate
        colorscale = [[0, "#c62828"], [0.5, "#37474f"], [1, "#1565c0"]]
        zmin, zmax = 0.05, 0.45
        cell_fmt = lambda v: f"{v*100:.0f}%"

    text = [[
        (cell_fmt(z_display[r][c]) if not np.isnan(z_display[r][c]) else "—") + f"<br>n={n_display[r][c]}"
        for c in range(3)
    ] for r in range(3)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_centers, y=y_centers, z=z_display,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        text=text, texttemplate="%{text}",
        textfont=dict(color="#fff", size=13),
        showscale=False, hoverinfo="skip",
        xgap=2, ygap=2,
    ))
    _add_zone_shapes(fig)
    layout = _zone_layout("")
    layout["margin"] = dict(l=10, r=10, t=10, b=40)
    layout["height"] = 420
    fig.update_layout(**layout)
    return fig

# ─── 打球質（EV × 打球角度） ─────────────────────────────
BB_TYPE_COLORS = {
    "line_drive": "#43a047", "fly_ball": "#1e88e5",
    "ground_ball": "#8d6e63", "popup": "#fb8c00",
}
BB_TYPE_LABELS = {
    "line_drive": "ライナー", "fly_ball": "フライ",
    "ground_ball": "ゴロ", "popup": "ポップ",
}

def build_ev_scatter_fig(ev_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "bb_type" not in ev_df.columns:
        fig.add_trace(go.Scatter(
            x=ev_df["launch_angle"], y=ev_df["launch_speed"],
            mode="markers", name="打球",
            marker=dict(color="#0078ff", size=7, opacity=0.65),
            hovertemplate="角度 %{x:.0f}°　速度 %{y:.1f} mph<extra></extra>",
        ))
    else:
        for bb, color in BB_TYPE_COLORS.items():
            sub = ev_df[ev_df["bb_type"] == bb]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["launch_angle"], y=sub["launch_speed"],
                mode="markers", name=BB_TYPE_LABELS.get(bb, bb),
                marker=dict(color=color, size=7, opacity=0.65),
                hovertemplate="角度 %{x:.0f}°　速度 %{y:.1f} mph<extra></extra>",
            ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,25,35,1)",
        xaxis=dict(title=dict(text="打球角度 (°)", font=dict(color="#aaa", size=11)),
                   range=[-60, 60], tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=dict(text="打球速度 (mph)", font=dict(color="#aaa", size=11)),
                   range=[30, 125], tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(font=dict(color="#ccc", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=20, b=70),
        height=380,
        hoverlabel=dict(bgcolor="#1a2535", font_color="#fff"),
    )
    return fig

def build_ev_hist_fig(ev_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ev_df["launch_speed"], nbinsx=20, marker=dict(color="#0078ff")))
    fig.add_vline(x=95, line=dict(color="#fdd835", dash="dash", width=1.5))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,25,35,1)",
        xaxis=dict(title=dict(text="打球速度 (mph)　点線=ハードヒット基準95mph", font=dict(color="#aaa", size=11)),
                   tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=dict(text="本数", font=dict(color="#aaa", size=11)),
                   tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10, r=10, t=20, b=50),
        height=380, showlegend=False,
        hoverlabel=dict(bgcolor="#1a2535", font_color="#fff"),
    )
    return fig

def build_monthly_trend_fig(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df["month"], y=monthly_df["woba"], mode="lines+markers",
                              name="wOBA", line=dict(color="#0078ff", width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=monthly_df["month"], y=monthly_df["xwoba"], mode="lines+markers",
                              name="xwOBA", line=dict(color="#fdd835", width=2, dash="dot"), marker=dict(size=7)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,25,35,1)",
        xaxis=dict(tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=dict(text="wOBA / xwOBA", font=dict(color="#aaa", size=11)),
                   tickfont=dict(color="#aaa"), gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(font=dict(color="#ccc", size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=40),
        height=320,
        hoverlabel=dict(bgcolor="#1a2535", font_color="#fff"),
    )
    return fig

# ─── MLB Stats API ライブフィード取得 ──────────────────────
@st.cache_data(ttl=60, show_spinner=False)   # 60秒キャッシュ（ライブ更新対応）
def fetch_live_pitches(game_pk: int) -> tuple[pd.DataFrame, str]:
    """
    Ohtani 打席の投球データを MLB Stats API ライブフィードから取得。
    返り値: (df, game_status)  game_status = "Preview" / "Live" / "Final"
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(), "Error"

    game_status = data.get("gameData", {}).get("status", {}).get("abstractGameState", "Preview")
    if game_status == "Preview":
        return pd.DataFrame(), game_status

    rows = []
    all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play in all_plays:
        if play.get("matchup", {}).get("batter", {}).get("id") != OHTANI_ID:
            continue
        ab_num = play.get("atBatIndex", 0) + 1
        play_events = play.get("playEvents", [])
        for event in play_events:
            if event.get("type") != "pitch":
                continue
            pd_data = event.get("pitchData", {})
            coords = pd_data.get("coordinates", {})
            details = event.get("details", {})
            raw_desc = details.get("description", "").lower()
            desc = MLB_DESC_MAP.get(raw_desc, raw_desc.replace(" ", "_"))

            # 最後のイベントにのみ result event を付与
            is_last = (event == play_events[-1])
            ev = play.get("result", {}).get("event", "") if is_last else None

            rows.append({
                "game_date": pd.Timestamp("today"),
                "at_bat_number": ab_num,
                "pitch_number": event.get("pitchNumber", len(rows) + 1),
                "pitch_type": details.get("type", {}).get("code", ""),
                "release_speed": pd_data.get("startSpeed"),
                "plate_x": coords.get("pX"),
                "plate_z": coords.get("pZ"),
                "description": desc,
                "events": ev,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(), game_status

# ─── キャッシュ付きデータ読み込み ──────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule_cached():
    return _load_schedule()

@st.cache_data(ttl=3600, show_spinner=False)
def run_prediction_cached(game_date_str):
    return run_prediction(game_date=game_date_str)

@st.cache_data(ttl=3600, show_spinner=False)
def load_ohtani_batter_cached():
    files = sorted((DATA_DIR / "statcast").glob("ohtani_batter_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

# ══════════════════════════════════════════════════════════
# メイン UI
# ══════════════════════════════════════════════════════════
st.markdown("## ⚾ 大谷翔平 対戦予想")
st.caption("Shohei Ohtani Matchup Predictor")

ohtani_batter_df = load_ohtani_batter_cached()

tab_pred, tab_stats = st.tabs(["⚾ 対戦予想", "📊 今季Stats"])

with tab_pred:
    try:
        schedule = load_schedule_cached()
    except FileNotFoundError as e:
        st.error(f"スケジュールファイルが見つかりません: {e}")
        st.stop()

    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    today = pd.Timestamp("today").normalize()
    future_sched = schedule[schedule["game_date"] >= today].sort_values("game_date")

    if future_sched.empty:
        st.warning("今後の試合がスケジュールに見つかりません。")
        st.stop()

    game_dates = future_sched["game_date"].dt.strftime("%Y-%m-%d").tolist()
    game_labels = []
    for _, row in future_sched.iterrows():
        ha = "🏠" if row["home_away"] == "home" else "✈"
        flag = " ⚡先発" if row.get("ohtani_starting_pitcher") else ""
        game_labels.append(f"{row['game_date'].strftime('%m/%d')} {ha} vs {row['opponent_team']}{flag}")

    selected_idx = st.selectbox("試合を選択", range(len(game_dates)),
                                 format_func=lambda i: game_labels[i], index=0)
    selected_date  = game_dates[selected_idx]
    selected_game  = future_sched.iloc[selected_idx]
    selected_gamepk = int(selected_game["game_pk"]) if pd.notna(selected_game.get("game_pk")) else None

    ha_emoji = "🏠 HOME" if selected_game["home_away"] == "home" else "✈ AWAY"
    st.markdown(f"""
    ### {selected_game['game_date'].strftime('%Y年%m月%d日')} &nbsp; {ha_emoji}
    **vs {selected_game['opponent_team']}**　相手先発: **{selected_game['opp_starter_name']}**
    """)
    if selected_game.get("ohtani_starting_pitcher"):
        st.success("⚡ 大谷 本日先発登板予定")

    st.divider()

    with st.spinner("予測計算中..."):
        try:
            output = run_prediction_cached(selected_date)
        except Exception as e:
            st.error(f"予測エラー: {e}")
            st.stop()

    # ── ① 大谷（打者）予測 ────────────────────────────────────
    st.markdown('<div class="section-header">🥊 大谷（打者）予測</div>', unsafe_allow_html=True)

    batter_out  = output.get("ohtani_batter", {})
    batter_pred = batter_out.get("prediction") or {}
    method      = batter_out.get("method", "")
    note        = batter_out.get("note", "")

    method_label = {
        "direct_history": "✅ 直接対面実績",
        "similarity_weighted": "🔍 類似投手推定",
        "insufficient_data": "⚠ データ不足",
    }.get(method, method)

    st.markdown(f'<span class="method-badge">{method_label} — {note}</span>', unsafe_allow_html=True)

    if method == "insufficient_data" or not batter_pred:
        st.markdown('<div class="warn-box">相手投手データが不足しています。collect_opponents.py を実行後、再読み込みしてください。</div>', unsafe_allow_html=True)
    else:
        woba_val  = batter_pred.get("woba")
        xwoba_val = batter_pred.get("xwoba")
        grade_label, grade_color = grade(xwoba_val or woba_val)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a2535,#0d1a2a);border-radius:16px;
                    padding:16px 20px;margin:8px 0 16px 0;border:1px solid #243447;">
          <div style="font-size:0.8rem;color:#8899aa;margin-bottom:4px">総合評価</div>
          <div style="font-size:2.2rem;font-weight:800;color:{grade_color}">{grade_label}</div>
          <div style="font-size:0.8rem;color:#8899aa;margin-top:4px">vs {batter_out.get('opponent_pitcher','---')}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("wOBA",   fmt(woba_val),              help=METRIC_HELP["wOBA"])
        with c2: st.metric("xwOBA",  fmt(xwoba_val),             help=METRIC_HELP["xwOBA"])
        with c3: st.metric("xBA",    fmt(batter_pred.get("xba")), help=METRIC_HELP["xBA"])

        c4, c5, c6 = st.columns(3)
        with c4: st.metric("K率",    pct(batter_pred.get("k_rate")),    help=METRIC_HELP["K率"])
        with c5: st.metric("BB率",   pct(batter_pred.get("bb_rate")),   help=METRIC_HELP["BB率"])
        with c6: st.metric("空振り率", pct(batter_pred.get("whiff_rate")), help=METRIC_HELP["空振り率"])

        if method == "similarity_weighted":
            n_sim = batter_pred.get("n_similar", 0)
            with st.expander(f"参照した類似投手 ({n_sim} 人)"):
                st.caption(f"{batter_out.get('opponent_pitcher','')} の球種特性に近い投手をもとに算出")
                ids = batter_pred.get("similar_pitcher_ids", [])
                if ids:
                    st.code(", ".join(str(x) for x in ids[:10]))

    st.divider()

    # ── ② 球コース図 ──────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 球コース図（大谷打席）</div>', unsafe_allow_html=True)

    opp_pitcher_id    = output.get("ohtani_batter", {}).get("opponent_pitcher_id")
    pitcher_name      = output.get("ohtani_batter", {}).get("opponent_pitcher", "")

    # 凡例説明
    with st.expander("色の見方"):
        cols = st.columns(3)
        legend_items = [
            ("ボール", "#43a047"), ("見逃し", "#e53935"), ("空振り", "#fb8c00"),
            ("ファウル", "#fdd835"), ("インプレー", "#1e88e5"),
        ]
        for i, (label, color) in enumerate(legend_items):
            with cols[i % 3]:
                st.markdown(
                    f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
                    f'background:{color};margin-right:6px;vertical-align:middle"></span>{label}',
                    unsafe_allow_html=True
                )

    # タブ構成
    tab_labels = []
    if opp_pitcher_id and not ohtani_batter_df.empty:
        direct_df = ohtani_batter_df[ohtani_batter_df["pitcher"] == int(opp_pitcher_id)]
        if len(direct_df) >= 3:
            tab_labels.append(f"🕰 過去 vs {pitcher_name}（{len(direct_df)}球）")
    tab_labels.append("🕰 直近5試合")
    tab_labels.append("⚡ 今試合 LIVE")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # 過去 vs 相手投手
    if len(tab_labels) == 3:
        with tabs[tab_idx]:
            fig = build_zone_fig_history(direct_df, f"過去 大谷 vs {pitcher_name}")
            st.plotly_chart(fig, use_container_width=True)
        tab_idx += 1

    # 直近5試合
    with tabs[tab_idx]:
        if not ohtani_batter_df.empty:
            recent_dates = sorted(ohtani_batter_df["game_date"].dt.date.unique())[-5:]
            recent_df    = ohtani_batter_df[ohtani_batter_df["game_date"].dt.date.isin(recent_dates)]
            fig2 = build_zone_fig_history(recent_df, f"大谷 直近5試合 ({len(recent_df)}球)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("データがありません。")
    tab_idx += 1

    # 今試合 LIVE
    with tabs[tab_idx]:
        if selected_gamepk is None:
            st.info("今試合の game_pk が不明です。")
        else:
            with st.spinner("ライブデータ取得中..."):
                live_df, game_status = fetch_live_pitches(selected_gamepk)

            status_labels = {
                "Preview": "⏳ 試合前",
                "Live":    "🔴 試合中",
                "Final":   "✅ 試合終了",
                "Error":   "⚠ 取得エラー",
            }
            st.caption(status_labels.get(game_status, game_status))

            if game_status == "Preview":
                st.info("試合がまだ始まっていません。開始後に自動更新されます。")
            elif live_df.empty:
                st.info("大谷の打席データがまだありません。")
            else:
                n_ab = live_df["at_bat_number"].nunique()
                n_p  = len(live_df.dropna(subset=["plate_x", "plate_z"]))
                fig3 = build_zone_fig_live(
                    live_df,
                    f"今試合 大谷打席 — {n_ab}打席 {n_p}球"
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("● の数字は投球順。球種は hover/タップで確認。")

            if game_status == "Live":
                if st.button("🔄 LIVE 更新", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()

    st.divider()

    # ── ③ 大谷（投手）予測 ────────────────────────────────────
    st.markdown('<div class="section-header">⚾ 大谷（投手）予測</div>', unsafe_allow_html=True)

    pitcher_out    = output.get("ohtani_pitcher", {})
    pitcher_method = pitcher_out.get("method", "")

    if not output.get("ohtani_starting_pitcher"):
        st.info("この試合で大谷は先発投手ではありません。")
    elif pitcher_method == "insufficient_data":
        st.markdown(f'<div class="warn-box">{pitcher_out.get("note","データ不足")}</div>', unsafe_allow_html=True)
    else:
        team_xwoba = pitcher_out.get("team_avg_xwoba_allowed")
        team_whiff = pitcher_out.get("team_avg_whiff_rate")
        p_grade_label, p_grade_color = pitcher_grade(team_xwoba)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a2535,#0d1a2a);border-radius:16px;
                    padding:16px 20px;margin:8px 0 16px 0;border:1px solid #243447;">
          <div style="font-size:0.8rem;color:#8899aa;margin-bottom:4px">投球評価 vs {pitcher_out.get('opponent_team','')}</div>
          <div style="font-size:2.2rem;font-weight:800;color:{p_grade_color}">{p_grade_label}</div>
          <div style="font-size:0.8rem;color:#8899aa;margin-top:4px">
            被xwOBA {fmt(team_xwoba)} / 空振り率 {pct(team_whiff)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca: st.metric("被 xwOBA", fmt(team_xwoba), help=METRIC_HELP["xwOBA"] + "（低いほど大谷に有利）")
        with cb: st.metric("空振り率", pct(team_whiff),  help=METRIC_HELP["空振り率"] + "（高いほど大谷に有利）")

        pitch_mix = pitcher_out.get("ohtani_pitch_mix", {})
        if pitch_mix:
            st.markdown("**大谷の球種構成（今シーズン）**")
            sorted_mix = sorted(pitch_mix.items(), key=lambda x: -x[1])
            badges = "".join(pitch_badge(pt, v) for pt, v in sorted_mix if v >= 0.02)
            st.markdown(badges, unsafe_allow_html=True)
            mix_df = pd.DataFrame(
                [(PITCH_LABELS.get(pt, pt), round(v*100, 1)) for pt, v in sorted_mix if v >= 0.02],
                columns=["球種", "割合(%)"]
            ).set_index("球種")
            st.bar_chart(mix_df, use_container_width=True, height=200)

        st.caption(f"分析対象打者: {pitcher_out.get('n_batters_analyzed', 0)} 人")

    st.divider()

with tab_stats:
    if ohtani_batter_df.empty:
        st.info("打者データがありません。collect_statcast.py を実行してください。")
    else:
        years = sorted(ohtani_batter_df["game_date"].dt.year.unique(), reverse=True)
        season_choice = st.radio(
            "シーズン", ["全期間"] + [str(y) for y in years],
            horizontal=True,
        )
        if season_choice == "全期間":
            season_df = ohtani_batter_df
        else:
            season_df = ohtani_batter_df[ohtani_batter_df["game_date"].dt.year == int(season_choice)]

        summary = compute_summary(season_df)
        if not summary:
            st.info("データがありません。")
        else:
            st.markdown('<div class="section-header">📋 サマリー</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("打席数", summary.get("n_pa", "---"))
            with s2: st.metric("wOBA", fmt(summary.get("woba")), help=METRIC_HELP["wOBA"])
            with s3: st.metric("xwOBA", fmt(summary.get("xwoba")), help=METRIC_HELP["xwOBA"])
            with s4: st.metric("K率", pct(summary.get("k_rate")), help=METRIC_HELP["K率"])

            s5, s6, s7, s8 = st.columns(4)
            with s5: st.metric("BB率", pct(summary.get("bb_rate")), help=METRIC_HELP["BB率"])
            with s6: st.metric("バレル率", pct(summary.get("barrel_rate")), help=METRIC_HELP["バレル率"])
            with s7: st.metric("ハードヒット率", pct(summary.get("hard_hit_rate")), help=METRIC_HELP["ハードヒット率"])
            with s8: st.metric("平均EV", f"{summary.get('avg_ev')} mph" if summary.get("avg_ev") is not None else "---")

            st.divider()

            # ── 得意・苦手な投手タイプ 要約 ──────────────────────
            st.markdown('<div class="section-header">🔍 得意・苦手な投手タイプ</div>', unsafe_allow_html=True)
            pt_split = compute_pitcher_type_split(season_df)
            if not pt_split:
                st.info("データが不足しています。")
            else:
                st.caption(
                    f"分析対象: {pt_split['n_pitchers']} 投手 ／ "
                    f"全体 xwOBA: {pt_split['overall_xwoba']:.3f}　"
                    f"（全体比 ±0.040 以上で得意・苦手と判定）"
                )
                good_list = pt_split["top_good"]
                bad_list  = pt_split["top_bad"]

                g_col, b_col = st.columns(2)
                with g_col:
                    st.markdown("##### ✅ 得意な投手タイプ")
                    if not good_list:
                        st.caption("該当なし（全体平均と大きな差なし）")
                    for item in good_list:
                        diff = item["xwoba"] - pt_split["overall_xwoba"]
                        st.markdown(f"""
<div style="background:#0a2818;border:1px solid #27ae60;border-radius:8px;
            padding:10px 14px;margin:4px 0;display:flex;justify-content:space-between;align-items:center">
  <div>
    <span style="color:#2ecc71;font-weight:700;font-size:0.95rem">{item['label']}</span><br>
    <span style="color:#668866;font-size:0.75rem">{item['n']} 投手</span>
  </div>
  <div style="text-align:right">
    <span style="color:#fff;font-weight:800;font-size:1.1rem">xwOBA {item['xwoba']:.3f}</span><br>
    <span style="color:#2ecc71;font-size:0.78rem">全体比 +{diff:.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)

                with b_col:
                    st.markdown("##### ❌ 苦手な投手タイプ")
                    if not bad_list:
                        st.caption("該当なし（全体平均と大きな差なし）")
                    for item in bad_list:
                        diff = item["xwoba"] - pt_split["overall_xwoba"]
                        st.markdown(f"""
<div style="background:#280a0a;border:1px solid #e74c3c;border-radius:8px;
            padding:10px 14px;margin:4px 0;display:flex;justify-content:space-between;align-items:center">
  <div>
    <span style="color:#e74c3c;font-weight:700;font-size:0.95rem">{item['label']}</span><br>
    <span style="color:#886666;font-size:0.75rem">{item['n']} 投手</span>
  </div>
  <div style="text-align:right">
    <span style="color:#fff;font-weight:800;font-size:1.1rem">xwOBA {item['xwoba']:.3f}</span><br>
    <span style="color:#e74c3c;font-size:0.78rem">全体比 {diff:.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)

                with st.expander("全カテゴリ一覧"):
                    all_df = pd.DataFrame(pt_split["all_categories"])
                    thresh = 0.040
                    all_df["評価"] = all_df["xwoba"].map(
                        lambda v: "✅ 得意" if v >= pt_split["overall_xwoba"] + thresh
                        else ("❌ 苦手" if v <= pt_split["overall_xwoba"] - thresh else "— 普通")
                    )
                    st.dataframe(
                        all_df.rename(columns={"label": "投手タイプ", "n": "投手数", "xwoba": "xwOBA"})[
                            ["投手タイプ", "投手数", "xwOBA", "評価"]
                        ],
                        hide_index=True, use_container_width=True,
                    )

            st.divider()

            # ── コース別ヒートマップ ──────────────────────────
            st.markdown('<div class="section-header">🎯 コース別 得意・苦手</div>', unsafe_allow_html=True)
            heat_tab1, heat_tab2 = st.tabs(["xwOBA", "空振り率"])
            with heat_tab1:
                grid_data = compute_zone_grid(season_df, metric="xwoba")
                if grid_data is None:
                    st.info("データが不足しています。")
                else:
                    st.plotly_chart(build_zone_heatmap_fig(grid_data, "xwoba"), use_container_width=True)
                    st.caption("赤=得意（高xwOBA）　青=苦手（低xwOBA）")
            with heat_tab2:
                grid_data_w = compute_zone_grid(season_df, metric="whiff_rate")
                if grid_data_w is None:
                    st.info("データが不足しています。")
                else:
                    st.plotly_chart(build_zone_heatmap_fig(grid_data_w, "whiff_rate"), use_container_width=True)
                    st.caption("青=空振りが多い（苦手）　赤=空振りが少ない（得意）")

            st.divider()

            # ── 球種別スプリット ──────────────────────────
            st.markdown('<div class="section-header">⚾ 球種別 得意・苦手</div>', unsafe_allow_html=True)
            pitch_split_df = compute_pitch_split(season_df)
            if pitch_split_df.empty:
                st.info("データが不足しています。")
            else:
                disp = pitch_split_df.copy()
                disp["球種"] = disp["pitch_type"].map(lambda pt: f"{pt} {PITCH_LABELS.get(pt, pt)}")
                disp["xwOBA"] = disp["xwoba"].map(lambda v: fmt(v) if v is not None else "---")
                disp["空振り率"] = disp["whiff_rate"].map(lambda v: pct(v) if v is not None else "---")
                disp["平均EV"] = disp["avg_ev"].map(lambda v: f"{v} mph" if v is not None else "---")
                disp["投球数"] = disp["n"]
                st.dataframe(
                    disp[["球種", "投球数", "xwOBA", "空振り率", "平均EV"]],
                    hide_index=True, use_container_width=True,
                )

            st.divider()

            # ── 打球質 ──────────────────────────
            st.markdown('<div class="section-header">💥 打球質（EV / 打球角度）</div>', unsafe_allow_html=True)
            ev_df = compute_ev_stats(season_df)
            if ev_df.empty:
                st.info("データが不足しています。")
            else:
                ev_c1, ev_c2 = st.columns(2)
                with ev_c1:
                    st.plotly_chart(build_ev_scatter_fig(ev_df), use_container_width=True)
                with ev_c2:
                    st.plotly_chart(build_ev_hist_fig(ev_df), use_container_width=True)

            st.divider()

            # ── 月別トレンド ──────────────────────────
            st.markdown('<div class="section-header">📈 月別 wOBA / xwOBA トレンド</div>', unsafe_allow_html=True)
            monthly_df = compute_monthly_trend(season_df)
            if monthly_df.empty:
                st.info("データが不足しています（最低打席数に達した月がありません）。")
            else:
                st.plotly_chart(build_monthly_trend_fig(monthly_df), use_container_width=True)

            st.divider()

            # ── 対左右投手 / カウント別 ──────────────────────────
            st.markdown('<div class="section-header">🔄 対左右投手 / カウント別</div>', unsafe_allow_html=True)
            sp_c1, sp_c2 = st.columns(2)
            with sp_c1:
                st.markdown("**対左右投手**")
                lr_df = compute_lr_split(season_df)
                if lr_df.empty:
                    st.info("データが不足しています。")
                else:
                    disp_lr = lr_df.copy()
                    disp_lr["wOBA"] = disp_lr["woba"].map(lambda v: fmt(v) if v is not None else "---")
                    disp_lr["xwOBA"] = disp_lr["xwoba"].map(lambda v: fmt(v) if v is not None else "---")
                    disp_lr["空振り率"] = disp_lr["whiff_rate"].map(lambda v: pct(v) if v is not None else "---")
                    st.dataframe(
                        disp_lr.rename(columns={"hand": "対戦", "n_pa": "打席数"})[
                            ["対戦", "打席数", "wOBA", "xwOBA", "空振り率"]
                        ],
                        hide_index=True, use_container_width=True,
                    )
            with sp_c2:
                st.markdown("**カウント別**")
                count_df = compute_count_split(season_df)
                if count_df.empty:
                    st.info("データが不足しています。")
                else:
                    disp_c = count_df.copy()
                    disp_c["wOBA"] = disp_c["woba"].map(lambda v: fmt(v) if v is not None else "---")
                    disp_c["xwOBA"] = disp_c["xwoba"].map(lambda v: fmt(v) if v is not None else "---")
                    disp_c["空振り率"] = disp_c["whiff_rate"].map(lambda v: pct(v) if v is not None else "---")
                    st.dataframe(
                        disp_c.rename(columns={"count_group": "カウント", "n_pa": "打席数"})[
                            ["カウント", "打席数", "wOBA", "xwOBA", "空振り率"]
                        ],
                        hide_index=True, use_container_width=True,
                    )

# ─── フッター ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:0.7rem;color:#445566;padding:16px 0">
  データソース: Baseball Savant (Statcast) / MLB Stats API<br>
  ※予測は統計モデルによる推定値です
</div>
""", unsafe_allow_html=True)

if st.button("🔄 データ再読み込み", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
