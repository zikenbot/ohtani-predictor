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
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from predict import run_prediction, _load_schedule

DATA_DIR = Path(__file__).parent / "data"

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
    background: #1e2a3a;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 4px 0;
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
DESC_LABELS = {
    "ball": "ボール",
    "called_strike": "見逃しストライク",
    "swinging_strike": "空振り",
    "swinging_strike_blocked": "空振り",
    "foul": "ファウル",
    "foul_tip": "ファウルチップ",
    "hit_into_play": "インプレー",
    "blocked_ball": "ブロックボール",
    "pitchout": "ピッチアウト",
}
DESC_SYMBOLS = {
    "ball": "circle",
    "called_strike": "x",
    "swinging_strike": "x-open",
    "swinging_strike_blocked": "x-open",
    "foul": "triangle-up",
    "foul_tip": "triangle-up-open",
    "hit_into_play": "star",
    "blocked_ball": "circle-open",
}

# 指標の説明テキスト
METRIC_HELP = {
    "wOBA":  "加重出塁率（Weighted On-Base Average）。打席結果に打点価値の重みをつけた指標。リーグ平均は約0.320。高いほど優秀。",
    "xwOBA": "期待加重出塁率。打球の速度・角度から計算した「本来の」wOBA。守備の運不運を除いた真の打力を示す。",
    "xBA":   "期待打率。打球速度・角度から算出した理論上の打率。守備シフトや運を排除した実力値。",
    "K率":   "三振率（Strikeout Rate）。打席のうち三振になる割合。低いほど良い。MLBリーグ平均は約22〜23%。",
    "BB率":  "四球率（Walk Rate）。打席のうち四球になる割合。高いほど選球眼が良い。リーグ平均は約8〜9%。",
    "空振り率": "Whiff Rate。スイングのうち空振りになる割合。低いほどコンタクト能力が高い。リーグ平均は約25%。",
    "Swing%": "スイング率。投球のうちスイングする割合。",
}

# ─── ヘルパー ──────────────────────────────────────────────
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

# ─── 球コース図 ────────────────────────────────────────────
def build_pitch_zone_fig(pitch_df: pd.DataFrame, title: str) -> go.Figure:
    """Statcast の plate_x / plate_z をストライクゾーンに重ねた散布図を返す"""
    fig = go.Figure()

    # ストライクゾーン（規格: 幅 17 インチ = ±0.708 ft, 高さは Ohtani 実測平均値）
    SZ_X = 0.708
    SZ_BOT = 1.55
    SZ_TOP = 3.40

    # ゾーン背景
    fig.add_shape(type="rect",
        x0=-SZ_X, x1=SZ_X, y0=SZ_BOT, y1=SZ_TOP,
        line=dict(color="#ffffff", width=2),
        fillcolor="rgba(255,255,255,0.04)",
    )
    # 9分割グリッド
    for x in [-SZ_X + SZ_X*2/3, SZ_X - SZ_X*2/3]:
        fig.add_shape(type="line", x0=x, x1=x, y0=SZ_BOT, y1=SZ_TOP,
                      line=dict(color="rgba(255,255,255,0.27)", width=1))
    h_step = (SZ_TOP - SZ_BOT) / 3
    for y in [SZ_BOT + h_step, SZ_BOT + h_step*2]:
        fig.add_shape(type="line", x0=-SZ_X, x1=SZ_X, y0=y, y1=y,
                      line=dict(color="rgba(255,255,255,0.27)", width=1))

    # ホームプレート（五角形）
    plate_y = 0.3
    px_pts = [-0.708, 0.708, 0.708, 0, -0.708]
    py_pts = [plate_y, plate_y, plate_y*0.4, 0, plate_y*0.4]
    fig.add_shape(type="path",
        path=f"M {px_pts[0]},{py_pts[0]} L {px_pts[1]},{py_pts[1]} L {px_pts[2]},{py_pts[2]} L {px_pts[3]},{py_pts[3]} L {px_pts[4]},{py_pts[4]} Z",
        fillcolor="rgba(255,255,255,0.15)",
        line=dict(color="#aaaaaa", width=1),
    )

    # 球種ごとにトレース追加
    plotted = 0
    for pt, color in PITCH_COLORS.items():
        sub = pitch_df[pitch_df["pitch_type"] == pt].copy()
        if sub.empty:
            continue
        sub = sub.dropna(subset=["plate_x", "plate_z"])
        if sub.empty:
            continue

        label_jp = PITCH_LABELS.get(pt, pt)
        hover_texts = []
        for _, r in sub.iterrows():
            velo = f"{r['release_speed']:.1f} mph" if pd.notna(r.get("release_speed")) else "---"
            desc = DESC_LABELS.get(r.get("description", ""), r.get("description", ""))
            ev = r.get("events", "")
            ev_str = f" ({ev})" if pd.notna(ev) and ev else ""
            dt = r["game_date"].strftime("%m/%d") if pd.notna(r.get("game_date")) else ""
            hover_texts.append(
                f"<b>{pt} {label_jp}</b><br>"
                f"球速: {velo}<br>"
                f"結果: {desc}{ev_str}<br>"
                f"日付: {dt}"
            )

        symbols = [DESC_SYMBOLS.get(d, "circle") for d in sub.get("description", [])]

        fig.add_trace(go.Scatter(
            x=sub["plate_x"],
            y=sub["plate_z"],
            mode="markers",
            name=f"{pt} {label_jp}",
            marker=dict(
                color=color,
                size=10,
                symbol=symbols,
                line=dict(color="#000000", width=0.5),
                opacity=0.85,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
        plotted += len(sub)

    # レイアウト
    # 大谷は左打者なので: 右側（+x）が内角、左側（-x）が外角（投手視点から見た場合）
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#cccccc"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,26,42,0.8)",
        xaxis=dict(
            range=[-1.8, 1.8],
            tickfont=dict(color="#888"),
            gridcolor="#1a2535",
            zeroline=False,
            title=dict(text="← 外角　　　　内角 →", font=dict(color="#888", size=11)),
        ),
        yaxis=dict(
            range=[0, 5],
            tickfont=dict(color="#888"),
            gridcolor="#1a2535",
            zeroline=False,
            title=dict(text="高さ (ft)", font=dict(color="#888", size=11)),
        ),
        legend=dict(
            font=dict(color="#ccc", size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
        margin=dict(l=10, r=10, t=60, b=40),
        height=430,
        hoverlabel=dict(bgcolor="#1a2535", font_color="#ffffff"),
    )

    return fig

# ─── メイン UI ─────────────────────────────────────────────
st.markdown("## ⚾ 大谷翔平 対戦予想")
st.caption("Shohei Ohtani Matchup Predictor")

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
selected_date = game_dates[selected_idx]
selected_game = future_sched.iloc[selected_idx]

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

# ══════════════════════════════════════════════════════════
# ① 大谷（打者）予測
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🥊 大谷（打者）予測</div>', unsafe_allow_html=True)

batter_out = output.get("ohtani_batter", {})
batter_pred = batter_out.get("prediction") or {}
method = batter_out.get("method", "")
note = batter_out.get("note", "")

method_label = {
    "direct_history": "✅ 直接対面実績",
    "similarity_weighted": "🔍 類似投手推定",
    "insufficient_data": "⚠ データ不足",
}.get(method, method)

st.markdown(f'<span class="method-badge">{method_label} — {note}</span>', unsafe_allow_html=True)

if method == "insufficient_data" or not batter_pred:
    st.markdown('<div class="warn-box">相手投手データが不足しています。collect_opponents.py を実行後、再読み込みしてください。</div>', unsafe_allow_html=True)
else:
    woba_val = batter_pred.get("woba")
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("wOBA", fmt(woba_val), help=METRIC_HELP["wOBA"])
    with col2:
        st.metric("xwOBA", fmt(xwoba_val), help=METRIC_HELP["xwOBA"])
    with col3:
        st.metric("xBA", fmt(batter_pred.get("xba")), help=METRIC_HELP["xBA"])

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("K率", pct(batter_pred.get("k_rate")), help=METRIC_HELP["K率"])
    with col5:
        st.metric("BB率", pct(batter_pred.get("bb_rate")), help=METRIC_HELP["BB率"])
    with col6:
        st.metric("空振り率", pct(batter_pred.get("whiff_rate")), help=METRIC_HELP["空振り率"])

    if method == "similarity_weighted":
        n_sim = batter_pred.get("n_similar", 0)
        with st.expander(f"参照した類似投手 ({n_sim} 人)"):
            st.caption(f"{batter_out.get('opponent_pitcher','')} の球種特性に近い、大谷が過去に対戦した投手をもとに算出")
            sim_ids = batter_pred.get("similar_pitcher_ids", [])
            if sim_ids:
                st.code(", ".join(str(x) for x in sim_ids[:10]))

st.divider()

# ══════════════════════════════════════════════════════════
# ② 球コース図
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 大谷（打者）球コース図</div>', unsafe_allow_html=True)

ohtani_batter_df = load_ohtani_batter_cached()
opp_pitcher_id = output.get("ohtani_batter", {}).get("opponent_pitcher_id")
pitcher_name = output.get("ohtani_batter", {}).get("opponent_pitcher", "")

if not ohtani_batter_df.empty:
    if opp_pitcher_id:
        direct_df = ohtani_batter_df[ohtani_batter_df["pitcher"] == int(opp_pitcher_id)]
    else:
        direct_df = pd.DataFrame()

    has_direct = len(direct_df) >= 5

    # タブ: 直接対面 / 直近5試合
    tabs_labels = []
    if has_direct:
        tabs_labels.append(f"vs {pitcher_name}（全{len(direct_df)}球）")
    tabs_labels.append("直近5試合")
    tabs = st.tabs(tabs_labels)

    tab_idx = 0
    if has_direct:
        with tabs[tab_idx]:
            fig = build_pitch_zone_fig(direct_df, f"大谷 vs {pitcher_name} — 全対戦球")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("○ ボール　× 見逃し　× 空振り（×open）　▲ ファウル　★ インプレー")
        tab_idx += 1

    with tabs[tab_idx]:
        recent_dates = sorted(ohtani_batter_df["game_date"].dt.date.unique())[-5:]
        recent_df = ohtani_batter_df[ohtani_batter_df["game_date"].dt.date.isin(recent_dates)]
        fig2 = build_pitch_zone_fig(recent_df, f"大谷 直近5試合 — {len(recent_df)}球")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("○ ボール　× 見逃し　×open 空振り　▲ ファウル　★ インプレー")
else:
    st.info("打者データが見つかりません。collect_statcast.py を実行してください。")

st.divider()

# ══════════════════════════════════════════════════════════
# ③ 大谷（投手）予測
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚾ 大谷（投手）予測</div>', unsafe_allow_html=True)

pitcher_out = output.get("ohtani_pitcher", {})
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

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("被 xwOBA", fmt(team_xwoba), help=METRIC_HELP["xwOBA"] + "（低いほど大谷に有利）")
    with col_b:
        st.metric("空振り率", pct(team_whiff), help=METRIC_HELP["空振り率"] + "（高いほど大谷に有利）")

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
