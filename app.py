"""
大谷翔平 対戦予想ダッシュボード
Streamlit アプリ

ローカル起動: streamlit run app.py
Streamlit Cloud: data/ ディレクトリを GitHub に commit して deploy する
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from predict import run_prediction, _load_schedule

# ─── ページ設定 ────────────────────────────────────────────
st.set_page_config(
    page_title="大谷翔平 対戦予想",
    page_icon="⚾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS（モバイル最適化） ──────────────────────────────────
st.markdown("""
<style>
  /* メトリクスカードを大きく・見やすく */
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
  /* 球種バッジ */
  .pitch-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
  }
  /* セクション見出し */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 16px 0 8px 0;
    padding-left: 8px;
    border-left: 4px solid #0078ff;
  }
  /* 手法バッジ */
  .method-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.7rem;
    background: #243447;
    color: #8899aa;
    margin-bottom: 8px;
  }
  /* データ不足警告 */
  .warn-box {
    background: #2a2010;
    border: 1px solid #664400;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #ccaa55;
  }
</style>
""", unsafe_allow_html=True)


# ─── ヘルパー ──────────────────────────────────────────────

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

def fmt(v, digits=3, suffix=""):
    return f"{v:.{digits}f}{suffix}" if v is not None else "---"

def pct(v):
    return f"{v*100:.1f}%" if v is not None else "---"

def grade(woba):
    """wOBA から評価ラベル"""
    if woba is None: return "---", "#666"
    if woba >= 0.400: return "大当たり ◎", "#27ae60"
    if woba >= 0.350: return "好調 ○", "#2ecc71"
    if woba >= 0.300: return "普通 △", "#f39c12"
    return "苦手 ×", "#e74c3c"

def pitcher_grade(xwoba):
    """被 xwOBA から評価ラベル（低いほど良い）"""
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


# ─── メイン UI ─────────────────────────────────────────────

st.markdown("## ⚾ 大谷翔平 対戦予想")
st.caption("Shohei Ohtani Matchup Predictor")

# スケジュール読み込み
try:
    schedule = load_schedule_cached()
except FileNotFoundError as e:
    st.error(f"スケジュールファイルが見つかりません: {e}")
    st.stop()

# 日付選択（スケジュール内の日付のみ）
schedule["game_date"] = pd.to_datetime(schedule["game_date"])
today = pd.Timestamp("today").normalize()
future_sched = schedule[schedule["game_date"] >= today].sort_values("game_date")

if future_sched.empty:
    st.warning("今後の試合がスケジュールに見つかりません。collect_schedule.py を再実行してください。")
    st.stop()

game_dates = future_sched["game_date"].dt.strftime("%Y-%m-%d").tolist()
game_labels = []
for _, row in future_sched.iterrows():
    ha = "🏠" if row["home_away"] == "home" else "✈"
    ohtani_flag = " ⚡先発" if row.get("ohtani_starting_pitcher") else ""
    label = f"{row['game_date'].strftime('%m/%d')} {ha} vs {row['opponent_team']}{ohtani_flag}"
    game_labels.append(label)

selected_idx = st.selectbox(
    "試合を選択",
    range(len(game_dates)),
    format_func=lambda i: game_labels[i],
    index=0,
)
selected_date = game_dates[selected_idx]
selected_game = future_sched.iloc[selected_idx]

# ゲームヘッダー
ha_emoji = "🏠 HOME" if selected_game["home_away"] == "home" else "✈ AWAY"
st.markdown(f"""
### {selected_game['game_date'].strftime('%Y年%m月%d日')} &nbsp; {ha_emoji}
**vs {selected_game['opponent_team']}**
相手先発: **{selected_game['opp_starter_name']}**
""")

if selected_game.get("ohtani_starting_pitcher"):
    st.success("⚡ 大谷 本日先発登板予定")

st.divider()

# 予測実行
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
    st.markdown('<div class="warn-box">相手投手データが不足しています。<br>collect_opponents.py を実行後、再読み込みしてください。</div>', unsafe_allow_html=True)
else:
    woba_val = batter_pred.get("woba")
    xwoba_val = batter_pred.get("xwoba")
    grade_label, grade_color = grade(xwoba_val or woba_val)

    # 総合評価
    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, #1a2535, #0d1a2a);
      border-radius: 16px;
      padding: 16px 20px;
      margin: 8px 0 16px 0;
      border: 1px solid #243447;
    ">
      <div style="font-size:0.8rem;color:#8899aa;margin-bottom:4px">総合評価</div>
      <div style="font-size:2.2rem;font-weight:800;color:{grade_color}">{grade_label}</div>
      <div style="font-size:0.8rem;color:#8899aa;margin-top:4px">
        vs {batter_out.get('opponent_pitcher','---')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 主要指標
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("wOBA", fmt(woba_val))
    with col2:
        st.metric("xwOBA", fmt(xwoba_val))
    with col3:
        st.metric("xBA", fmt(batter_pred.get("xba")))

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("K率", pct(batter_pred.get("k_rate")))
    with col5:
        st.metric("BB率", pct(batter_pred.get("bb_rate")))
    with col6:
        st.metric("空振り率", pct(batter_pred.get("whiff_rate")))

    # 類似投手（similarity_weighted の場合）
    if method == "similarity_weighted":
        n_sim = batter_pred.get("n_similar", 0)
        with st.expander(f"参照した類似投手 ({n_sim} 人)"):
            st.caption("Fedde の球種特性に近い、大谷が過去に対戦した投手")
            sim_ids = batter_pred.get("similar_pitcher_ids", [])
            if sim_ids:
                st.code(", ".join(str(x) for x in sim_ids[:10]))

st.divider()

# ══════════════════════════════════════════════════════════
# ② 大谷（投手）予測
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
    <div style="
      background: linear-gradient(135deg, #1a2535, #0d1a2a);
      border-radius: 16px;
      padding: 16px 20px;
      margin: 8px 0 16px 0;
      border: 1px solid #243447;
    ">
      <div style="font-size:0.8rem;color:#8899aa;margin-bottom:4px">投球評価 vs {pitcher_out.get('opponent_team','')}</div>
      <div style="font-size:2.2rem;font-weight:800;color:{p_grade_color}">{p_grade_label}</div>
      <div style="font-size:0.8rem;color:#8899aa;margin-top:4px">
        被 xwOBA {fmt(team_xwoba)} / 空振り率 {pct(team_whiff)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 球種割合
    pitch_mix = pitcher_out.get("ohtani_pitch_mix", {})
    if pitch_mix:
        st.markdown("**大谷の球種構成（2026年）**")
        sorted_mix = sorted(pitch_mix.items(), key=lambda x: -x[1])
        badges = "".join(pitch_badge(pt, pct_v) for pt, pct_v in sorted_mix if pct_v >= 0.02)
        st.markdown(badges, unsafe_allow_html=True)

        # 横棒グラフ
        mix_df = pd.DataFrame(
            [(PITCH_LABELS.get(pt, pt), round(v * 100, 1)) for pt, v in sorted_mix if v >= 0.02],
            columns=["球種", "割合(%)"]
        ).set_index("球種")
        st.bar_chart(mix_df, use_container_width=True, height=200)

    n_batters = pitcher_out.get("n_batters_analyzed", 0)
    st.caption(f"分析対象打者: {n_batters} 人")

st.divider()

# ══════════════════════════════════════════════════════════
# フッター
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center;font-size:0.7rem;color:#445566;padding:16px 0">
  データソース: Baseball Savant (Statcast) / MLB Stats API<br>
  ※予測は統計モデルによる推定値です
</div>
""", unsafe_allow_html=True)

# 更新ボタン
if st.button("🔄 データ再読み込み", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
