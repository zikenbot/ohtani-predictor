import warnings
warnings.filterwarnings("ignore")
import sys, traceback
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

# データ読み込み
DATA_DIR = Path(__file__).parent / "data"
files = sorted((DATA_DIR / "statcast").glob("ohtani_batter_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df["game_date"] = pd.to_datetime(df["game_date"])

recent_dates = sorted(df["game_date"].dt.date.unique())[-5:]
recent_df = df[df["game_date"].dt.date.isin(recent_dates)].copy()
print(f"recent_df: {len(recent_df)} rows")

# build_pitch_zone_fig を直接テスト
try:
    from app import build_zone_fig_history, build_zone_fig_live, PITCH_COLORS, PITCH_LABELS
    fig = build_zone_fig_history(recent_df, "テスト（過去）")
    print("build_zone_fig_history OK, traces:", len(fig.data))
    fig = build_zone_fig_live(recent_df.head(20), "テスト（今試合）")
    print("build_pitch_zone_fig OK, traces:", len(fig.data))
except Exception as e:
    traceback.print_exc()
    print("ERROR:", e)

# plotly shape path テスト
try:
    fig2 = go.Figure()
    fig2.add_shape(type="path",
        path="M -0.708,0.3 L 0.708,0.3 L 0.708,0.12 L 0,0 L -0.708,0.12 Z",
        fillcolor="rgba(255,255,255,0.1)",
        line=dict(color="#aaa", width=1),
    )
    print("add_shape path OK")
except Exception as e:
    traceback.print_exc()
    print("add_shape ERROR:", e)

# marker symbol list テスト
try:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=[0, 1, 2], y=[0, 1, 2],
        mode="markers",
        marker=dict(symbol=["circle", "x", "star"], size=10),
    ))
    print("marker symbol list OK")
except Exception as e:
    traceback.print_exc()
    print("marker symbol list ERROR:", e)
