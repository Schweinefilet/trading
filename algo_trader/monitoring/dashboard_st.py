import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
STATE_FILE = LOGS_DIR / "state.json"
TELEMETRY_FILE = LOGS_DIR / "telemetry.log"
ALERTS_FILE = LOGS_DIR / "alerts.log"

st.set_page_config(page_title="AlgoTrader Dashboard", layout="wide")

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

def load_telemetry():
    if TELEMETRY_FILE.exists():
        data = []
        with open(TELEMETRY_FILE, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    return pd.DataFrame()

def load_alerts():
    if ALERTS_FILE.exists():
        data = []
        with open(ALERTS_FILE, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("AlgoTrader v1.0")
state = load_state()

if state:
    last_update = datetime.fromisoformat(state["last_update"])
    st.sidebar.write(f"Last Update: {last_update.strftime('%H:%M:%S')}")
    
    st.sidebar.divider()
    m = state["metrics"]
    st.sidebar.metric("Equity", f"${m['equity']:,.2f}", f"{m['daily_pnl']:+,.2f}")
    st.sidebar.metric("Regime", m["regime"])
    st.sidebar.metric("Heat", f"{m['heat_pct']:.1f}%")
    st.sidebar.write(f"Connected: {'🟢' if m['connected'] else '🔴'}")

# --- Main Dashboard ---
st.title("AlgoTrader Live Performance")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Equity Curve (Daily)")
    telemetry_df = load_telemetry()
    if not telemetry_df.empty:
        telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['equity'], mode='lines', name='Equity'))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No telemetry data yet.")

with col2:
    st.subheader("Current Positions")
    if state and state["positions"]:
        pos_df = pd.DataFrame.from_dict(state["positions"], orient='index')
        st.table(pos_df)
    else:
        st.info("No open positions.")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Recent Alerts")
    alerts_df = load_alerts()
    if not alerts_df.empty:
        # Show last 10
        recent_alerts = alerts_df.tail(10).iloc[::-1]
        for _, row in recent_alerts.iterrows():
            color = "red" if row['level'] == "CRITICAL" else "orange" if row['level'] == "WARNING" else "green"
            st.markdown(f"**{row['timestamp'][11:19]}** :{color}[{row['level']}] **{row['title']}**: {row['message']}")
    else:
        st.info("No alerts logged.")

with col4:
    st.subheader("System Stats")
    if state:
        stats = {
            "Day Trades Remaining": state["pdt"]["day_trades_remaining"],
            "Trades Today": state["metrics"]["trades_today"],
            "Portfolio Heat ($)": f"${state['metrics']['heat_pct'] * state['metrics']['equity'] / 100:,.0f}"
        }
        st.json(stats)

# Auto-refresh
if st.button("Refresh Now"):
    st.rerun()

st.caption("Auto-refresh: This dashboard does not auto-refresh. Please manually refresh or use streamlit run with --server.headless delay.")
