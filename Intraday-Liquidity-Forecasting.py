import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime

st.set_page_config(page_title="Intraday Liquidity Forecasting", layout="wide")

# =========================================================
# Helpers
# =========================================================
def ensure_df_has_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; compute fallbacks if missing."""
    need = {"timestamp", "inflow", "outflow", "net_flow", "balance"}
    have = set(df.columns)
    df = df.copy()

    # make net_flow if missing
    if "net_flow" not in have and {"inflow", "outflow"}.issubset(have):
        df["net_flow"] = df["inflow"] - df["outflow"]

    # synthesize balance if missing
    if "balance" not in df.columns and "net_flow" in df.columns:
        alpha = 0.2
        bal0 = 5_000_000.0
        df = df.sort_values("timestamp")
        df["balance"] = bal0 + (df["net_flow"] * alpha).cumsum()

    # Types & order
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    cols = [c for c in ["timestamp","inflow","outflow","net_flow","balance","channel","business_line"] if c in df.columns]
    return df[cols].sort_values("timestamp").reset_index(drop=True)

def generate_synthetic_csv(csv_path: str, periods=1000, freq_min=15):
    """Create a realistic synthetic dataset compatible with this app."""
    start_time = datetime(2024, 1, 2, 9, 0)
    timestamps = pd.date_range(start_time, periods=periods, freq=f"{freq_min}min")

    rng = np.random.default_rng(42)
    inflow  = np.maximum(0, rng.normal(1_000_000, 250_000, size=periods))
    outflow = np.maximum(0, rng.normal(  950_000, 240_000, size=periods))
    net     = inflow - outflow

    balance = 5_000_000 + np.cumsum(net * 0.2 + rng.normal(0, 25_000, size=periods))
    channels = rng.choice(["FEDWIRE","ACH","INTERNAL"], size=periods)
    lob      = rng.choice(["Markets","Treasury","Retail"], size=periods)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "inflow": np.round(inflow, 2),
        "outflow": np.round(outflow, 2),
        "net_flow": np.round(net, 2),
        "balance": np.round(balance, 2),
        "channel": channels,
        "business_line": lob
    })
    df.to_csv(csv_path, index=False)
    return df

def baseline_forecast(df: pd.DataFrame, freq_minutes: int, horizon_steps: int) -> pd.DataFrame:
    """
    Fast seasonal baseline: median net_flow per intraday bucket (by minute-of-day),
    then use that profile to forecast forward horizon.
    """
    s = (df.set_index("timestamp")["net_flow"]
           .asfreq(f"{freq_minutes}min")
           .fillna(0.0))

    tmp = s.to_frame("net_flow")
    tmp["bucket"] = ((tmp.index.hour * 60 + tmp.index.minute) // freq_minutes).astype(int)
    profile = tmp.groupby("bucket")["net_flow"].median()

    future_idx = pd.date_range(
        s.index[-1] + pd.Timedelta(minutes=freq_minutes),
        periods=horizon_steps,
        freq=f"{freq_minutes}min"
    )
    buckets = ((future_idx.hour * 60 + future_idx.minute) // freq_minutes).astype(int)
    yhat = profile.reindex(buckets).to_numpy()

    fc = pd.DataFrame({
        "timestamp": future_idx,
        "netflow_forecast": yhat
    })
    fc["yhat_lower"] = np.nan
    fc["yhat_upper"] = np.nan
    return fc

def apply_stress_to_forecast(forecast_df: pd.DataFrame, inflow_pct: int, outflow_pct: int, pause_wires: bool) -> pd.DataFrame:
    """Apply what-if shocks to baseline forecast."""
    inflow_mult  = 1.0 + (inflow_pct  / 100.0)
    outflow_mult = 1.0 + (outflow_pct / 100.0)
    extra_cut = 0.8 if pause_wires else 1.0

    def _adj(nf):
        if nf >= 0:
            return nf * inflow_mult
        adj = nf * outflow_mult
        if pause_wires:
            adj = adj * extra_cut  # reduce magnitude of negative outflows
        return adj

    out = forecast_df.copy()
    out["netflow_stressed"] = out["netflow_forecast"].apply(_adj)
    return out

def project_balance(start_balance: float, netflows: np.ndarray, pass_through_pct: int) -> np.ndarray:
    """Project balance applying pass-through percent of netflows each step."""
    alpha = pass_through_pct / 100.0
    b = start_balance
    path = []
    for nf in netflows:
        b = b + nf * alpha
        path.append(b)
    return np.array(path)

def find_first_breach(balances: np.ndarray, threshold: float):
    for i, v in enumerate(balances):
        if v < threshold:
            return i
    return None

def fig_netflow(df_hist: pd.DataFrame, forecast_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["timestamp"], y=df_hist["net_flow"],
        name="Historical NetFlow", line=dict(color="steelblue")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["netflow_forecast"],
        name="Baseline Forecast", line=dict(color="orange", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["netflow_stressed"],
        name="Stressed Forecast", line=dict(color="crimson", dash="dash")
    ))
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    return fig

def fig_balance(df_hist: pd.DataFrame, forecast_df: pd.DataFrame, breach_threshold: float, breach_idx: int | None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["timestamp"], y=df_hist["balance"],
        name="Historical Balance", line=dict(color="seagreen")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["balance_projected"],
        name="Projected Balance (Scenario)", line=dict(color="purple")
    ))
    fig.add_hline(y=breach_threshold, line=dict(color="red", dash="dot"),
                  annotation_text="Breach Threshold", annotation_position="top left")
    if breach_idx is not None:
        fig.add_trace(go.Scatter(
            x=[forecast_df["timestamp"].iloc[breach_idx]],
            y=[forecast_df["balance_projected"].iloc[breach_idx]],
            mode="markers", marker=dict(color="red", size=10),
            name="First Breach"
        ))
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    return fig

# =========================================================
# UI — Sidebar
# =========================================================
st.title("🏦 Intraday Liquidity Forecasting — What-If Stress Simulator")

with st.sidebar:
    st.header("📁 Data")
    data_path = st.text_input("CSV path", "intraday_liquidity_data.csv")
    freq_minutes = st.number_input("Data frequency (minutes)", value=15, min_value=1, step=1)

# Load data (or auto-generate if file missing)
if not os.path.exists(data_path):
    st.warning(f"CSV not found at '{data_path}'. Generating a synthetic file now…")
    df = generate_synthetic_csv(data_path, periods=1000, freq_min=freq_minutes)
else:
    try:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

df = ensure_df_has_required_cols(df)

# =========================================================
# Forecast configuration
# =========================================================
st.subheader("🔮 Forecast setup")
with st.sidebar:
    st.header("📈 Forecast horizon")
    horizon_steps = st.slider("Future steps", min_value=16, max_value=96, value=48, step=8)
    pass_through = st.slider("Settlement pass-through to balance (%)", 0, 100, 20, step=5)

# Baseline forecast (fast, no heavy deps)
forecast = baseline_forecast(df, freq_minutes=freq_minutes, horizon_steps=horizon_steps)

# =========================================================
# What-if stress
# =========================================================
st.subheader("🧪 What-If Liquidity Stress")
with st.expander("Open stress panel", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        inflow_pct = st.slider("Change in inflows (%)", -80, 80, 0, step=5)
    with colB:
        outflow_pct = st.slider("Change in outflows (%)", -80, 80, 0, step=5)
    with colC:
        pause_wires = st.checkbox("Pause non-critical wires (extra 20% cut on outflows)")

    colD, colE, colF = st.columns(3)
    with colD:
        starting_balance_override = st.number_input(
            "Starting balance (USD)", value=float(df["balance"].iloc[-1]), step=10000.0, format="%.2f"
        )
    with colE:
        breach_threshold = st.number_input("Breach threshold (USD)", value=500_000.0, step=50_000.0, format="%.2f")
    with colF:
        commentary_on = st.checkbox("Generate commentary", value=True)

# Apply stress to forecast
forecast = apply_stress_to_forecast(forecast, inflow_pct, outflow_pct, pause_wires)

# Project balance under scenario
balance_path = project_balance(
    start_balance=starting_balance_override,
    netflows=forecast["netflow_stressed"].to_numpy(),
    pass_through_pct=pass_through
)
forecast["balance_projected"] = balance_path

# Find first breach
breach_idx = find_first_breach(forecast["balance_projected"].to_numpy(), breach_threshold)

# =========================================================
# Charts
# =========================================================
st.subheader("📊 Historical vs Forecast Net Flow")
st.plotly_chart(fig_netflow(df, forecast), use_container_width=True)

st.subheader("💧 Balance Path: Historical vs Projected (Scenario)")
st.plotly_chart(fig_balance(df, forecast, breach_threshold, breach_idx), use_container_width=True)

# =========================================================
# KPIs & Commentary
# =========================================================
st.subheader("📌 Scenario KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Start Balance", f"${starting_balance_override:,.0f}")
c2.metric("Min Projected Balance", f"${forecast['balance_projected'].min():,.0f}")
c3.metric("Pass-through %", f"{pass_through}%")
if breach_idx is None:
    c4.metric("Breach", "No breach")
else:
    c4.metric("Breach Time", f"{forecast['timestamp'].iloc[breach_idx].strftime('%Y-%m-%d %H:%M')}")

if commentary_on:
    st.subheader("🗒️ Auto-Commentary")
    min_bal = forecast['balance_projected'].min()
    end_bal = forecast['balance_projected'].iloc[-1]
    drift = end_bal - starting_balance_override
    direction = "increase" if drift >= 0 else "decrease"
    shock_text = []
    if inflow_pct != 0:
        shock_text.append(f"inflows {inflow_pct:+d}%")
    if outflow_pct != 0:
        shock_text.append(f"outflows {outflow_pct:+d}%")
    if pause_wires:
        shock_text.append("paused non-critical wires")
    shock_clause = (" with " + ", ".join(shock_text)) if shock_text else ""
    breach_clause = (
        f" A breach is projected at {forecast['timestamp'].iloc[breach_idx].strftime('%Y-%m-%d %H:%M')} below ${breach_threshold:,.0f}."
        if breach_idx is not None else " No breach is projected against the set threshold."
    )
    st.write(
        f"Under the stressed scenario{shock_clause}, projected balances {direction} by "
        f"${abs(drift):,.0f} over the horizon (min ${min_bal:,.0f}).{breach_clause} "
        f"Assuming {pass_through}% settlement pass-through per {freq_minutes}-minute interval."
    )

# =========================================================
# IDL Playbook Simulation
# =========================================================
st.subheader("🧭 IDL Playbook — Breach Response Simulation")
with st.expander("Open Playbook simulation", expanded=True):
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        intercompany_amount = st.number_input(
            "Intercompany funding (USD)", value=2_000_000.0, step=100_000.0, format="%.2f"
        )
    with colp2:
        intraday_repo_amount = st.number_input(
            "Intraday repo (USD)", value=3_000_000.0, step=100_000.0, format="%.2f"
        )
    with colp3:
        notify_minutes = st.slider("Notify stakeholders within (mins)", 1, 30, 5, step=1)

    run_playbook = st.button("Run Playbook Simulation")
    playbook_log = None

    if run_playbook:
        # Choose time: actual breach or first forecast point (drill)
        if breach_idx is None:
            breach_time = forecast["timestamp"].iloc[0]
            breach_bal = forecast["balance_projected"].iloc[0]
            breach_note = "No actual breach — running drill at forecast start."
        else:
            breach_time = forecast["timestamp"].iloc[breach_idx]
            breach_bal = forecast["balance_projected"].iloc[breach_idx]
            breach_note = f"Breach detected below ${breach_threshold:,.0f}."

        t0 = breach_time
        steps = [
            {"time": t0, "action": "Detect breach & open incident", "owner": "IDL Manager", "detail": breach_note},
            {"time": t0 + pd.Timedelta(minutes=notify_minutes), "action": "Notify stakeholders", "owner": "IDL Manager",
             "detail": f"Notify within {notify_minutes} min; share snapshot & forecast."},
            {"time": t0 + pd.Timedelta(minutes=notify_minutes + 5), "action": "Pause non-critical wires", "owner": "Ops",
             "detail": "Throttle discretionary outflows pending funding."},
            {"time": t0 + pd.Timedelta(minutes=notify_minutes + 10), "action": "Execute intercompany funding",
             "owner": "Treasury Funding", "detail": f"Book +${intercompany_amount:,.0f} cash movement."},
            {"time": t0 + pd.Timedelta(minutes=notify_minutes + 25), "action": "Execute intraday repo (backup)",
             "owner": "Treasury Funding", "detail": f"Raise +${intraday_repo_amount:,.0f} if balance still < threshold."},
            {"time": t0 + pd.Timedelta(minutes=notify_minutes + 45), "action": "Post-mortem & resiliency record",
             "owner": "IDL Manager", "detail": "Document breach, timings, actions, lessons learned."},
        ]
        playbook_log = pd.DataFrame(steps)

        # Apply funding impacts
        scenario_path = forecast[["timestamp","balance_projected"]].copy()
        hit_ic_time   = steps[3]["time"] + pd.Timedelta(minutes=2)
        hit_repo_time = steps[4]["time"] + pd.Timedelta(minutes=2)

        scenario_path["balance_projected_playbook"] = scenario_path["balance_projected"]

        # Intercompany hit
        scenario_path.loc[scenario_path["timestamp"] >= hit_ic_time, "balance_projected_playbook"] += intercompany_amount

        # Repo hit (apply if still below threshold at hit time after intercompany)
        # Evaluate balance at/after intercompany hit
        after_ic = scenario_path.loc[scenario_path["timestamp"] >= hit_ic_time, "balance_projected_playbook"]
        if not after_ic.empty and (after_ic.iloc[0] < breach_threshold):
            scenario_path.loc[scenario_path["timestamp"] >= hit_repo_time, "balance_projected_playbook"] += intraday_repo_amount

        st.markdown("**Projected balance with Playbook actions**")
        fig_play = go.Figure()
        fig_play.add_trace(go.Scatter(
            x=df["timestamp"], y=df["balance"], name="Historical Balance", line=dict(color="seagreen")
        ))
        fig_play.add_trace(go.Scatter(
            x=forecast["timestamp"], y=forecast["balance_projected"],
            name="Projected (No Actions)", line=dict(color="purple", dash="dot")
        ))
        fig_play.add_trace(go.Scatter(
            x=scenario_path["timestamp"], y=scenario_path["balance_projected_playbook"],
            name="Projected (Playbook Applied)", line=dict(color="dodgerblue")
        ))
        fig_play.add_hline(y=breach_threshold, line=dict(color="red", dash="dot"),
                           annotation_text="Breach Threshold", annotation_position="top left")
        st.plotly_chart(fig_play, use_container_width=True)

        st.markdown("**Escalation Timeline**")
        st.dataframe(playbook_log, use_container_width=True)

        st.download_button(
            "Download Incident Log (CSV)",
            playbook_log.to_csv(index=False).encode("utf-8"),
            file_name="idl_playbook_incident_log.csv",
            mime="text/csv"
        )

# =========================================================
# Export results
# =========================================================
st.subheader("⬇️ Download Scenario Results")
export_df = forecast.copy()
export_df["pass_through_pct"] = pass_through
export_df["start_balance"] = float(starting_balance_override)
export_df["breach_threshold"] = float(breach_threshold)
st.download_button(
    "Download Forecast CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="liquidity_scenario_results.csv",
    mime="text/csv"
)

st.sidebar.download_button(
    "Download Historical Data (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="intraday_liquidity_data_out.csv",
    mime="text/csv"
)
