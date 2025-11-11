import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

SIMPLE_MODE = True


st.set_page_config(page_title="Intraday Liquidity Forecasting", layout="wide")

# -----------------------------
# 1) LOAD DATA (LOCAL CSV ONLY)
# -----------------------------
st.title("🏦 Intraday Liquidity Forecasting — What-if Stress Simulator")
st.markdown(
    "This dashboard loads **local intraday liquidity data** and provides **AI forecasts** "
    "with a **What-if stress panel** to simulate Treasury actions and shocks."
)

with st.sidebar:
    st.header("📁 Data")
    data_path = st.text_input("CSV path", "intraday_liquidity_data.csv")
    freq_minutes = st.number_input("Data frequency (minutes)", value=15, min_value=1, step=1)

try:
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
except Exception as e:
    st.error(f"Could not read CSV at `{data_path}`. Error: {e}")
    st.stop()

# Basic sanity checks / repairs
required_cols = {"timestamp", "inflow", "outflow", "net_flow", "balance"}
missing = required_cols - set(df.columns)
if missing:
    st.warning(f"Missing columns {missing}. Attempting to infer `net_flow` if possible.")
    if "inflow" in df.columns and "outflow" in df.columns and "net_flow" not in df.columns:
        df["net_flow"] = df["inflow"] - df["outflow"]
    if "balance" not in df.columns:
        # Make a synthetic balance if not present
        alpha = 0.2
        bal0 = 5_000_000.0
        df = df.sort_values("timestamp")
        df["balance"] = bal0 + (df["net_flow"] * alpha).cumsum()

df = df.sort_values("timestamp").reset_index(drop=True)


# -----------------------------
# 2) FORECAST: Seasonal profile baseline (fast, no heavy deps)
# -----------------------------
st.subheader("🔮 Forecast setup")

steps_per_day = max(1, int(round(24 * 60 / freq_minutes)))
y_series = (df.set_index("timestamp")["net_flow"]
              .asfreq(f"{freq_minutes}min")
              .fillna(0.0))

# Build a median intraday profile by time bucket
tmp = y_series.to_frame("net_flow")
tmp["bucket"] = ((tmp.index.hour * 60 + tmp.index.minute) // freq_minutes).astype(int)
profile = tmp.groupby("bucket")["net_flow"].median()

future_idx = pd.date_range(
    y_series.index[-1] + pd.Timedelta(minutes=freq_minutes),
    periods=horizon_steps,
    freq=f"{freq_minutes}min"
)
buckets = ((future_idx.hour * 60 + future_idx.minute) // freq_minutes).astype(int)

forecast = pd.DataFrame({
    "timestamp": future_idx,
    "netflow_forecast": profile.reindex(buckets).to_numpy()
})
forecast["yhat_lower"] = np.nan
forecast["yhat_upper"] = np.nan


# ----------------------------------------------------
# 3) WHAT-IF STRESS PANEL (APPLIED TO FORECAST PERIOD)
# ----------------------------------------------------
st.subheader("🧪 What-if Liquidity Stress Simulation")
with st.expander("Open What-if panel", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        inflow_pct = st.slider("Change in inflows (%)", min_value=-80, max_value=80, value=0, step=5,
                               help="Applies to periods with positive net flow (more inflow than outflow).")
    with colB:
        outflow_pct = st.slider("Change in outflows (%)", min_value=-80, max_value=80, value=0, step=5,
                                help="Applies to periods with negative net flow (more outflow than inflow).")
    with colC:
        pause_wires = st.checkbox("Pause non-critical wires (extra 20% outflow cut when net outflow)")

    colD, colE, colF = st.columns(3)
    with colD:
        starting_balance_override = st.number_input(
            "Starting balance override (USD, optional)", value=float(df["balance"].iloc[-1]), step=10000.0, format="%.2f"
        )
    with colE:
        breach_threshold = st.number_input("Breach threshold (USD)", value=500_000.0, step=50_000.0, format="%.2f")
    with colF:
        commentary_on = st.checkbox("Generate auto-commentary", value=True)

# Build stressed forecast of net flows
inflow_mult = 1.0 + (inflow_pct / 100.0)
outflow_mult = 1.0 + (outflow_pct / 100.0)
extra_outflow_cut = 0.8 if pause_wires else 1.0  # if pause wires, we cut outflows by extra 20% -> multiply magnitude by 0.8

def apply_stress(row):
    nf = row["netflow_forecast"]
    # Positive net flow -> inflow-dominant
    if nf >= 0:
        return nf * inflow_mult
    # Negative net flow -> outflow-dominant
    adj = nf * outflow_mult
    if pause_wires:
        adj = adj * extra_outflow_cut  # reduces magnitude (less negative)
    return adj

forecast["netflow_stressed"] = forecast.apply(apply_stress, axis=1)

# -------------------------------------------
# 4) PROJECT BALANCE PATH UNDER THE SCENARIO
# -------------------------------------------
alpha = pass_through / 100.0  # % of net flow that actually hits central bank balance per step
start_balance = float(starting_balance_override)

proj_balance = []
b = start_balance
for nf in forecast["netflow_stressed"].values:
    b = b + (nf * alpha)
    proj_balance.append(b)

forecast["balance_projected"] = proj_balance

# Find breach (first time balance falls below threshold)
breach_idx = None
for i, val in enumerate(forecast["balance_projected"].values):
    if val < breach_threshold:
        breach_idx = i
        break

# -----------------------------
# 5) VISUALS: FLOWS & BALANCE
# -----------------------------
st.subheader("📊 Historical vs Forecast Net Flow")

fig_nf = go.Figure()
fig_nf.add_trace(go.Scatter(
    x=df["timestamp"], y=df["net_flow"], name="Historical NetFlow", line=dict(color="steelblue")
))
fig_nf.add_trace(go.Scatter(
    x=forecast["timestamp"], y=forecast["netflow_forecast"], name="Baseline Forecast", line=dict(dash="dot", color="orange")
))
fig_nf.add_trace(go.Scatter(
    x=forecast["timestamp"], y=forecast["netflow_stressed"], name="Stressed Forecast", line=dict(dash="dash", color="crimson")
))
st.plotly_chart(fig_nf, use_container_width=True)

st.subheader("💧 Balance Path: Historical vs Projected (Scenario)")
fig_bal = go.Figure()
fig_bal.add_trace(go.Scatter(
    x=df["timestamp"], y=df["balance"], name="Historical Balance", line=dict(color="seagreen")
))
fig_bal.add_trace(go.Scatter(
    x=forecast["timestamp"], y=forecast["balance_projected"], name="Projected Balance (Scenario)",
    line=dict(color="purple")
))
# Draw breach line
fig_bal.add_hline(y=breach_threshold, line=dict(color="red", dash="dot"), annotation_text="Breach Threshold", annotation_position="top left")
# Mark breach point
if breach_idx is not None:
    fig_bal.add_trace(go.Scatter(
        x=[forecast["timestamp"].iloc[breach_idx]],
        y=[forecast["balance_projected"].iloc[breach_idx]],
        mode="markers",
        marker=dict(color="red", size=10),
        name="First Breach"
    ))
st.plotly_chart(fig_bal, use_container_width=True)

# -----------------------------
# 6) METRICS & COMMENTARY
# -----------------------------
st.subheader("📌 Scenario KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Start Balance", f"${start_balance:,.0f}")
c2.metric("Min Projected Balance", f"${forecast['balance_projected'].min():,.0f}")
c3.metric("Pass-through %", f"{pass_through}%")
if breach_idx is None:
    c4.metric("Breach", "No breach")
else:
    c4.metric("Breach Time", f"{forecast['timestamp'].iloc[breach_idx].strftime('%Y-%m-%d %H:%M')}")

if commentary_on:
    st.subheader("🗒️ Auto-Commentary")
    # Simple narrative generator
    min_bal = forecast['balance_projected'].min()
    end_bal = forecast['balance_projected'].iloc[-1]
    drift = end_bal - start_balance
    direction = "increase" if drift >= 0 else "decrease"
    shock_text = []
    if inflow_pct != 0:
        shock_text.append(f"inflows {inflow_pct:+d}%")
    if outflow_pct != 0:
        shock_text.append(f"outflows {outflow_pct:+d}%")
    if pause_wires:
        shock_text.append("paused non-critical wires")

    shock_clause = " with " + ", ".join(shock_text) if shock_text else ""
    breach_clause = (
        f" A breach is projected at {forecast['timestamp'].iloc[breach_idx].strftime('%Y-%m-%d %H:%M')} below ${breach_threshold:,.0f}."
        if breach_idx is not None else " No breach is projected against the set threshold."
    )
    st.write(
        f"Under the stressed scenario{shock_clause}, projected balances {direction} by "
        f"${abs(drift):,.0f} over the horizon (min ${min_bal:,.0f}).{breach_clause} "
        f"Assuming {pass_through}% settlement pass-through per {freq_minutes}-minute interval."
    )
# -----------------------------
# 6.1) IDL PLAYBOOK SIMULATION
# -----------------------------
st.subheader("🧭 IDL Playbook — Breach Response Simulation")

with st.expander("Open Playbook simulation", expanded=True):
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        intercompany_amount = st.number_input(
            "Intercompany funding (USD)", value=2_000_000.0, step=100_000.0, format="%.2f",
            help="Size of immediate intercompany loan to restore balance"
        )
    with colp2:
        intraday_repo_amount = st.number_input(
            "Intraday repo (USD)", value=3_000_000.0, step=100_000.0, format="%.2f",
            help="Emergency repo line to deploy if breach persists"
        )
    with colp3:
        notify_minutes = st.slider(
            "Notify stakeholders within (mins)", min_value=1, max_value=30, value=5, step=1
        )

    run_playbook = st.button("Run Playbook Simulation")

    playbook_log = None
    if run_playbook:
        # If no breach, we simulate a precautionary drill at first future timestamp
        if breach_idx is None:
            breach_time = forecast["timestamp"].iloc[0]
            breach_bal = forecast["balance_projected"].iloc[0]
            breach_note = "No actual breach — running drill at forecast start."
        else:
            breach_time = forecast["timestamp"].iloc[breach_idx]
            breach_bal = forecast["balance_projected"].iloc[breach_idx]
            breach_note = f"Breach detected below ${breach_threshold:,.0f}."

        # Build escalation timeline (relative to breach_time)
        t0 = breach_time
        steps = [
            {
                "time": t0,
                "action": "Detect breach & open incident",
                "owner": "IDL Manager",
                "detail": breach_note
            },
            {
                "time": t0 + pd.Timedelta(minutes=notify_minutes),
                "action": "Notify stakeholders (Treasury, Ops, LOBs)",
                "owner": "IDL Manager",
                "detail": f"Notify within {notify_minutes} min; circulate snapshot & forecast."
            },
            {
                "time": t0 + pd.Timedelta(minutes=notify_minutes + 5),
                "action": "Pause non-critical wires (if enabled)",
                "owner": "Ops",
                "detail": "Throttle discretionary outflows pending funding."
            },
            {
                "time": t0 + pd.Timedelta(minutes=notify_minutes + 10),
                "action": "Execute intercompany funding",
                "owner": "Treasury Funding",
                "detail": f"Book +${intercompany_amount:,.0f} cash movement."
            },
            {
                "time": t0 + pd.Timedelta(minutes=notify_minutes + 25),
                "action": "Execute intraday repo (backup)",
                "owner": "Treasury Funding",
                "detail": f"Raise +${intraday_repo_amount:,.0f} if balance still < threshold."
            },
            {
                "time": t0 + pd.Timedelta(minutes=notify_minutes + 45),
                "action": "Post-mortem and resiliency record",
                "owner": "IDL Manager",
                "detail": "Document breach, timings, actions, lessons learned."
            },
        ]
        playbook_log = pd.DataFrame(steps)

        # Apply funding effects to a copy of projected path after breach
        scenario_path = forecast[["timestamp", "balance_projected"]].copy()
        # Funding hits shortly after each scheduled step
        hit_ic_time = steps[3]["time"] + pd.Timedelta(minutes=2)
        hit_repo_time = steps[4]["time"] + pd.Timedelta(minutes=2)

        scenario_path["balance_projected_playbook"] = scenario_path["balance_projected"]
        # Intercompany hit
        scenario_path.loc[scenario_path["timestamp"] >= hit_ic_time, "balance_projected_playbook"] += intercompany_amount
        # Repo hit (only if still below threshold at hit time)
        # Evaluate balance at hit time after intercompany
        if (
            scenario_path.loc[scenario_path["timestamp"] >= hit_ic_time, "balance_projected_playbook"]
            .head(1)
            .min()
            < breach_threshold
        ):
            scenario_path.loc[scenario_path["timestamp"] >= hit_repo_time, "balance_projected_playbook"] += intraday_repo_amount

        # Plot comparison
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

        # Show timeline table
        st.markdown("**Escalation Timeline**")
        st.dataframe(playbook_log, use_container_width=True)

        # Export incident log
        st.download_button(
            "Download Incident Log (CSV)",
            playbook_log.to_csv(index=False).encode("utf-8"),
            file_name="idl_playbook_incident_log.csv",
            mime="text/csv"
        )

# -----------------------------
# 7) EXPORT RESULTS
# -----------------------------
st.subheader("⬇️ Download Scenario Results")
export_df = forecast.copy()
export_df["pass_through_pct"] = pass_through
export_df["start_balance"] = start_balance
export_df["breach_threshold"] = breach_threshold
st.download_button(
    "Download CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="liquidity_scenario_results.csv",
    mime="text/csv"
)

st.caption("Tip: In your interview, drive the discussion with the stress panel—explain how the playbook would react at the first breach time.")
