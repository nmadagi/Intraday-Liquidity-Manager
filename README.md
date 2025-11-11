# 💧 Intraday Liquidity Forecasting — What-If Stress Simulator  
**Python | Streamlit | Prophet | Plotly | Pandas | NumPy**

An interactive **Treasury & Liquidity Analytics** application that models and forecasts intraday cash-flow patterns, simulates liquidity shocks, and visualizes breach scenarios with automated commentary and incident logs.

---

## 🧭 Overview
Effective intraday liquidity management requires accurate short-term forecasting of cash inflows, outflows, and balances.  
This project demonstrates how **AI-based time-series forecasting** and **scenario simulation** can support liquidity monitoring and funding decisions in real time.

The dashboard allows users to:
- Forecast **intraday net cash flows** using Prophet time-series modeling  
- Project **central-bank or reserve account balances** over a short horizon  
- Run **What-If stress simulations** (e.g., inflow/outflow shocks, paused wires, funding pass-through changes)  
- Generate **breach detection alerts** and simulate **response playbooks**  
- Export forecasts and incident logs for reporting or resiliency documentation  

---

## ⚙️ Tech Stack
| Layer | Tools / Libraries | Purpose |
|-------|-------------------|----------|
| Data | `pandas`, `numpy` | Manage and transform payment & balance data |
| Forecasting | `prophet` | Predict intraday net-flow trends |
| Visualization | `plotly`, `matplotlib` (via Streamlit) | Interactive charts and dashboards |
| App Interface | `streamlit` | Front-end user experience |
| Simulation Logic | Pure Python | What-If shocks, breach detection, playbook actions |

---

## 🚀 Features

### 🔮 Forecasting Engine
- Uses **Prophet** to model and forecast net cash flows in 15-minute intervals.  
- Supports customizable horizon and settlement pass-through percentages.  
- Displays predicted range bands (upper/lower confidence limits).

### 🧪 What-If Stress Simulation
- Apply inflow/outflow percentage shocks.  
- Optionally “pause non-critical wires” to simulate operational throttling.  
- Set breach thresholds and evaluate potential liquidity gaps.  
- Generates automated commentary explaining the forecasted outcome.

### 🧭 Playbook Simulation
- Mimics a treasury **breach response workflow**:
  - Detect breach & open incident
  - Notify stakeholders
  - Execute intercompany funding or intraday repo
  - Log response timeline  
- Plots **before/after balance trajectories** to show the impact of actions.  
- Exports the **incident log (CSV)** for resiliency testing documentation.

### 💾 Data Flexibility
- Works with any structured CSV of timestamped inflows/outflows and balances.  
- Includes a data generator (`generate_liquidity_data.py`) to create synthetic test data.

---
