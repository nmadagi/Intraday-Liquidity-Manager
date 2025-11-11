import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ------------------------------
# Synthetic Intraday Liquidity Data Generator
# ------------------------------

# Simulation parameters
start_time = datetime(2024, 1, 2, 9, 0)  # start at 9 AM
periods = 1000                            # 1000 intervals (~10 business days)
interval = timedelta(minutes=15)          # 15-minute frequency

channels = ['FEDWIRE', 'ACH', 'INTERNAL']
business_lines = ['Markets', 'Treasury', 'Retail']

data = []
balance = 5_000_000  # initial central bank balance in USD

for i in range(periods):
    timestamp = start_time + i * interval
    
    # Random inflows/outflows (simulate payment activity)
    inflow = max(0, np.random.normal(1_000_000, 250_000))
    outflow = max(0, np.random.normal(950_000, 240_000))
    net_flow = inflow - outflow
    
    # Update balance with small drift + random noise
    balance += net_flow * 0.2 + np.random.normal(0, 25_000)
    
    data.append({
        "timestamp": timestamp,
        "inflow": round(inflow, 2),
        "outflow": round(outflow, 2),
        "net_flow": round(net_flow, 2),
        "balance": round(balance, 2),
        "channel": random.choice(channels),
        "business_line": random.choice(business_lines)
    })

df = pd.DataFrame(data)
df.to_csv("intraday_liquidity_data.csv", index=False)

print("✅ intraday_liquidity_data.csv generated successfully!")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("Preview:")
print(df.head())
