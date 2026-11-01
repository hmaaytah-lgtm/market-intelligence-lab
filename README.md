# Market Intelligence Lab V2

Arabic-first, mobile-friendly, research-only dashboard for **GLD** (gold ETF) and **SPY** (S&P 500 ETF).

## Features
- Automatically loads both markets when opened (Yahoo Finance via yfinance, one-hour cache).
- Separate five-trading-session probability forecasts and historical expanding-window evaluations.
- Compares model accuracy with majority-class baseline and reports Brier score.
- Explains wait/watch status in Arabic and suppresses conditional signals if the price feed is stale.
- Includes historical forecast tables and CSV downloads.
- Paper portfolio remains **inactive at JOD 300** pending a future leakage-safe, next-session execution backtest. No real trades, no brokerage connection, no background monitoring.

## Deploy from iPhone
1. In the existing GitHub repository, replace `app.py` and `requirements.txt` with the files in this folder; optionally replace `README.md`.
2. Commit changes to `main`.
3. Streamlit Community Cloud should redeploy the same app automatically. If needed, reboot the app from its management screen.

## Important limitations
The forecasts are uncalibrated estimates, not guarantees. ETFs differ from the spot gold price and S&P 500 index. Yahoo Finance prices may be delayed or unavailable. Historical evaluation is exploratory, not evidence of tradable profit. **No capital should be committed on the basis of these outputs.**
