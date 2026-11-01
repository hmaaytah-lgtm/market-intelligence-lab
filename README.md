# Market Intelligence Lab V3

Arabic-first, research-only Streamlit dashboard for GLD and SPY. **No real orders, brokerage integrations, or verified trading returns.**

## Features
- On-page automatic price analysis and 5-session statistical direction estimates.
- Expanding-window historical evaluation; conservative wait/watch labels and stale-price suppression.
- Scheduled forward-prediction journal via GitHub Actions (one row per symbol per price date, no retroactive rewriting).
- CSV download, mobile-friendly interface, and optional reference paper capital.

## iPhone update — important
1. In GitHub repository `hmaaytah-lgtm/market-intelligence-lab`, upload/replace the top-level `app.py`, `engine.py`, `scanner.py`, `requirements.txt`, `README.md` from this package. Do **not** upload the ZIP as a single file.
2. To add the scheduled task, GitHub must have `.github/workflows/scan.yml` in the exact nested path. The easiest reliable approach is GitHub **Add file → Create new file**, type `.github/workflows/scan.yml` in the filename, then paste the YAML from the provided package; commit to `main`. Alternatively upload the workflow file into an existing `.github/workflows/` directory.
3. Go to repository **Actions** → **Daily research signals** → **Run workflow** to test. Actions may need enabling. Check repository Settings → Actions → General → Workflow permissions → **Read and write permissions** if the CSV commit fails. Avoid enabling unrelated workflows.
4. Streamlit Community Cloud redeploys after the `main` commit. Keep the existing app URL and ensure the entrypoint is `app.py`.
5. Once the scheduled workflow runs successfully, `signals.csv` appears in the repository. It may take time for Streamlit's deployed app to refresh after GitHub commits.

## Limitations
- GitHub scheduled workflows may run late or be skipped; not continuous monitoring or push notifications.
- The daily scheduled task is at 23:15 UTC Monday–Friday, usually after US equity market close, but market holidays, DST, feed delays and maintenance can cause stale/missing records.
- The recorded prediction uses the latest available close. The timestamp is when the prediction was recorded; a row does **not** mean an executable trade occurred.
- No paper-trading P&L yet: next-session fill assumptions, costs, position sizing and resolved outcomes must be added and validated separately.
- No real money should be risked on these research outputs.
