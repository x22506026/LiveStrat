# LiveStrat

Web platform for inspecting cryptocurrency markets and running a set of trading strategies. Each signal is shown with the evidence behind it.
Built with Python, Flask, SQLAlchemy, and PostgreSQL. Plain HTML, CSS, JS on the frontend.

## Data sources

- Binance Spot: OHLCV, returns, volatility, taker flow
- Binance Futures: funding, open interest, long/short ratio, basis
- GDELT: asset news, scored with FinBERT
- Coin Metrics: on-chain network telemetry
- DeFiLlama: chain level TVL
- Alternative.me: Fear and Greed Index

## Tracked assets

BTC, ETH, SOL, BNB, XRP, ADA, DOGE.

## Pages

- Dashboard
- Markets
- Strategies
- Strategy detail
- Analytics
- Account
- Messages

## Setup

PostgreSQL runs in Docker. Flask runs locally.

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
pip install -r requirements.txt
pip install -r analytics_pipeline\requirements.txt
$env:FLASK_APP = 'app.py'
flask init-db
python app.py
```

Default database URL is `postgresql+psycopg2://livestrat:livestrat_password@localhost:5432/livestrat`.

One go bootstrap that does all the steps above plus the test suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
To run without Docker, set `DATABASE_URL=sqlite:///livestrat.db` in `.env`.


## Running the app after first setup

```powershell
cd LiveStrat-main
docker compose up -d postgres
py app.py
```
Open http://127.0.0.1:5000 in a browser.


## Refreshing the analytics pipeline

The app reads pre-computed CSVs from `analytics_pipeline/data/processed/`. To regenerate them:

```powershell
$env:PYTHONPATH = 'analytics_pipeline'

python -m src.build.run_market_intelligence_pipeline `
    --timeframe 4h --start-date 2026-02-04 --end-date 2026-05-04 --core-only

python -m src.build.run_market_intelligence_pipeline `
    --timeframe 4h --start-date 2026-02-04 --end-date 2026-05-04 --context-only

python -m src.build.run_multi_timeframe_market_intelligence_pipeline `
    --timeframes 1h 4h 1d --core-only
```

Add `--no-resume` to force a full rebuild. The pipeline writes a refresh manifest at `analytics_pipeline/data/processed/market_intelligence_refresh_manifest.json`.


## Testing

```powershell
python -m unittest discover -s tests -v
```
Three test suites:

- `test_backend_contracts.py` covers the seven asset universe and source governance.
- `test_ml_correctness.py` covers chronological splits, label leakage prevention, calibration isolation.
- `test_statistical_tests.py` covers the bootstrap p-value, Diebold-Mariano, Probabilistic Sharpe and Deflated Sharpe.


## Reproducibility

All randomness is seeded from `LIVESTRAT_SEED = 1729` in `analytics_pipeline/src/reproducibility.py`. Both `requirements.txt` files pin versions with `==`.


## Report figures

```powershell
$env:PYTHONPATH = 'analytics_pipeline'
python -m src.reports.build_report_figures
```
Writes seven PNGs to `report_figures/`.


## FinBERT validation

```powershell
$env:PYTHONPATH = 'analytics_pipeline'
python -m src.sentiment.validate_finbert
```
Runs FinBERT and the lexical fallback on the labelled headlines in `analytics_pipeline/data/processed/finbert_validation_set.csv`. Writes per row results to `finbert_validation_run.csv` and prints overall and per class agreement.


## Project structure

```
LiveStrat/
  app.py
  account_services.py
  database.py
  models.py
  telegram_delivery.py
  setup.ps1
  requirements.txt
  docker-compose.yml
  templates/
  static/css/
  static/css/js/
  tests/
  analytics_pipeline/
    src/audit/
    src/build/
    src/models/
    src/sentiment/
    src/reports/
    data/processed/
  report_figures/
```
