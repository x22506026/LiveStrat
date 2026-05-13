# LiveStrat

LiveStrat is a web platform for inspecting the cryptocurrency market
and running a small set of well-defined trading strategies on it. The
point of the project is not to predict the market - it is to show *the
evidence behind every signal* so the person reading it can make their
own mind up.

Most retail crypto dashboards either hide their methodology entirely or
overpromise on what their signals do. LiveStrat goes the other way:
every number on the page is traceable back to a walk-forward fold, a
labelled dataset, and a refresh manifest. If a signal looks weak, the
app says so.

Built with Python and Flask. Data lives in PostgreSQL. The analytics
pipeline pulls:

- spot market data from Binance,
- futures structure (funding, open interest, basis) from Binance Futures,
- news sentiment from GDELT, scored with FinBERT,
- on-chain telemetry from Coin Metrics,
- ecosystem TVL from DeFiLlama.

Seven assets are tracked end-to-end: BTC, ETH, SOL, BNB, XRP, ADA and
DOGE.

---

## What the platform does

The app has five pages, all wired to the same evaluation outputs:

- **Dashboard** - tracked asset strip with price, 24h move, and a quick
  buy/hold/avoid pill per asset. Today's market read. A featured asset
  with its walk-forward Sharpe and fold count.
- **Markets** - pick an asset to inspect price behaviour and see which
  context layers (futures, sentiment, on-chain, TVL) are available for
  it.
- **Strategies** - the main decision page. Five user-facing strategies
  (trend, mean-reversion, breakout, sentiment-overlay, on-chain
  overlay). Each one shows the current signal, the evidence behind it,
  and a track record block based on walk-forward folds.
- **Analytics** - methodology overview, current evaluation numbers,
  ablation cards, transaction-cost sensitivity, and evidence tables
  across all seven assets. Technical detail is in collapsible blocks so
  the page reads top-down.
- **Account / Messages** - sign-up, login, saved strategy profiles, and
  an in-app message centre with optional Telegram delivery.

---

## What makes this different from other dashboards

Three things drove the design.

**1. Strategies are evaluated with walk-forward folds, not a single
train/test split.** Every fold uses a chronological split, so the model
never sees the future. The reported Sharpe is the median across folds,
and the page also shows fold count - if there are too few folds to
trust the number, the page says so.

**2. Every figure on the page is governed.** A source governance policy
decides which assets are allowed to use which data layer. For example,
DeFiLlama TVL is only attached to assets that actually have ecosystem
TVL to track, so no spurious overlay ever lands on BTC. The policy
lives in `analytics_pipeline/src/models/source_governance.py` and is
covered by the test suite.

**3. Ablations are first-class.** The Analytics page shows ablation
cards that compare each strategy to its own baseline (for example,
trend strategy vs. trend strategy with the sentiment overlay removed).
If a feature does not pull its weight, the ablation card shows it.

The point is not that LiveStrat predicts well - it is that the parts
that work are *separable from the parts that do not*, and the page
shows both.

---

## Local setup

PostgreSQL runs in Docker, Flask runs locally.

1. Copy the example environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env` and set a strong `SECRET_KEY`. If you want Telegram
   delivery, paste a fresh bot token there too.

2. Start PostgreSQL:

   ```powershell
   docker compose up -d postgres
   ```

3. Install the pinned Python dependencies:

   ```powershell
   pip install -r requirements.txt
   pip install -r analytics_pipeline\requirements.txt
   ```

4. Create the database tables:

   ```powershell
   $env:FLASK_APP = 'app.py'
   flask init-db
   ```

5. Run Flask:

   ```powershell
   python app.py
   ```

The default local database URL is
`postgresql+psycopg2://livestrat:livestrat_password@localhost:5432/livestrat`.

A one-command bootstrap is provided in `setup.ps1`. It runs all of the
steps above plus the test suite.

---

## Refreshing the analytics pipeline

The app reads pre-computed evaluation outputs from
`analytics_pipeline/data/processed/`. To regenerate those:

```powershell
cd LiveStrat-main
$env:PYTHONPATH = 'analytics_pipeline'

# Fast core refresh (market + futures + strategies + walk-forward + backtests)
python -m src.build.run_market_intelligence_pipeline `
    --timeframe 4h --start-date 2026-02-04 --end-date 2026-05-04 --core-only

# Context-only refresh (sentiment, on-chain, TVL)
python -m src.build.run_market_intelligence_pipeline `
    --timeframe 4h --start-date 2026-02-04 --end-date 2026-05-04 --context-only

# Multi-timeframe core refresh
python -m src.build.run_multi_timeframe_market_intelligence_pipeline `
    --timeframes 1h 4h 1d --core-only
```

Add `--no-resume` to force a full rebuild instead of reusing existing
context outputs. The pipeline writes a refresh manifest at
`analytics_pipeline/data/processed/market_intelligence_refresh_manifest.json`
which the app reads to decide whether the data is current, stale, or
missing. The dashboard surfaces this so the user knows what they are
looking at.

---

## Testing

Three test suites cover the codebase:

- **Backend contracts** (`tests/test_backend_contracts.py`) - seven-asset
  universe, source governance policy, DeFiLlama coverage, separation
  between Coin Metrics and DeFiLlama.
- **ML correctness** (`tests/test_ml_correctness.py`) - look-ahead
  prevention, chronological splits, label causality, four-way
  calibration isolation, deterministic seeds.
- **Statistical tests** (`tests/test_statistical_tests.py`) - paired
  bootstrap p-value, Diebold-Mariano test, Probabilistic Sharpe,
  Deflated Sharpe.

Run everything:

```powershell
python -m unittest discover -s tests -v
```

The first two suites are the most important. They are what stops the
project from silently regressing into a look-ahead bug or a
seven-asset-becomes-six bug, both of which would invalidate every
number on the page.

---

## Reproducibility

- Every stochastic library (Python `random`, NumPy, scikit-learn,
  TensorFlow, PyTorch) is seeded from a single master constant
  declared in `analytics_pipeline/src/reproducibility.py`
  (`LIVESTRAT_SEED = 1729`).
- Both `requirements.txt` files pin every library to a known version
  with `==`. A rebuild on a fresh machine should produce identical
  evaluation CSVs.

---

## Regenerating the report figures

```powershell
$env:PYTHONPATH = 'analytics_pipeline'
python -m src.reports.build_report_figures
```

Writes seven publication-quality PNGs into `report_figures/`:
per-asset accuracy, strategy vs hold, Sharpe per asset, transaction
cost sensitivity, ablation macro-F1, FinBERT confusion matrix,
walk-forward stability.

---

## Confirming the FinBERT path

```powershell
$env:PYTHONPATH = 'analytics_pipeline'
python -m src.sentiment.validate_finbert
```

Runs FinBERT on 16 hand-curated crypto-news headlines and writes
`finbert_validation_run.csv`. The expected agreement is in the high
70 percent range overall (100 percent on negatives, ~70 percent on
positives, lower on neutrals - the standard FinBERT pattern).

---

## Project structure

```
LiveStrat/
  app.py                  Flask routes and view payloads
  account_services.py     Auth and saved profile helpers
  database.py             SQLAlchemy instance
  models.py               User, AlertPreference, SavedStrategyProfile, NotificationEvent
  telegram_delivery.py    Telegram bot integration
  setup.ps1               One-command bootstrap
  requirements.txt        App dependencies (pinned)
  docker-compose.yml      PostgreSQL container
  templates/              Jinja2 templates per page
  static/                 Page-scoped CSS and JS
  tests/                  Three test suites
  analytics_pipeline/
    src/audit/            API fetchers (Binance, GDELT, Coin Metrics, DeFiLlama)
    src/build/            Feature engineering and dataset merge
    src/models/           Strategy evaluation and statistical tests
    src/sentiment/        FinBERT validation
    src/reports/          Summary builders and report figures
    data/processed/       Generated CSVs that the app reads
  report_figures/         Auto-generated PNGs for the report
```
