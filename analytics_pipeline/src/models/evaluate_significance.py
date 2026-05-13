"""Apply statistical-significance tests to live LiveStrat strategy outputs.

For every available backtest curve in ``analytics_pipeline/data/processed/``
this script reads the per-period strategy returns and the matching
buy-and-hold returns, then runs the four tests from
``statistical_tests.py``:

* paired bootstrap on the mean return difference,
* Diebold-Mariano on the negative-return loss,
* Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012),
* Deflated Sharpe Ratio with the multiple-trials correction
  (Bailey and Lopez de Prado, 2014).

Output: ``analytics_pipeline/data/processed/strategy_significance_summary_{timeframe}.csv``
with one row per (symbol, comparison) plus an ``evidence_label`` column.

This operationalises the statistical hooks declared in
``statistical_tests.py``: instead of only existing as a tested module, the
tests are now run against actual evaluation outputs and persisted as part
of the report-ready artefact set.

Run from the repo root with:

    $env:PYTHONPATH = 'analytics_pipeline'
    python -m src.models.evaluate_significance
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DEFAULT_TIMEFRAME, PROCESSED_DIR, get_all_symbols
from src.models.statistical_tests import summarise_strategy_vs_benchmark


PERIODS_PER_YEAR = {
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}


def _latest_curve_for(symbol: str, timeframe: str) -> Path | None:
    pattern = f"{symbol}_{timeframe}_market_futures_backtest_curve_*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _read_returns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    curve = pd.read_csv(path)
    strat = pd.to_numeric(curve.get("strategy_return"), errors="coerce").fillna(0.0).to_numpy()
    hold = pd.to_numeric(curve.get("buy_hold_return"), errors="coerce").fillna(0.0).to_numpy()
    return strat, hold


def run_significance_sweep(
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: Iterable[str] | None = None,
    n_strategy_trials: int = 35,
) -> pd.DataFrame:
    """Run statistical tests across every asset's strategy-vs-hold comparison.

    ``n_strategy_trials`` should reflect the *number of strategy variants
    that were tried before the winner was selected* for each asset. In
    LiveStrat this is roughly: 7 assets x 5 target horizons x ~1 winning
    policy = ~35 candidates per asset. The Deflated Sharpe Ratio penalises
    by this count.
    """
    symbols = list(symbols) if symbols is not None else get_all_symbols()
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, PERIODS_PER_YEAR["4h"])

    rows: list[dict] = []
    for symbol in symbols:
        path = _latest_curve_for(symbol, timeframe)
        if path is None:
            continue
        strat, hold = _read_returns(path)
        if len(strat) < 10:
            continue

        summary = summarise_strategy_vs_benchmark(
            strat,
            hold,
            n_trials=n_strategy_trials,
            periods_per_year=periods_per_year,
        )
        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "window_source": path.name,
            **summary,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    output_path = PROCESSED_DIR / f"strategy_significance_summary_{timeframe}.csv"
    df.to_csv(output_path, index=False)
    return df


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _print_table(df: pd.DataFrame) -> None:
    print(f"{'asset':>8} {'mean_diff':>10} {'p_boot':>10} {'DM_p':>10} {'sharpe':>8} {'PSR':>6} {'DSR':>6}  evidence")
    print("-" * 92)
    for _, row in df.iterrows():
        print(
            f"{row['symbol']:>8} "
            f"{row['mean_difference']:>10.5f} "
            f"{_format_p(row['bootstrap_p_value']):>10} "
            f"{_format_p(row['diebold_mariano_p_value']):>10} "
            f"{row['annualised_sharpe']:>8.2f} "
            f"{row['probabilistic_sharpe_ratio']:>6.2f} "
            f"{row['deflated_sharpe_ratio']:>6.2f}  "
            f"{row['evidence_label']}"
        )


def main() -> None:
    for tf in ("1h", "4h", "1d"):
        df = run_significance_sweep(timeframe=tf)
        if df.empty:
            print(f"[{tf}] no backtest curves found")
            continue
        print(f"\n=== {tf} significance sweep ({len(df)} comparisons) ===")
        _print_table(df)


if __name__ == "__main__":
    main()
