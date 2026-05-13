"""Transaction-cost sensitivity sweep for LiveStrat backtests.

Replays every existing market+futures backtest curve at 1x, 2x, 3x and 5x the
baseline taker fee and re-computes the headline metrics (total return,
excess return, Sharpe, max drawdown). A strategy that survives the 2x and
3x stress tests is materially more credible than one whose edge disappears
the moment fees double - which is the standard "is this real?" test in
the quant-trading literature.

The baseline taker fee assumed in the original backtest curves is 10 basis
points (0.10%). The sensitivity sweep re-applies the fee on each *position
change* by subtracting ``baseline_fee * multiplier`` from the strategy
return for that row.

Output: one summary CSV per timeframe, written next to the existing
strategy summaries:
    market_futures_cost_sensitivity_summary_{timeframe}_{window}.csv

CSV columns:
    symbol, timeframe, cost_multiplier, total_strategy_return,
    total_buy_hold_return, excess_return, sharpe_ratio, max_drawdown,
    hit_rate, trade_count, exposure_ratio.

References
----------
Lopez de Prado, M. (2018) *Advances in Financial Machine Learning*,
    chapter 14 (backtesting protocols).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_TIMEFRAME,
    PROCESSED_DIR,
    get_all_symbols,
)


BASELINE_TAKER_FEE = 0.001  # 10 basis points per side, the LiveStrat default.
COST_MULTIPLIERS = (1.0, 2.0, 3.0, 5.0)
POSITION_CHANGE_ACTIONS = {"enter_long", "exit_to_cash", "enter_short", "exit_to_long"}


def _annualisation_factor(timeframe: str) -> int:
    """Return periods-per-year for a given timeframe label."""
    return {
        "1h": 24 * 365,
        "4h": 6 * 365,
        "1d": 365,
    }.get(timeframe, 6 * 365)


def _find_backtest_curve(symbol: str, timeframe: str, window_end: Optional[str] = None) -> Optional[Path]:
    """Return the most recent market+futures backtest curve CSV for one asset."""
    pattern = f"{symbol}_{timeframe}_market_futures_backtest_curve_*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return None
    if window_end:
        windowed = [m for m in matches if window_end in m.name]
        if windowed:
            return max(windowed, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def _recompute_returns_with_cost(curve: pd.DataFrame, cost: float) -> pd.Series:
    """Apply a per-trade cost to a curve's per-period strategy return.

    The original curve already includes the baseline 10 bps cost on the
    position-change rows. We strip that out and add ``cost`` instead. The
    result is the per-period return series that *would have been earned*
    had fees been ``cost`` instead of ``BASELINE_TAKER_FEE``.
    """
    base_returns = curve["strategy_return"].astype(float).fillna(0.0).copy()
    actions = curve["action"].astype(str)
    is_trade = actions.isin(POSITION_CHANGE_ACTIONS)
    # Add the old cost back, subtract the new one.
    adjusted = base_returns.copy()
    adjusted.loc[is_trade] = base_returns.loc[is_trade] + BASELINE_TAKER_FEE - cost
    return adjusted


def _summarise_curve(
    returns: pd.Series,
    buy_hold_returns: pd.Series,
    timeframe: str,
    trade_count: int,
    position_series: pd.Series,
) -> dict:
    """Compute the headline metrics over a per-period return series."""
    equity = (1.0 + returns).cumprod()
    buy_hold_equity = (1.0 + buy_hold_returns).cumprod()

    total_strategy_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    total_buy_hold_return = float(buy_hold_equity.iloc[-1] - 1.0) if not buy_hold_equity.empty else 0.0

    periods_per_year = _annualisation_factor(timeframe)
    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1)) if returns.shape[0] > 1 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0

    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    in_market_count = int((position_series.astype(float) != 0).sum())
    exposure_ratio = in_market_count / max(len(position_series), 1)

    winners = returns[returns > 0]
    hit_rate = float(len(winners) / max(in_market_count, 1)) if in_market_count else 0.0

    return {
        "total_strategy_return": total_strategy_return,
        "total_buy_hold_return": total_buy_hold_return,
        "excess_return": total_strategy_return - total_buy_hold_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "trade_count": trade_count,
        "exposure_ratio": exposure_ratio,
    }


def run_sensitivity_sweep(
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    symbols: Optional[Iterable[str]] = None,
    cost_multipliers: Iterable[float] = COST_MULTIPLIERS,
    window_end: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the cost sensitivity sweep across all assets for one timeframe.

    Returns the resulting DataFrame and writes it to disk under
    ``analytics_pipeline/data/processed/`` if ``output_path`` is None.
    """
    symbols = list(symbols) if symbols is not None else get_all_symbols()
    rows: list[dict] = []

    for symbol in symbols:
        path = _find_backtest_curve(symbol, timeframe, window_end)
        if path is None:
            continue

        curve = pd.read_csv(path)
        required = {"strategy_return", "buy_hold_return", "action", "position"}
        if not required.issubset(curve.columns):
            continue

        buy_hold = curve["buy_hold_return"].astype(float).fillna(0.0)
        trade_count = int(curve["action"].astype(str).isin(POSITION_CHANGE_ACTIONS).sum())
        position_series = curve["position"].astype(float).fillna(0.0)

        for multiplier in cost_multipliers:
            cost = BASELINE_TAKER_FEE * multiplier
            adjusted = _recompute_returns_with_cost(curve, cost)
            summary = _summarise_curve(
                adjusted, buy_hold, timeframe, trade_count, position_series
            )
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "window_source": path.name,
                    "cost_multiplier": multiplier,
                    "per_trade_cost_bps": cost * 10000.0,
                    **summary,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if output_path is None:
        suffix = f"_{window_end}" if window_end else ""
        output_path = (
            PROCESSED_DIR
            / f"market_futures_cost_sensitivity_summary_{timeframe}{suffix}.csv"
        )
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    summary = run_sensitivity_sweep(timeframe="4h")
    if summary.empty:
        print("No backtest curves found - run the strategy evaluation first.")
    else:
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
