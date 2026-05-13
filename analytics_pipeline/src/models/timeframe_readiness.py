"""Summarize real timeframe readiness for LiveStrat display, decision, and context lanes."""

from pathlib import Path

import pandas as pd

from src.models.evaluation_coverage import build_evaluation_coverage_snapshot
from src.models.runtime_support import (
    DEFAULT_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    build_runtime_support_snapshot,
)


def _sorted_unique(values):
    return sorted(set(values), key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item)


def _build_timeframe_entry(timeframe):
    return {
        "timeframe": timeframe,
        "asset_count": 0,
        "chart_assets": [],
        "display_assets": [],
        "decision_assets": [],
        "multimodal_assets": [],
        "sentiment_assets": [],
        "onchain_assets": [],
        "defi_assets": [],
        "market_futures_evaluated_assets": [],
        "multimodal_evaluated_assets": [],
        "onchain_evaluated_assets": [],
        "readiness_label": "missing",
        "summary": "",
        "recommendation": "",
    }


def _market_futures_has_walkforward_folds(processed_dir, symbol, timeframe):
    matches = list(Path(processed_dir).glob(f"market_futures_walkforward_summary_{timeframe}_*.csv"))
    if not matches:
        return False
    path = max(matches, key=lambda item: item.stat().st_mtime)
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return False
    if df.empty or "symbol" not in df.columns or "walkforward_fold_count" not in df.columns:
        return False
    asset_rows = df[df["symbol"] == symbol]
    if asset_rows.empty:
        return False
    return bool(pd.to_numeric(asset_rows["walkforward_fold_count"], errors="coerce").fillna(0).max() > 0)


def _label_timeframe_entry(entry):
    asset_count = entry["asset_count"] or 1
    display_ready = len(entry["display_assets"]) == asset_count
    decision_ready = len(entry["decision_assets"]) == asset_count
    market_futures_evaluated = len(entry["market_futures_evaluated_assets"]) == asset_count
    multimodal_ready = len(entry["multimodal_assets"]) == asset_count and len(entry["multimodal_evaluated_assets"]) == asset_count

    if display_ready and decision_ready and market_futures_evaluated and multimodal_ready:
        entry["readiness_label"] = "ready"
        entry["recommendation"] = (
            f"{entry['timeframe']} already has full display, decision, and multimodal coverage for tracked assets."
        )
    elif display_ready and decision_ready:
        entry["readiness_label"] = "partial"
        entry["recommendation"] = (
            f"{entry['timeframe']} is usable for display and core decision logic, but multimodal/context evaluation still needs work."
        )
    elif display_ready:
        entry["readiness_label"] = "display_only"
        entry["recommendation"] = (
            f"{entry['timeframe']} has enough generated market data for charts and summaries, but strategy evaluations are still incomplete."
        )
    else:
        entry["readiness_label"] = "missing"
        entry["recommendation"] = (
            f"{entry['timeframe']} still needs a fresh market-intelligence pipeline run before it should be exposed as a supported timeframe."
        )

    entry["summary"] = (
        f"{entry['timeframe']} currently has display support for {len(entry['display_assets'])}/{asset_count} assets, "
        f"decision support for {len(entry['decision_assets'])}/{asset_count}, "
        f"market+futures evaluation for {len(entry['market_futures_evaluated_assets'])}/{asset_count}, "
        f"and multimodal evaluation for {len(entry['multimodal_evaluated_assets'])}/{asset_count}."
    )


def build_timeframe_readiness_snapshot(processed_dir, raw_binance_dir, market_symbols=None):
    """Describe what each timeframe can genuinely support right now."""
    processed_dir = Path(processed_dir)
    raw_binance_dir = Path(raw_binance_dir)
    market_symbols = tuple(market_symbols or DEFAULT_SYMBOLS)
    runtime = build_runtime_support_snapshot(processed_dir, raw_binance_dir, market_symbols)
    coverage = build_evaluation_coverage_snapshot(processed_dir)

    assets = {}
    timeframe_summary = {timeframe: _build_timeframe_entry(timeframe) for timeframe in SUPPORTED_TIMEFRAMES}

    for timeframe in SUPPORTED_TIMEFRAMES:
        timeframe_summary[timeframe]["asset_count"] = len(market_symbols)

    for symbol in market_symbols:
        asset_runtime = runtime["assets"].get(symbol, {})
        asset_coverage = coverage["assets"].get(symbol, {})
        asset_timeframes = {}

        for timeframe in SUPPORTED_TIMEFRAMES:
            chart_ready = timeframe in asset_runtime.get("chart_timeframes", [])
            display_ready = timeframe in asset_runtime.get("market_summary_timeframes", [])
            decision_ready = timeframe in asset_runtime.get("decision_timeframes", [])
            multimodal_ready = timeframe in asset_runtime.get("multimodal_timeframes", [])
            sentiment_ready = timeframe in asset_runtime.get("sentiment_timeframes", [])
            onchain_ready = timeframe in asset_runtime.get("onchain_timeframes", [])
            defi_ready = timeframe in asset_runtime.get("defi_timeframes", [])
            market_futures_evaluated = (
                timeframe in asset_coverage.get("market_futures", {}).get("available_timeframes", [])
                and _market_futures_has_walkforward_folds(processed_dir, symbol, timeframe)
            )
            multimodal_evaluated = timeframe in asset_coverage.get("multimodal", {}).get("available_timeframes", [])
            onchain_evaluated = timeframe in asset_coverage.get("onchain_specialist", {}).get("available_timeframes", [])

            asset_timeframes[timeframe] = {
                "chart_ready": chart_ready,
                "display_ready": display_ready,
                "decision_ready": decision_ready,
                "multimodal_ready": multimodal_ready,
                "sentiment_ready": sentiment_ready,
                "onchain_ready": onchain_ready,
                "defi_ready": defi_ready,
                "market_futures_evaluated": market_futures_evaluated,
                "multimodal_evaluated": multimodal_evaluated,
                "onchain_evaluated": onchain_evaluated,
            }

            summary_entry = timeframe_summary[timeframe]
            if chart_ready:
                summary_entry["chart_assets"].append(symbol)
            if display_ready:
                summary_entry["display_assets"].append(symbol)
            if decision_ready:
                summary_entry["decision_assets"].append(symbol)
            if multimodal_ready:
                summary_entry["multimodal_assets"].append(symbol)
            if sentiment_ready:
                summary_entry["sentiment_assets"].append(symbol)
            if onchain_ready:
                summary_entry["onchain_assets"].append(symbol)
            if defi_ready:
                summary_entry["defi_assets"].append(symbol)
            if market_futures_evaluated:
                summary_entry["market_futures_evaluated_assets"].append(symbol)
            if multimodal_evaluated:
                summary_entry["multimodal_evaluated_assets"].append(symbol)
            if onchain_evaluated:
                summary_entry["onchain_evaluated_assets"].append(symbol)

        assets[symbol] = {
            "asset": symbol,
            "timeframes": asset_timeframes,
        }

    for timeframe, summary_entry in timeframe_summary.items():
        for key in (
            "chart_assets",
            "display_assets",
            "decision_assets",
            "multimodal_assets",
            "sentiment_assets",
            "onchain_assets",
            "defi_assets",
            "market_futures_evaluated_assets",
            "multimodal_evaluated_assets",
            "onchain_evaluated_assets",
        ):
            summary_entry[key] = _sorted_unique(summary_entry[key])
        _label_timeframe_entry(summary_entry)

    recommended_next_runs = []
    for timeframe in SUPPORTED_TIMEFRAMES:
        entry = timeframe_summary[timeframe]
        if entry["readiness_label"] != "ready":
            recommended_next_runs.append(
                {
                    "timeframe": timeframe,
                    "priority": "high" if entry["readiness_label"] == "missing" else "medium",
                    "reason": entry["recommendation"],
                }
            )

    return {
        "assets": assets,
        "timeframes": timeframe_summary,
        "recommended_next_runs": recommended_next_runs,
    }
