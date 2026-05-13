"""Runtime support snapshot for assets, timeframes, and strategy families."""

from pathlib import Path
import re

import pandas as pd

from src.config import get_all_symbols

DEFAULT_SYMBOLS = tuple(get_all_symbols())
SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")
NAMED_DECISION_STRATEGIES = {
    "recommended": "primary_production_candidate",
    "conservative_trend": "strict_trend_candidate",
    "momentum_breakout": "intraday_momentum_candidate",
    "futures_crowd_reversal": "futures_positioning_candidate",
    "multimodal_balanced": "context_adjusted_candidate",
    "daily_structural_confirmation": "higher_timeframe_confirmation",
}
MARKET_FILE_RE = re.compile(r"^(?P<symbol>[A-Z]+USDT)_(?P<timeframe>1h|4h|1d)_")


def _build_asset_support(symbol):
    return {
        "asset": symbol,
        "chart_timeframes": set(),
        "market_summary_timeframes": set(),
        "decision_timeframes": set(),
        "multimodal_timeframes": set(),
        "sentiment_timeframes": set(),
        "onchain_timeframes": set(),
        "defi_timeframes": set(),
        "strategy_statuses": {},
    }


def _finalize_statuses(asset_support):
    decision_timeframes = asset_support["decision_timeframes"]
    market_timeframes = asset_support["market_summary_timeframes"]
    multimodal_timeframes = asset_support["multimodal_timeframes"]
    sentiment_timeframes = asset_support["sentiment_timeframes"]
    onchain_timeframes = asset_support["onchain_timeframes"]
    defi_timeframes = asset_support["defi_timeframes"]

    asset_support["strategy_statuses"] = {
        **{
            strategy_key: {
                "role": role,
                "status": "available" if decision_timeframes else "unavailable",
                "available_timeframes": sorted(decision_timeframes),
            }
            for strategy_key, role in NAMED_DECISION_STRATEGIES.items()
        },
        "market_futures_backend": {
            "role": "primary_production_candidate",
            "status": "available" if decision_timeframes else "unavailable",
            "available_timeframes": sorted(decision_timeframes),
        },
        "rule_based": {
            "role": "transparent_benchmark",
            "status": "available" if market_timeframes else "unavailable",
            "available_timeframes": sorted(market_timeframes),
        },
        "scaled_model": {
            "role": "market_only_ml_baseline",
            "status": "available" if market_timeframes else "unavailable",
            "available_timeframes": sorted(market_timeframes),
        },
        "unscaled_model": {
            "role": "comparison_baseline",
            "status": "research_only" if market_timeframes else "unavailable",
            "available_timeframes": sorted(market_timeframes),
        },
        "multimodal_balanced": {
            "role": "experimental_context_strategy",
            "status": "experimental" if multimodal_timeframes else "unavailable",
            "available_timeframes": sorted(multimodal_timeframes),
        },
        "market_onchain_specialist": {
            "role": "daily_specialist_strategy",
            "status": "research_only" if onchain_timeframes else "unavailable",
            "available_timeframes": sorted(onchain_timeframes),
        },
        "sentiment_confirmation": {
            "role": "confirmation_layer_only",
            "status": "conditional" if sentiment_timeframes else "unavailable",
            "available_timeframes": sorted(sentiment_timeframes),
        },
        "defi_ecosystem_context": {
            "role": "ecosystem_confirmation_layer",
            "status": "conditional" if defi_timeframes else "unavailable",
            "available_timeframes": sorted(defi_timeframes),
        },
    }


def _onchain_file_has_available_rows(path):
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return False
    if df.empty:
        return False
    if "onchain_data_available" not in df.columns:
        return True
    return bool(df["onchain_data_available"].astype(str).str.lower().eq("true").any())


def build_runtime_support_snapshot(processed_dir, raw_binance_dir, market_symbols=None):
    """Scan generated artifacts to describe real current support in the project."""
    processed_dir = Path(processed_dir)
    raw_binance_dir = Path(raw_binance_dir)
    market_symbols = tuple(market_symbols or DEFAULT_SYMBOLS)
    asset_support = {symbol: _build_asset_support(symbol) for symbol in market_symbols}

    for raw_path in raw_binance_dir.glob("*.csv"):
        match = MARKET_FILE_RE.match(raw_path.name)
        if not match:
            continue
        symbol = match.group("symbol")
        timeframe = match.group("timeframe")
        if symbol in asset_support:
            asset_support[symbol]["chart_timeframes"].add(timeframe)

    for processed_path in processed_dir.glob("*.csv"):
        name = processed_path.name
        market_match = MARKET_FILE_RE.match(name)

        if market_match:
            symbol = market_match.group("symbol")
            timeframe = market_match.group("timeframe")
            if symbol not in asset_support:
                asset_support[symbol] = _build_asset_support(symbol)

            if "_market_summary" in name or "_market_features" in name:
                asset_support[symbol]["market_summary_timeframes"].add(timeframe)

            if "_market_futures_" in name and (
                "evaluation_metrics" in name
                or "dataset" in name
                or "backtest_curve" in name
            ) and timeframe in {"1h", "4h"}:
                asset_support[symbol]["decision_timeframes"].add(timeframe)

            if "_market_multimodal_" in name and (
                "evaluation_metrics" in name
                or "dataset" in name
            ):
                asset_support[symbol]["multimodal_timeframes"].add(timeframe)

        for symbol in tuple(asset_support.keys()):
            base_asset = symbol.replace("USDT", "")
            if name.startswith(f"{base_asset}_") and "_gdelt_sentiment_" in name:
                asset_support[symbol]["sentiment_timeframes"].add("1d")
            if name.startswith(f"{base_asset}_") and (
                "_market_onchain_" in name
                or "_onchain_" in name
            ) and _onchain_file_has_available_rows(processed_path):
                asset_support[symbol]["onchain_timeframes"].add("1d")
            if name.startswith(f"{base_asset}_") and "_defillama_features_" in name:
                asset_support[symbol]["defi_timeframes"].add("1d")

    for symbol, support in asset_support.items():
        _finalize_statuses(support)
        for key in (
            "chart_timeframes",
            "market_summary_timeframes",
            "decision_timeframes",
            "multimodal_timeframes",
            "sentiment_timeframes",
            "onchain_timeframes",
            "defi_timeframes",
        ):
            support[key] = sorted(support[key], key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item)

    return {
        "assets": asset_support,
        "global_timeframes": {
            "display_lane": sorted(
                {timeframe for support in asset_support.values() for timeframe in support["market_summary_timeframes"]},
                key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item,
            ),
            "decision_lane": sorted(
                {timeframe for support in asset_support.values() for timeframe in support["decision_timeframes"]},
                key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item,
            ),
            "context_daily": sorted(
                {timeframe for support in asset_support.values() for timeframe in (support["sentiment_timeframes"] + support["onchain_timeframes"])},
                key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item,
            ),
            "ecosystem_context": sorted(
                {timeframe for support in asset_support.values() for timeframe in support["defi_timeframes"]},
                key=lambda item: SUPPORTED_TIMEFRAMES.index(item) if item in SUPPORTED_TIMEFRAMES else item,
            ),
        },
    }
