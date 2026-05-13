"""Shared target definitions for market + futures modelling."""

import pandas as pd

from src.config import BUY_THRESHOLD, LABEL_HORIZON_STEPS, PROCESSED_DIR

TIMEFRAME_TO_HOURS = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
}

TARGET_HOUR_MAP = {
    "fixed_h8": 8,
    "fixed_h24": 24,
    "fixed_h72": 72,
    "voladj_h24": 24,
    "voladj_h72": 72,
}


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_default_preferred_target_map():
    target_map = get_target_config_map()
    return {
        "BTCUSDT": target_map["fixed_h24"],
        "ETHUSDT": target_map["fixed_h24"],
        "SOLUSDT": target_map["voladj_h24"],
        "BNBUSDT": target_map["fixed_h24"],
        "XRPUSDT": target_map["fixed_h24"],
        "ADAUSDT": target_map["fixed_h24"],
        "DOGEUSDT": target_map["fixed_h24"],
    }


def _get_latest_target_variant_summary_path(timeframe):
    pattern = f"market_futures_target_variant_summary_{timeframe}_*.csv"
    matches = sorted(PROCESSED_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _score_target_variant_row(row):
    macro_f1 = _safe_float(row.get("test_macro_f1"))
    balanced_accuracy = _safe_float(row.get("test_balanced_accuracy"))
    accuracy = _safe_float(row.get("test_accuracy"))
    exact_horizon_match = str(row.get("target_exact_horizon_match", "")).lower() == "true"
    horizon_hours = _safe_float(row.get("target_horizon_hours"))
    status = str(row.get("status", "evaluated") or "evaluated")

    if status != "evaluated":
        return -float("inf")

    score = (
        macro_f1 * 1.0
        + balanced_accuracy * 0.35
        + accuracy * 0.15
    )

    if exact_horizon_match:
        score += 0.01

    if horizon_hours == 24:
        score += 0.005

    return score


def build_target_configs():
    """Define the first set of horizon-aware and volatility-aware target variants."""
    base_step_threshold = BUY_THRESHOLD / LABEL_HORIZON_STEPS
    return [
        {
            "target_name": "fixed_h8",
            "label_mode": "fixed",
            "horizon_steps": 2,
            "horizon_hours": 8,
            "buy_threshold": base_step_threshold * 2,
            "dont_buy_threshold": -(base_step_threshold * 2),
        },
        {
            "target_name": "fixed_h24",
            "label_mode": "fixed",
            "horizon_steps": 6,
            "horizon_hours": 24,
            "buy_threshold": BUY_THRESHOLD,
            "dont_buy_threshold": -BUY_THRESHOLD,
        },
        {
            "target_name": "fixed_h72",
            "label_mode": "fixed",
            "horizon_steps": 18,
            "horizon_hours": 72,
            "buy_threshold": base_step_threshold * 18,
            "dont_buy_threshold": -(base_step_threshold * 18),
        },
        {
            "target_name": "voladj_h24",
            "label_mode": "vol_adjusted",
            "horizon_steps": 6,
            "horizon_hours": 24,
            "vol_multiplier": 0.75,
        },
        {
            "target_name": "voladj_h72",
            "label_mode": "vol_adjusted",
            "horizon_steps": 18,
            "horizon_hours": 72,
            "vol_multiplier": 0.75,
        },
    ]


def get_target_config_map():
    """Return target configs keyed by target name."""
    return {config["target_name"]: config for config in build_target_configs()}


def resolve_target_config_for_timeframe(target_config, timeframe):
    """Translate an abstract target into timeframe-aware horizon semantics."""
    resolved = dict(target_config)
    timeframe_hours = TIMEFRAME_TO_HOURS.get(timeframe, 4)
    target_hours = int(resolved.get("horizon_hours", TARGET_HOUR_MAP.get(resolved.get("target_name"), timeframe_hours)))
    resolved_steps = max(1, round(target_hours / timeframe_hours))
    effective_horizon_hours = resolved_steps * timeframe_hours
    exact_horizon_match = effective_horizon_hours == target_hours

    resolved["requested_timeframe"] = timeframe
    resolved["timeframe_hours"] = timeframe_hours
    resolved["requested_horizon_hours"] = target_hours
    resolved["horizon_steps"] = resolved_steps
    resolved["effective_horizon_hours"] = effective_horizon_hours
    resolved["exact_horizon_match"] = exact_horizon_match
    resolved["horizon_resolution_note"] = (
        f"Target horizon resolves cleanly to {resolved_steps} {timeframe} candles."
        if exact_horizon_match
        else (
            f"Target horizon {target_hours}h does not map exactly to {timeframe} candles, "
            f"so LiveStrat uses {resolved_steps} candle(s) = {effective_horizon_hours}h."
        )
    )

    if resolved["label_mode"] == "fixed":
        effective_buy_threshold = BUY_THRESHOLD * (effective_horizon_hours / 24)
        resolved["buy_threshold"] = effective_buy_threshold
        resolved["dont_buy_threshold"] = -effective_buy_threshold

    return resolved


def describe_target_for_timeframe(target_name, timeframe):
    """Return timeframe-aware target semantics for one named target."""
    target_map = get_target_config_map()
    target_config = target_map.get(target_name)
    if not target_config:
        return {
            "target_name": target_name,
            "requested_timeframe": timeframe,
            "requested_horizon_hours": None,
            "effective_horizon_hours": None,
            "horizon_steps": None,
            "exact_horizon_match": False,
            "horizon_resolution_note": "Target semantics are not defined for this target name.",
        }
    return resolve_target_config_for_timeframe(target_config, timeframe)


def get_preferred_market_futures_targets(timeframe=None):
    """Return the current preferred target per symbol from the variant study."""
    preferred = _build_default_preferred_target_map()

    if timeframe in TIMEFRAME_TO_HOURS:
        summary_path = _get_latest_target_variant_summary_path(timeframe)
        if summary_path is not None and summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty:
                target_map = get_target_config_map()
                for symbol, symbol_df in summary_df.groupby("symbol"):
                    symbol_df = symbol_df.copy()
                    symbol_df["selection_score"] = symbol_df.apply(_score_target_variant_row, axis=1)
                    symbol_df = symbol_df.sort_values(
                        ["selection_score", "test_macro_f1", "test_balanced_accuracy", "test_accuracy"],
                        ascending=False,
                    )
                    best_row = symbol_df.iloc[0]
                    target_name = str(best_row.get("target_name", "") or "")
                    if target_name in target_map:
                        preferred[symbol] = target_map[target_name]

    if timeframe is None:
        return preferred
    return {
        symbol: resolve_target_config_for_timeframe(config, timeframe)
        for symbol, config in preferred.items()
    }


def build_target_labels(df, target_config, timeframe=None):
    """Create labels for one target definition on the combined dataset."""
    df = df.copy()
    if timeframe is not None:
        target_config = resolve_target_config_for_timeframe(target_config, timeframe)
    horizon_steps = target_config["horizon_steps"]
    df["future_close"] = df["close"].shift(-horizon_steps)
    df["future_return"] = (df["future_close"] / df["close"]) - 1.0
    df["label"] = "hold"

    if target_config["label_mode"] == "fixed":
        buy_threshold = target_config["buy_threshold"]
        dont_buy_threshold = target_config["dont_buy_threshold"]
        df["buy_threshold"] = buy_threshold
        df["dont_buy_threshold"] = dont_buy_threshold
    else:
        volatility = pd.to_numeric(df["volatility_20"], errors="coerce").fillna(0.0)
        horizon_vol = volatility * (horizon_steps ** 0.5) * target_config["vol_multiplier"]
        df["buy_threshold"] = horizon_vol
        df["dont_buy_threshold"] = -horizon_vol

    df.loc[df["future_return"] >= df["buy_threshold"], "label"] = "buy"
    df.loc[df["future_return"] <= df["dont_buy_threshold"], "label"] = "dont_buy"
    df["target_name"] = target_config.get("target_name")
    df["target_horizon_hours"] = target_config.get("effective_horizon_hours")
    df["target_exact_horizon_match"] = target_config.get("exact_horizon_match")
    return df.dropna(subset=["future_close", "future_return"]).reset_index(drop=True)
