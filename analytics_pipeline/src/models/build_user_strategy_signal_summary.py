"""Build the user-facing forecast and indicator decision layer."""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, get_all_symbols


TIMEFRAMES = ("1h", "4h")
SIGNAL_SCORE = {
    "buy": 1.0,
    "long": 1.0,
    "hold": 0.0,
    "flat": 0.0,
    "dont_buy": -1.0,
    "sell": -1.0,
    "avoid": -1.0,
}


def _latest_file(pattern):
    matches = list(Path(PROCESSED_DIR).glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _read_latest(pattern):
    path = _latest_file(pattern)
    if path is None:
        return path, pd.DataFrame()
    return path, pd.read_csv(path)


def _safe_float(value, default=0.0):
    try:
        if value in (None, "") or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_to_score(value):
    return SIGNAL_SCORE.get(str(value or "").strip().lower(), 0.0)


def _score_to_signal(score):
    if score >= 0.35:
        return "buy"
    if score <= -0.35:
        return "dont_buy"
    return "hold"


def _direction_label(value, positive="supportive", negative="caution", neutral="neutral"):
    value = _safe_float(value)
    if value >= 0.15:
        return positive
    if value <= -0.15:
        return negative
    return neutral


def _best_regression_rows(regression_df):
    if regression_df.empty:
        return {}
    working = regression_df.copy()
    working["directional_accuracy"] = pd.to_numeric(working.get("directional_accuracy", 0), errors="coerce").fillna(0)
    working["rmse"] = pd.to_numeric(working.get("rmse", 0), errors="coerce").fillna(0)
    working = working.sort_values(["symbol", "directional_accuracy", "rmse"], ascending=[True, False, True])
    return {row["symbol"]: row for _, row in working.drop_duplicates("symbol").iterrows()}


def _best_classification_rows(classification_df):
    if classification_df.empty:
        return {}
    working = classification_df.copy()
    working["macro_f1"] = pd.to_numeric(working.get("macro_f1", 0), errors="coerce").fillna(0)
    working["balanced_accuracy"] = pd.to_numeric(working.get("balanced_accuracy", 0), errors="coerce").fillna(0)
    working = working.sort_values(["symbol", "macro_f1", "balanced_accuracy"], ascending=[True, False, False])
    return {row["symbol"]: row for _, row in working.drop_duplicates("symbol").iterrows()}


def _rows_by_symbol(df):
    if df.empty or "symbol" not in df.columns:
        return {}
    return {row["symbol"]: row for _, row in df.drop_duplicates("symbol").iterrows()}


def _indicator_component(value, low=-0.01, high=0.01):
    value = _safe_float(value)
    if value >= high:
        return 1.0
    if value <= low:
        return -1.0
    return value / high if value >= 0 else abs(value / low) * -1.0


def _indicator_read(latest):
    trend_score = (
        _indicator_component(latest.get("price_sma_20_diff"), -0.015, 0.015) * 0.55
        + _indicator_component(latest.get("price_sma_50_diff"), -0.025, 0.025) * 0.45
    )
    momentum_score = (
        _indicator_component(latest.get("return_24h"), -0.025, 0.025) * 0.65
        + _indicator_component(latest.get("return_3d"), -0.05, 0.05) * 0.35
    )
    participation_score = (
        _indicator_component(_safe_float(latest.get("taker_buy_volume_ratio")) - 0.5, -0.05, 0.05) * 0.55
        + _indicator_component(latest.get("volume_zscore"), -1.0, 1.0) * 0.25
        + _indicator_component(latest.get("trade_count_zscore"), -1.0, 1.0) * 0.20
    )
    risk_penalty = min(max(_safe_float(latest.get("volatility_20")) / 0.04, 0.0), 1.0) * 0.20
    score = (trend_score * 0.40) + (momentum_score * 0.35) + (participation_score * 0.25)
    score = score - risk_penalty if score > 0 else score + risk_penalty
    score = max(min(score, 1.0), -1.0)
    return {
        "indicator_score": score,
        "indicator_signal": _score_to_signal(score),
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "participation_score": participation_score,
        "volatility_penalty": risk_penalty,
        "indicator_summary": (
            f"Trend is {_direction_label(trend_score)}, momentum is {_direction_label(momentum_score)}, "
            f"and participation is {_direction_label(participation_score)}."
        ),
    }


def _context_adjustment(context_row):
    sentiment = str(context_row.get("latest_effective_sentiment_label", context_row.get("latest_gdelt_regime_label", "")) or "").lower()
    onchain = str(context_row.get("latest_onchain_regime_label", context_row.get("latest_onchain_snapshot_label", "")) or "").lower()
    adjustment = 0.0
    notes = []

    if any(token in sentiment for token in ("support", "positive", "greed", "risk_on")):
        adjustment += 0.08
        notes.append("sentiment supports the market read")
    elif any(token in sentiment for token in ("fear", "negative", "risk", "caution")):
        adjustment -= 0.08
        notes.append("sentiment adds caution")

    if any(token in onchain for token in ("support", "accumulation", "healthy", "constructive")):
        adjustment += 0.10
        notes.append("on-chain context supports the signal")
    elif any(token in onchain for token in ("distribution", "fragility", "risk", "divergence")):
        adjustment -= 0.10
        notes.append("on-chain context warns against full conviction")

    return adjustment, "; ".join(notes) if notes else "no strong sentiment/on-chain override is available"


def build_user_strategy_signal_summary(timeframe, start_date=None, end_date=None, symbols=None):
    """Combine forecast, indicators, and available context into one strategy decision row per asset."""
    symbols = tuple(symbols or get_all_symbols())
    date_suffix = f"_{start_date}_{end_date}" if start_date and end_date else "_*"
    regression_path, regression_df = _read_latest(f"market_trend_regression_summary_{timeframe}{date_suffix}.csv")
    classification_path, classification_df = _read_latest(f"market_trend_forecast_summary_{timeframe}{date_suffix}.csv")
    futures_path, futures_df = _read_latest(f"market_futures_signal_summary_{timeframe}{date_suffix}.csv")
    context_path, context_df = _read_latest(f"market_intelligence_overview_{timeframe}{date_suffix}.csv")

    regressions = _best_regression_rows(regression_df)
    classifications = _best_classification_rows(classification_df)
    futures_rows = _rows_by_symbol(futures_df)
    context_rows = _rows_by_symbol(context_df)
    rows = []

    for symbol in symbols:
        feature_path = _latest_file(f"{symbol}_{timeframe}_market_features_labeled{date_suffix}.csv")
        if feature_path is None:
            continue
        market_df = pd.read_csv(feature_path)
        if market_df.empty:
            continue
        latest = market_df.iloc[-1]
        close = _safe_float(latest.get("close"))
        regression = regressions.get(symbol, {})
        classification = classifications.get(symbol, {})
        futures = futures_rows.get(symbol, {})
        context = context_rows.get(symbol, {})
        predicted_return = _safe_float(regression.get("latest_predicted_return"))
        predicted_price = close * (1.0 + predicted_return)
        indicator = _indicator_read(latest)
        context_adjustment, context_note = _context_adjustment(context)

        regression_score = _signal_to_score(regression.get("latest_predicted_posture"))
        classification_score = _signal_to_score(classification.get("latest_prediction"))
        futures_score = _signal_to_score(futures.get("latest_signal"))
        combined_score = (
            regression_score * 0.35
            + indicator["indicator_score"] * 0.25
            + futures_score * 0.25
            + classification_score * 0.15
            + context_adjustment
        )
        combined_score = max(min(combined_score, 1.0), -1.0)
        final_signal = _score_to_signal(combined_score)

        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "window_start": start_date or regression.get("window_start") or classification.get("window_start"),
                "window_end": end_date or regression.get("window_end") or classification.get("window_end"),
                "latest_close": close,
                "forecast_model": regression.get("model_name", "n/a"),
                "predicted_return": predicted_return,
                "predicted_price": predicted_price,
                "forecast_posture": regression.get("latest_predicted_posture", "n/a"),
                "forecast_directional_accuracy": _safe_float(regression.get("directional_accuracy")),
                "forecast_rmse": _safe_float(regression.get("rmse")),
                "classification_model": classification.get("model_name", "n/a"),
                "classification_signal": classification.get("latest_prediction", "n/a"),
                "classification_macro_f1": _safe_float(classification.get("macro_f1")),
                "futures_signal": futures.get("latest_signal", "n/a"),
                "futures_confidence": _safe_float(futures.get("latest_signal_confidence")),
                "indicator_signal": indicator["indicator_signal"],
                "indicator_score": indicator["indicator_score"],
                "trend_score": indicator["trend_score"],
                "momentum_score": indicator["momentum_score"],
                "participation_score": indicator["participation_score"],
                "volatility_penalty": indicator["volatility_penalty"],
                "context_adjustment": context_adjustment,
                "final_strategy_score": combined_score,
                "final_strategy_signal": final_signal,
                "strategy_signal_summary": (
                    f"{symbol} {timeframe} forecast points to {regression.get('latest_predicted_posture', 'n/a')} "
                    f"with a predicted price near {predicted_price:,.2f}. "
                    f"Indicators read {indicator['indicator_signal']}; futures read {futures.get('latest_signal', 'n/a')}. "
                    f"Context note: {context_note}. Final signal: {final_signal}."
                ),
                "indicator_summary": indicator["indicator_summary"],
                "source_regression_file": regression_path.name if regression_path else "n/a",
                "source_classification_file": classification_path.name if classification_path else "n/a",
                "source_futures_file": futures_path.name if futures_path else "n/a",
                "source_context_file": context_path.name if context_path else "n/a",
            }
        )

    output = pd.DataFrame(rows)
    output_name = f"user_strategy_signal_summary_{timeframe}{date_suffix if date_suffix != '_*' else ''}.csv"
    output_path = Path(PROCESSED_DIR) / output_name
    output.to_csv(output_path, index=False)
    print("user strategy signal summary generated")
    print(f"rows saved: {len(output)}")
    print(f"summary saved to: {output_path}")
    return output


if __name__ == "__main__":
    for timeframe in TIMEFRAMES:
        build_user_strategy_signal_summary(timeframe)
