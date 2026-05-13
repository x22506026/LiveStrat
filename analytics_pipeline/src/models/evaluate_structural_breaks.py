"""Evaluate recent structural-break risk across LiveStrat datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_all_symbols,
    get_market_futures_dataset_path,
    get_market_multimodal_dataset_path,
    get_structural_break_detail_path,
    get_structural_break_summary_path,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_SYMBOLS = get_all_symbols()
EPSILON = 1e-9

BREAK_FEATURE_SPECS = [
    {"column": "return_4h", "label": "spot_return", "kind": "volatility"},
    {"column": "volatility_20", "label": "spot_volatility", "kind": "level"},
    {"column": "volume_zscore", "label": "volume_pressure", "kind": "anomaly"},
    {"column": "funding_rate_zscore_21", "label": "funding_shift", "kind": "anomaly"},
    {"column": "open_interest_change_pct_zscore_21", "label": "open_interest_shift", "kind": "anomaly"},
    {"column": "long_short_ratio_zscore_21", "label": "positioning_shift", "kind": "level"},
    {"column": "taker_buy_sell_ratio_zscore_21", "label": "taker_flow_shift", "kind": "level"},
    {"column": "basis_rate_zscore_21", "label": "basis_shift", "kind": "level"},
    {"column": "futures_crowding_score", "label": "crowding_score", "kind": "level"},
    {"column": "market_futures_alignment_score", "label": "alignment_score", "kind": "level"},
    {"column": "effective_sentiment_score", "label": "effective_sentiment", "kind": "level"},
    {"column": "gdelt_article_count_zscore_30d", "label": "news_intensity", "kind": "anomaly"},
    {"column": "onchain_regime_score", "label": "onchain_regime", "kind": "level"},
    {"column": "exchange_netflow_zscore_30d", "label": "exchange_flow", "kind": "anomaly"},
    {"column": "multimodal_context_score", "label": "multimodal_context", "kind": "level"},
]


def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_recent_window_rows(timeframe, total_rows):
    defaults = {"1h": 24, "4h": 12, "1d": 10}
    desired = defaults.get(timeframe, 12)
    return max(6, min(desired, max(total_rows // 3, 6)))


def _resolve_baseline_window_rows(total_rows, recent_rows):
    baseline_rows = total_rows - recent_rows
    return max(18, baseline_rows)


def _load_best_available_dataset(symbol, timeframe, start_date, end_date):
    multimodal_path = get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date)
    market_futures_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)

    if Path(multimodal_path).exists():
        return pd.read_csv(multimodal_path, parse_dates=["open_time"]), "market_multimodal"
    if Path(market_futures_path).exists():
        return pd.read_csv(market_futures_path, parse_dates=["open_time"]), "market_futures"
    raise FileNotFoundError(f"No merged dataset found for {symbol} {timeframe} {start_date} {end_date}")


def _score_level_shift(baseline_series, recent_series):
    baseline_mean = baseline_series.mean()
    recent_mean = recent_series.mean()
    baseline_std = baseline_series.std(ddof=0)
    shift_score = abs(recent_mean - baseline_mean) / max(abs(baseline_std), EPSILON)
    sign_flip = int(np.sign(baseline_mean) != np.sign(recent_mean) and abs(recent_mean - baseline_mean) > max(baseline_std, 0.25))
    return shift_score, sign_flip


def _score_volatility_shift(baseline_series, recent_series):
    baseline_std = baseline_series.std(ddof=0)
    recent_std = recent_series.std(ddof=0)
    volatility_ratio = recent_std / max(abs(baseline_std), EPSILON)
    shift_score = max(volatility_ratio - 1.0, 0.0)
    return shift_score, 0


def _score_anomaly_shift(baseline_series, recent_series):
    baseline_share = float((baseline_series.abs() >= 2.0).mean())
    recent_share = float((recent_series.abs() >= 2.0).mean())
    baseline_mean = baseline_series.mean()
    recent_mean = recent_series.mean()
    shift_score = ((recent_share - baseline_share) * 3.0) + (
        abs(recent_mean - baseline_mean) / max(baseline_series.std(ddof=0), 1.0)
    )
    sign_flip = int(np.sign(baseline_mean) != np.sign(recent_mean) and abs(recent_mean) >= 1.0)
    return max(shift_score, 0.0), sign_flip


def _classify_feature_status(shift_score):
    if shift_score >= 2.5:
        return "break"
    if shift_score >= 1.25:
        return "watch"
    return "stable"


def _build_feature_break_row(symbol, timeframe, dataset_label, feature_spec, baseline_series, recent_series, recent_start, recent_end):
    kind = feature_spec["kind"]
    if kind == "volatility":
        shift_score, sign_flip = _score_volatility_shift(baseline_series, recent_series)
    elif kind == "anomaly":
        shift_score, sign_flip = _score_anomaly_shift(baseline_series, recent_series)
    else:
        shift_score, sign_flip = _score_level_shift(baseline_series, recent_series)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_family": dataset_label,
        "feature_group": feature_spec["label"],
        "feature_column": feature_spec["column"],
        "feature_kind": kind,
        "recent_window_start": recent_start,
        "recent_window_end": recent_end,
        "baseline_mean": _safe_float(baseline_series.mean()),
        "baseline_std": _safe_float(baseline_series.std(ddof=0)),
        "recent_mean": _safe_float(recent_series.mean()),
        "recent_std": _safe_float(recent_series.std(ddof=0)),
        "recent_min": _safe_float(recent_series.min()),
        "recent_max": _safe_float(recent_series.max()),
        "shift_score": _safe_float(shift_score),
        "sign_flip_flag": int(sign_flip),
        "feature_status": _classify_feature_status(shift_score),
    }


def _derive_asset_status(detail_df):
    break_count = int((detail_df["feature_status"] == "break").sum())
    watch_count = int((detail_df["feature_status"] == "watch").sum())
    feature_count = int(len(detail_df))
    avg_shift = _safe_float(detail_df["shift_score"].mean())
    max_shift = _safe_float(detail_df["shift_score"].max())
    sign_flip_count = int(detail_df["sign_flip_flag"].sum())

    score = avg_shift + (break_count * 0.7) + (watch_count * 0.25) + (sign_flip_count * 0.2)
    if break_count >= 2 or score >= 4.5 or max_shift >= 4.0:
        return "break", "reduced_trust", score
    if break_count >= 1 or watch_count >= 2 or score >= 2.5:
        return "watch", "caution", score
    return "stable", "normal", score


def _build_asset_summary(symbol, timeframe, dataset_label, source_df, detail_df, recent_rows):
    latest_row = source_df.iloc[-1]
    recent_window_df = source_df.tail(recent_rows)
    status, trust_mode, score = _derive_asset_status(detail_df)
    top_groups = detail_df.sort_values("shift_score", ascending=False)["feature_group"].head(3).tolist()
    top_groups_text = ", ".join(top_groups) if top_groups else "none"

    if status == "break":
        summary_text = (
            f"{symbol} shows a recent structural break signature on {timeframe}. "
            f"The strongest shifts are in {top_groups_text}, so LiveStrat should downgrade trust "
            "in older learned relationships until fresh evidence stabilizes."
        )
    elif status == "watch":
        summary_text = (
            f"{symbol} is in a regime-watch state on {timeframe}. "
            f"Recent pressure is building in {top_groups_text}, so signals should be read with caution."
        )
    else:
        summary_text = (
            f"{symbol} remains structurally stable on {timeframe}. "
            f"No major recent break was detected beyond normal variation."
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_family": dataset_label,
        "window_rows": int(len(source_df)),
        "recent_window_rows": int(recent_rows),
        "recent_window_start": recent_window_df["open_time"].iloc[0],
        "recent_window_end": recent_window_df["open_time"].iloc[-1],
        "latest_open_time": latest_row["open_time"],
        "current_close": _safe_float(latest_row.get("close")),
        "current_return_24h": _safe_float(latest_row.get("return_24h")),
        "current_volatility_20": _safe_float(latest_row.get("volatility_20")),
        "current_futures_crowding_score": _safe_float(latest_row.get("futures_crowding_score")),
        "current_alignment_score": _safe_float(latest_row.get("market_futures_alignment_score")),
        "current_effective_sentiment_score": _safe_float(latest_row.get("effective_sentiment_score")),
        "current_onchain_regime_score": _safe_float(latest_row.get("onchain_regime_score")),
        "structural_break_status": status,
        "trust_mode": trust_mode,
        "break_score": _safe_float(score),
        "break_feature_count": int((detail_df["feature_status"] == "break").sum()),
        "watch_feature_count": int((detail_df["feature_status"] == "watch").sum()),
        "sign_flip_feature_count": int(detail_df["sign_flip_flag"].sum()),
        "top_break_features": top_groups_text,
        "structural_break_summary": summary_text,
    }


def evaluate_structural_breaks(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Generate structural-break detail and summary snapshots for recent datasets."""
    symbols = tuple(symbols or SUPPORTED_SYMBOLS)
    detail_rows = []
    summary_rows = []

    for symbol in symbols:
        try:
            df, dataset_label = _load_best_available_dataset(symbol, timeframe, start_date, end_date)
        except FileNotFoundError:
            continue

        df = df.sort_values("open_time").reset_index(drop=True)
        if len(df) < 24:
            continue

        recent_rows = _resolve_recent_window_rows(timeframe, len(df))
        baseline_rows = _resolve_baseline_window_rows(len(df), recent_rows)
        baseline_df = df.iloc[:baseline_rows].copy()
        recent_df = df.iloc[-recent_rows:].copy()

        symbol_rows = []
        for feature_spec in BREAK_FEATURE_SPECS:
            column = feature_spec["column"]
            if column not in df.columns:
                continue

            baseline_series = pd.to_numeric(baseline_df[column], errors="coerce").dropna()
            recent_series = pd.to_numeric(recent_df[column], errors="coerce").dropna()
            if min(len(baseline_series), len(recent_series)) < 6:
                continue

            row = _build_feature_break_row(
                symbol,
                timeframe,
                dataset_label,
                feature_spec,
                baseline_series,
                recent_series,
                recent_df["open_time"].iloc[0],
                recent_df["open_time"].iloc[-1],
            )
            symbol_rows.append(row)
            detail_rows.append(row)

        if symbol_rows:
            detail_df = pd.DataFrame(symbol_rows)
            summary_rows.append(
                _build_asset_summary(
                    symbol,
                    timeframe,
                    dataset_label,
                    df,
                    detail_df,
                    recent_rows,
                )
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = get_structural_break_detail_path(timeframe, start_date, end_date)
    summary_path = get_structural_break_summary_path(timeframe, start_date, end_date)
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("structural break detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {detail_path}")
    print("structural break summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_structural_breaks()
