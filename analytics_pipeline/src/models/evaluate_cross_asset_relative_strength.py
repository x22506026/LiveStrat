"""Rank supported assets with a cross-sectional relative-strength strategy."""

from pathlib import Path

import pandas as pd

from src.config import (
    ASSET_REGISTRY,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    PROCESSED_DIR,
    get_cross_asset_relative_strength_summary_path,
    get_labeled_market_path,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def _latest_file(pattern):
    matches = [path for path in PROCESSED_DIR.glob(pattern) if "labeled" in path.name]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _resolve_labeled_market_path(symbol, timeframe, start_date=None, end_date=None):
    dated_path = get_labeled_market_path(symbol, timeframe, start_date, end_date)
    if dated_path.exists():
        return dated_path
    return _latest_file(f"{symbol}_{timeframe}_market_features_labeled_*.csv")


def _asset_code(symbol):
    for code, config in ASSET_REGISTRY.items():
        if config["market_symbol"] == symbol:
            return code
    return symbol.replace("USDT", "")


def _market_supported_symbols():
    symbols = []
    for asset_config in ASSET_REGISTRY.values():
        if asset_config.get("support_flags", {}).get("market_supported"):
            symbols.append(asset_config["market_symbol"])
    return tuple(dict.fromkeys(symbols))


def _rank_series(values, ascending=False):
    return values.rank(pct=True, ascending=ascending, method="average").fillna(0.5)


def _score_cross_section(frame):
    scored = frame.copy()
    scored["momentum_score"] = (
        _rank_series(scored["return_24h"], ascending=False) * 0.55 +
        _rank_series(scored["return_3d"], ascending=False) * 0.45
    )
    scored["trend_score"] = _rank_series(scored["price_sma_20_diff"], ascending=False)
    scored["activity_score"] = (
        _rank_series(scored["volume_zscore"], ascending=False) * 0.6 +
        _rank_series(scored["trade_count_zscore"], ascending=False) * 0.4
    )
    scored["risk_adjustment_score"] = _rank_series(scored["volatility_20"], ascending=True)
    scored["relative_strength_score"] = (
        scored["momentum_score"] * 0.40 +
        scored["trend_score"] * 0.25 +
        scored["activity_score"] * 0.20 +
        scored["risk_adjustment_score"] * 0.15
    )
    scored["cross_asset_rank"] = scored["relative_strength_score"].rank(ascending=False, method="first").astype(int)
    scored["asset_count"] = len(scored)
    scored["relative_strength_signal"] = scored["cross_asset_rank"].map(
        lambda rank: "top_candidate" if rank == 1 else ("watchlist" if rank <= 3 else "laggard")
    )
    return scored


def _load_symbol_frame(symbol, timeframe, start_date=None, end_date=None):
    path = _resolve_labeled_market_path(symbol, timeframe, start_date, end_date)
    if path is None or not path.exists():
        return pd.DataFrame(), None

    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    if df.empty:
        return pd.DataFrame(), path

    needed_columns = [
        "open_time",
        "close_time",
        "close",
        "future_return",
        "return_24h",
        "return_3d",
        "price_sma_20_diff",
        "volatility_20",
        "volume_zscore",
        "trade_count_zscore",
    ]
    available = [column for column in needed_columns if column in df.columns]
    df = df[available].copy()
    for column in needed_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["symbol"] = symbol
    df["asset"] = _asset_code(symbol)
    df["source_file"] = path.name
    return df.sort_values("close_time").reset_index(drop=True), path


def evaluate_cross_asset_relative_strength(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Build a relative-strength summary using every asset with matching market features."""
    symbols = tuple(symbols or _market_supported_symbols())
    frames = []
    source_files = {}

    for symbol in symbols:
        symbol_df, path = _load_symbol_frame(symbol, timeframe, start_date, end_date)
        if symbol_df.empty:
            continue
        frames.append(symbol_df)
        source_files[symbol] = path.name if path else ""

    if len(frames) < 2:
        raise ValueError("Cross-asset relative strength needs at least two assets with market feature data.")

    universe_df = pd.concat(frames, ignore_index=True)
    score_frames = []
    for _, window_df in universe_df.groupby("close_time"):
        if len(window_df) >= 2:
            score_frames.append(_score_cross_section(window_df))

    scored_df = pd.concat(score_frames, ignore_index=True)
    scored_df["future_return"] = pd.to_numeric(scored_df["future_return"], errors="coerce")
    scored_df["universe_median_future_return"] = scored_df.groupby("close_time")["future_return"].transform("median")
    scored_df["beat_universe_median"] = scored_df["future_return"] > scored_df["universe_median_future_return"]
    scored_df["top_candidate_flag"] = scored_df["cross_asset_rank"] == 1
    latest_time = scored_df["close_time"].max()
    latest_df = scored_df[scored_df["close_time"] == latest_time].copy()

    summary_rows = []
    for _, latest in latest_df.sort_values("cross_asset_rank").iterrows():
        asset_history = scored_df[scored_df["symbol"] == latest["symbol"]]
        top_history = asset_history[asset_history["top_candidate_flag"]].copy()
        top_pick_hit_rate = float(top_history["beat_universe_median"].mean()) if not top_history.empty else 0.0
        avg_excess_forward_return = float(
            (top_history["future_return"] - top_history["universe_median_future_return"]).mean()
        ) if not top_history.empty else 0.0

        summary_rows.append(
            {
                "symbol": latest["symbol"],
                "asset": latest["asset"],
                "timeframe": timeframe,
                "window_start": start_date,
                "window_end": end_date,
                "latest_close_time": latest_time.isoformat(),
                "latest_close": float(latest["close"]),
                "cross_asset_rank": int(latest["cross_asset_rank"]),
                "asset_count": int(latest["asset_count"]),
                "relative_strength_score": float(latest["relative_strength_score"]),
                "momentum_score": float(latest["momentum_score"]),
                "trend_score": float(latest["trend_score"]),
                "activity_score": float(latest["activity_score"]),
                "risk_adjustment_score": float(latest["risk_adjustment_score"]),
                "relative_strength_signal": latest["relative_strength_signal"],
                "historical_top_pick_count": int(len(top_history)),
                "top_pick_hit_rate": top_pick_hit_rate,
                "avg_excess_forward_return": avg_excess_forward_return,
                "coverage": "available",
                "source_file": source_files.get(latest["symbol"], latest["source_file"]),
                "relative_strength_summary": (
                    f"{latest['symbol']} ranks {int(latest['cross_asset_rank'])} of {int(latest['asset_count'])} "
                    f"on cross-asset relative strength with a {latest['relative_strength_signal']} signal."
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    output_path = get_cross_asset_relative_strength_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(output_path, index=False)

    print("cross-asset relative strength summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {output_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_cross_asset_relative_strength()
