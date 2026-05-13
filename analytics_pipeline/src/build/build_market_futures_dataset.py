"""Merge spot market features with futures market-structure features."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    get_futures_features_path,
    get_market_features_path,
    get_market_futures_dataset_path,
)
from src.io_paths import ensure_dirs


SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def add_rolling_zscore(df, source_column, output_column, window=20, min_periods=5):
    """Add a rolling z-score for one numeric series if it exists."""
    if source_column not in df.columns:
        df[output_column] = pd.NA
        return df

    source = pd.to_numeric(df[source_column], errors="coerce")
    rolling_mean = source.rolling(window, min_periods=min_periods).mean()
    rolling_std = source.rolling(window, min_periods=min_periods).std()
    df[output_column] = ((source - rolling_mean) / rolling_std.replace(0, pd.NA)).fillna(0.0)
    return df


def apply_effective_basis_fallback(merged_df, timeframe):
    """Use a conservative spread-based proxy when official basis is missing."""
    merged_df["basis_feature_mode"] = "official"
    merged_df["basis_proxy_active"] = False
    merged_df["effective_basis_feature_available"] = pd.to_numeric(
        merged_df.get("basis_feature_available"), errors="coerce"
    ).fillna(0.0).astype(bool)

    basis_missing = ~merged_df["effective_basis_feature_available"]
    proxy_specs = [
        ("futures_price_spot_spread_pct", "futures_price_spot_spread_zscore_20", "proxy_futures_price_spread"),
        ("mark_spot_spread_pct", "mark_spot_spread_zscore_20", "proxy_mark_spread"),
        ("funding_mark_spot_spread_pct", "funding_mark_spot_spread_zscore_20", "proxy_funding_mark_spread"),
    ]
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, PERIODS_PER_YEAR["4h"])

    for proxy_column, proxy_zscore_column, proxy_mode in proxy_specs:
        proxy_available = merged_df[proxy_column].notna()
        use_proxy = basis_missing & proxy_available
        if not use_proxy.any():
            continue

        proxy_basis_rate = pd.to_numeric(merged_df.loc[use_proxy, proxy_column], errors="coerce")
        merged_df.loc[use_proxy, "basis_rate"] = proxy_basis_rate
        merged_df.loc[use_proxy, "basis_rate_zscore_21"] = pd.to_numeric(
            merged_df.loc[use_proxy, proxy_zscore_column], errors="coerce"
        )
        merged_df.loc[use_proxy, "basis_value"] = (
            proxy_basis_rate * pd.to_numeric(merged_df.loc[use_proxy, "close"], errors="coerce")
        )
        merged_df.loc[use_proxy, "annualized_basis_rate"] = proxy_basis_rate * periods_per_year
        merged_df.loc[use_proxy, "basis_feature_mode"] = proxy_mode
        merged_df.loc[use_proxy, "basis_proxy_active"] = True
        merged_df.loc[use_proxy, "effective_basis_feature_available"] = True
        basis_missing = basis_missing & ~use_proxy

    merged_df.loc[
        (~merged_df["effective_basis_feature_available"]) & merged_df["basis_feature_mode"].eq("official"),
        "basis_feature_mode",
    ] = "unavailable"
    return merged_df


def build_market_futures_dataset_for_symbol(symbol, timeframe=TIMEFRAME,
                                            start_date=None, end_date=None):
    """Join spot market features and futures features on aligned candle windows."""
    ensure_dirs()
    market_path = get_market_features_path(symbol, timeframe, start_date, end_date)
    futures_path = get_futures_features_path(symbol, timeframe, start_date, end_date)

    if not market_path.exists():
        raise FileNotFoundError(f"spot market feature file not found: {market_path}")
    if not futures_path.exists():
        raise FileNotFoundError(f"futures feature file not found: {futures_path}")

    market_df = pd.read_csv(market_path, parse_dates=["open_time", "close_time"])
    futures_df = pd.read_csv(futures_path, parse_dates=["open_time", "close_time"])

    market_df = market_df.sort_values("open_time").reset_index(drop=True)
    futures_df = futures_df.sort_values("open_time").reset_index(drop=True)

    merged_df = market_df.merge(
        futures_df.drop(columns=["symbol"]),
        on=["open_time", "close_time"],
        how="left",
        suffixes=("", "_futures"),
    )

    merged_df["mark_spot_spread_pct"] = (
        (pd.to_numeric(merged_df["mark_close"], errors="coerce") - pd.to_numeric(merged_df["close"], errors="coerce")) /
        pd.to_numeric(merged_df["close"], errors="coerce")
    )
    merged_df["futures_price_spot_spread_pct"] = (
        (pd.to_numeric(merged_df["futures_price"], errors="coerce") - pd.to_numeric(merged_df["close"], errors="coerce")) /
        pd.to_numeric(merged_df["close"], errors="coerce")
    )
    merged_df["funding_mark_spot_spread_pct"] = (
        (pd.to_numeric(merged_df["funding_mark_price"], errors="coerce") - pd.to_numeric(merged_df["close"], errors="coerce")) /
        pd.to_numeric(merged_df["close"], errors="coerce")
    )
    merged_df = add_rolling_zscore(merged_df, "mark_spot_spread_pct", "mark_spot_spread_zscore_20")
    merged_df = add_rolling_zscore(
        merged_df,
        "futures_price_spot_spread_pct",
        "futures_price_spot_spread_zscore_20",
    )
    merged_df = add_rolling_zscore(
        merged_df,
        "funding_mark_spot_spread_pct",
        "funding_mark_spot_spread_zscore_20",
    )
    merged_df = apply_effective_basis_fallback(merged_df, timeframe)

    merged_df["funding_oi_pressure"] = (
        pd.to_numeric(merged_df["funding_rate"], errors="coerce") *
        pd.to_numeric(merged_df["open_interest_change_pct"], errors="coerce")
    )
    merged_df["basis_momentum_agreement"] = (
        pd.to_numeric(merged_df["basis_rate"], errors="coerce") *
        pd.to_numeric(merged_df["return_24h"], errors="coerce")
    )
    merged_df["taker_pressure_return_alignment"] = (
        pd.to_numeric(merged_df["taker_buy_sell_ratio"], errors="coerce") *
        pd.to_numeric(merged_df["return_4h"], errors="coerce")
    )
    merged_df["market_futures_alignment_score"] = (
        pd.to_numeric(merged_df["futures_activity_score"], errors="coerce").fillna(0.0) +
        pd.to_numeric(merged_df["taker_buy_sell_ratio_zscore_21"], errors="coerce").fillna(0.0) +
        pd.to_numeric(merged_df["mark_spot_spread_zscore_20"], errors="coerce").fillna(0.0)
    ) / 3.0
    merged_df["futures_basis_reliance_score"] = (
        pd.to_numeric(merged_df["basis_feature_available"], errors="coerce").fillna(0.0) * 0.5
        + pd.to_numeric(merged_df["basis_rate_zscore_21"], errors="coerce").abs().fillna(0.0).clip(upper=2.0) / 4.0
    ).clip(lower=0.0, upper=1.0)
    merged_df["futures_context_resilience_score"] = (
        0.45 * pd.to_numeric(merged_df["futures_feature_completeness_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        + 0.30 * pd.to_numeric(merged_df["funding_data_available"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(merged_df["futures_structure_data_available"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    merged_df["futures_context_resilience_label"] = "partial"
    merged_df.loc[merged_df["futures_context_resilience_score"] >= 0.80, "futures_context_resilience_label"] = "robust"
    merged_df.loc[merged_df["futures_context_resilience_score"] < 0.45, "futures_context_resilience_label"] = "fragile"

    output_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
    merged_df.to_csv(output_path, index=False)

    print(f"market + futures dataset built for {symbol}")
    print(f"timeframe: {timeframe}")
    print(f"rows saved: {len(merged_df)}")
    print(f"combined dataset saved to: {output_path}")

    return merged_df


def build_market_futures_dataset():
    return build_market_futures_dataset_for_symbol(SYMBOL, TIMEFRAME, START_DATE, END_DATE)


if __name__ == "__main__":
    build_market_futures_dataset()
