"""Build aligned daily market + on-chain datasets for supported assets."""

import pandas as pd

from src.config import (
    ONCHAIN_FREQUENCY,
    PROCESSED_DIR,
    get_asset_config,
    get_market_features_path,
    get_market_onchain_dataset_path,
    get_onchain_features_path,
    get_supported_onchain_assets,
)
from src.io_paths import ensure_dirs


MARKET_DAILY_COLUMNS = [
    "close",
    "return_4h",
    "return_24h",
    "return_3d",
    "sma_20",
    "sma_50",
    "price_sma_20_diff",
    "price_sma_50_diff",
    "volatility_20",
    "volatility_50",
    "volume_zscore",
    "high_low_range_pct",
    "candle_body_pct",
    "taker_buy_volume_ratio",
    "trade_count_zscore",
]


def _market_trend_label(row):
    close = pd.to_numeric(pd.Series([row.get("market_close")]), errors="coerce").iloc[0]
    sma_20 = pd.to_numeric(pd.Series([row.get("market_sma_20")]), errors="coerce").iloc[0]
    sma_50 = pd.to_numeric(pd.Series([row.get("market_sma_50")]), errors="coerce").iloc[0]
    if pd.notna(close) and pd.notna(sma_20) and pd.notna(sma_50):
        if close > sma_20 > sma_50:
            return "bullish"
        if close < sma_20 < sma_50:
            return "bearish"
    return "mixed"


def _momentum_label(row):
    return_24h = pd.to_numeric(pd.Series([row.get("market_return_24h")]), errors="coerce").iloc[0]
    price_sma_diff = pd.to_numeric(pd.Series([row.get("market_price_sma_20_diff")]), errors="coerce").iloc[0]
    if pd.notna(return_24h) and pd.notna(price_sma_diff):
        if return_24h > 0 and price_sma_diff > 0:
            return "positive"
        if return_24h < 0 and price_sma_diff < 0:
            return "negative"
    return "neutral"


def enrich_market_onchain_dataset(df):
    """Add shared structural features so all on-chain specialists use the same backbone."""
    enriched_df = df.copy()

    enriched_df["market_trend_label"] = enriched_df.apply(_market_trend_label, axis=1)
    enriched_df["market_momentum_label"] = enriched_df.apply(_momentum_label, axis=1)
    enriched_df["market_trend_direction"] = 0.0
    enriched_df.loc[enriched_df["market_trend_label"] == "bullish", "market_trend_direction"] = 1.0
    enriched_df.loc[enriched_df["market_trend_label"] == "bearish", "market_trend_direction"] = -1.0

    regime_score = pd.to_numeric(enriched_df["onchain_regime_score"], errors="coerce").fillna(0.0)
    component_count = pd.to_numeric(enriched_df.get("onchain_component_count"), errors="coerce").fillna(0.0)
    exchange_flow_score = pd.to_numeric(enriched_df.get("exchange_flow_score"), errors="coerce").fillna(0.0)
    valuation_state_score = pd.to_numeric(enriched_df.get("valuation_state_score"), errors="coerce").fillna(0.0)
    network_activity_score = pd.to_numeric(enriched_df.get("network_activity_score"), errors="coerce").fillna(0.0)
    economic_activity_score = pd.to_numeric(enriched_df.get("economic_activity_score"), errors="coerce").fillna(0.0)

    enriched_df["onchain_support_bias"] = regime_score.clip(lower=0.0)
    enriched_df["onchain_risk_bias"] = (-regime_score).clip(lower=0.0)
    enriched_df["onchain_confidence_score"] = (
        0.60 * (component_count.clip(lower=0.0, upper=4.0) / 4.0) +
        0.40 * regime_score.abs().clip(upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_distribution_risk_score"] = (
        (-exchange_flow_score).clip(lower=0.0, upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_exchange_relief_score"] = (
        exchange_flow_score.clip(lower=0.0, upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_network_tailwind_score"] = (
        0.65 * network_activity_score.clip(lower=0.0, upper=1.5) / 1.5 +
        0.35 * economic_activity_score.clip(lower=0.0, upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_participation_breadth_score"] = (
        0.50 * network_activity_score.clip(lower=0.0, upper=1.5) / 1.5 +
        0.50 * economic_activity_score.clip(lower=0.0, upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_activity_deterioration_score"] = (
        0.50 * (-network_activity_score).clip(lower=0.0, upper=1.5) / 1.5 +
        0.50 * (-economic_activity_score).clip(lower=0.0, upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_valuation_support_score"] = valuation_state_score.clip(lower=0.0, upper=1.5) / 1.5
    enriched_df["onchain_valuation_stretch_score"] = (-valuation_state_score).clip(lower=0.0, upper=1.5) / 1.5
    enriched_df["market_onchain_alignment_score"] = (
        enriched_df["market_trend_direction"] * regime_score * enriched_df["onchain_confidence_score"]
    ).clip(lower=-1.0, upper=1.0)
    enriched_df["market_onchain_divergence_score"] = (
        (
            enriched_df["market_trend_direction"].abs() > 0
        ).astype(float)
        * (enriched_df["market_trend_direction"] * regime_score < 0).astype(float)
        * (0.60 * regime_score.abs().clip(upper=1.0) + 0.40 * enriched_df["onchain_distribution_risk_score"])
    ).clip(lower=0.0, upper=1.0)
    enriched_df["onchain_structural_fragility_score"] = (
        0.40 * enriched_df["onchain_distribution_risk_score"] +
        0.30 * enriched_df["onchain_valuation_stretch_score"] +
        0.30 * enriched_df["onchain_activity_deterioration_score"]
    ).clip(lower=0.0, upper=1.0)

    enriched_df["market_onchain_structural_label"] = "mixed"
    enriched_df.loc[
        (enriched_df["market_onchain_alignment_score"] >= 0.20) &
        (enriched_df["market_trend_label"] == "bullish"),
        "market_onchain_structural_label"
    ] = "bullish_alignment"
    enriched_df.loc[
        (enriched_df["market_onchain_alignment_score"] >= 0.20) &
        (enriched_df["market_trend_label"] == "bearish"),
        "market_onchain_structural_label"
    ] = "bearish_alignment"
    enriched_df.loc[
        enriched_df["market_onchain_divergence_score"] >= 0.25,
        "market_onchain_structural_label"
    ] = "divergence_watch"
    enriched_df.loc[
        (enriched_df["market_onchain_divergence_score"] >= 0.45) &
        (enriched_df["onchain_distribution_risk_score"] >= 0.35),
        "market_onchain_structural_label"
    ] = "distribution_risk"

    support_driver_scores = {
        "network_tailwind": enriched_df["onchain_network_tailwind_score"],
        "exchange_relief": enriched_df["onchain_exchange_relief_score"],
        "participation_breadth": enriched_df["onchain_participation_breadth_score"],
        "valuation_support": enriched_df["onchain_valuation_support_score"],
        "trend_alignment": enriched_df["market_onchain_alignment_score"].clip(lower=0.0),
    }
    risk_driver_scores = {
        "distribution_risk": enriched_df["onchain_distribution_risk_score"],
        "activity_deterioration": enriched_df["onchain_activity_deterioration_score"],
        "structural_fragility": enriched_df["onchain_structural_fragility_score"],
        "valuation_stretch": enriched_df["onchain_valuation_stretch_score"],
        "trend_divergence": enriched_df["market_onchain_divergence_score"],
    }
    support_driver_df = pd.DataFrame(support_driver_scores)
    risk_driver_df = pd.DataFrame(risk_driver_scores)
    enriched_df["onchain_primary_support_driver"] = support_driver_df.idxmax(axis=1)
    enriched_df["onchain_primary_risk_driver"] = risk_driver_df.idxmax(axis=1)
    enriched_df.loc[support_driver_df.max(axis=1) <= 0.05, "onchain_primary_support_driver"] = "none"
    enriched_df.loc[risk_driver_df.max(axis=1) <= 0.05, "onchain_primary_risk_driver"] = "none"
    enriched_df["market_onchain_driver_summary"] = (
        "support driver: "
        + enriched_df["onchain_primary_support_driver"].astype(str).str.replace("_", " ")
        + "; risk driver: "
        + enriched_df["onchain_primary_risk_driver"].astype(str).str.replace("_", " ")
    )
    return enriched_df


def _resolve_market_features_path(symbol, timeframe):
    path = get_market_features_path(symbol, timeframe)
    if path.exists():
        return path

    matches = list(PROCESSED_DIR.glob(f"{symbol}_{timeframe}_market_features_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Market features not found for {symbol} on {timeframe}.")
    return max(matches, key=lambda item: item.stat().st_mtime)


def build_daily_market_snapshot(symbol, timeframe="1d"):
    """Collapse the selected market feature table into one end-of-day snapshot per UTC day."""
    market_path = _resolve_market_features_path(symbol, timeframe)
    market_df = pd.read_csv(market_path, parse_dates=["open_time", "close_time"])
    market_df = market_df.sort_values("close_time").reset_index(drop=True)

    # align each market row to the next midnight so it matches the Coin Metrics daily timestamp
    market_df["window_end_utc"] = market_df["close_time"].dt.ceil("D")
    daily_market = market_df.groupby("window_end_utc", as_index=False).tail(1).copy()
    daily_market["window_end_utc"] = daily_market["window_end_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_columns = ["window_end_utc", "open_time", "close_time"] + MARKET_DAILY_COLUMNS
    daily_market = daily_market[output_columns].copy()
    daily_market = daily_market.rename(
        columns={column: f"market_{column}" for column in MARKET_DAILY_COLUMNS}
    )
    return daily_market.reset_index(drop=True)


def build_market_onchain_dataset_for_asset(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    """Join daily market snapshots to daily Coin Metrics on-chain features."""
    ensure_dirs()
    asset_config = get_asset_config(asset_symbol)
    market_symbol = asset_config["market_symbol"]

    market_daily = build_daily_market_snapshot(market_symbol, timeframe=frequency)
    onchain_path = get_onchain_features_path(asset_symbol, frequency=frequency)
    onchain_df = pd.read_csv(onchain_path)
    if "asset_symbol" in onchain_df.columns:
        onchain_df = onchain_df.drop(columns=["asset_symbol"])
    if not onchain_df.empty:
        onchain_df["window_end_utc"] = pd.to_datetime(onchain_df["window_end_utc"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    merged_df = market_daily.merge(
        onchain_df,
        on="window_end_utc",
        how="left",
        suffixes=("", "_onchain"),
    )
    merged_df.insert(0, "asset_symbol", asset_symbol)
    merged_df.insert(1, "market_symbol", market_symbol)
    merged_df = enrich_market_onchain_dataset(merged_df)

    output_path = get_market_onchain_dataset_path(asset_symbol, frequency=frequency)
    merged_df.to_csv(output_path, index=False)

    print(f"built market + on-chain dataset for {asset_symbol}")
    print(f"rows saved: {len(merged_df)}")
    print(f"combined dataset saved to: {output_path}")

    return merged_df


def build_market_onchain_datasets_for_all_supported_assets(frequency=ONCHAIN_FREQUENCY):
    """Build aligned daily market + on-chain datasets for all configured on-chain assets."""
    outputs = {}
    for asset_symbol in get_supported_onchain_assets():
        outputs[asset_symbol] = build_market_onchain_dataset_for_asset(
            asset_symbol,
            frequency=frequency,
        )
    return outputs


if __name__ == "__main__":
    build_market_onchain_datasets_for_all_supported_assets()
