"""Create latest combined market + on-chain summaries for supported assets."""

import pandas as pd

from src.config import (
    ONCHAIN_FREQUENCY,
    get_market_onchain_dataset_path,
    get_market_onchain_overview_path,
    get_market_onchain_summary_path,
    get_supported_onchain_assets,
)
from src.io_paths import ensure_dirs


def classify_combined_view(row):
    """Translate market trend and on-chain regime into a simple combined view."""
    structural_label = row.get("market_onchain_structural_label", "mixed")
    if pd.notna(structural_label):
        structural_label = str(structural_label)
        if structural_label == "bullish_alignment":
            return "aligned_bullish"
        if structural_label == "bearish_alignment":
            return "aligned_bearish"
        if structural_label == "distribution_risk":
            return "distribution_risk"
        if structural_label == "divergence_watch":
            return "divergence_watch"

    market_bullish = bool(row["market_close"] > row["market_sma_20"] > row["market_sma_50"])
    market_bearish = bool(row["market_close"] < row["market_sma_20"] < row["market_sma_50"])
    onchain_label = row.get("onchain_regime_label", "unavailable")
    if pd.isna(onchain_label):
        onchain_label = "unavailable"

    if market_bullish and onchain_label == "supportive":
        return "aligned_bullish"
    if market_bearish and onchain_label == "weakening":
        return "aligned_bearish"
    if onchain_label == "unavailable":
        return "market_only"
    return "mixed"


def build_market_onchain_summary_for_asset(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    """Create the latest combined market + on-chain summary for one asset."""
    dataset_path = get_market_onchain_dataset_path(asset_symbol, frequency=frequency)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Combined market + on-chain dataset not found: {dataset_path}")

    dataset_df = pd.read_csv(dataset_path)
    latest = dataset_df.sort_values("window_end_utc").iloc[-1]
    combined_view = classify_combined_view(latest)
    raw_onchain_available = latest.get("onchain_data_available", False)
    onchain_available = bool(pd.notna(raw_onchain_available) and raw_onchain_available is True)
    latest_onchain_label = latest.get("onchain_regime_label", "unavailable")
    if pd.isna(latest_onchain_label):
        latest_onchain_label = "unavailable"
    latest_onchain_score = latest.get("onchain_regime_score", 0.0)
    if pd.isna(latest_onchain_score):
        latest_onchain_score = 0.0
    latest_onchain_reason = latest.get("onchain_regime_reason", "on-chain structure is mixed")
    if pd.isna(latest_onchain_reason):
        latest_onchain_reason = "on-chain structure is mixed"
    support_driver = str(latest.get("onchain_primary_support_driver", "none") or "none")
    risk_driver = str(latest.get("onchain_primary_risk_driver", "none") or "none")
    driver_summary = str(latest.get("market_onchain_driver_summary", "") or "").strip()

    if onchain_available:
        summary_text = (
            f"{asset_symbol} currently shows a {combined_view.replace('_', ' ')} view. "
            f"The latest market 24h return is {float(latest['market_return_24h']) * 100:.2f}% "
            f"and the on-chain regime is {latest_onchain_label}. "
            f"Reason: {latest_onchain_reason}. Primary support driver is {support_driver.replace('_', ' ')}, "
            f"primary risk driver is {risk_driver.replace('_', ' ')}, participation breadth is "
            f"{float(latest.get('onchain_participation_breadth_score', 0.0)):.2f}, and structural fragility is "
            f"{float(latest.get('onchain_structural_fragility_score', 0.0)):.2f}."
        )
    else:
        summary_text = (
            f"{asset_symbol} currently falls back to market-only interpretation because "
            "on-chain data is unavailable."
        )

    summary_df = pd.DataFrame(
        [{
            "asset_symbol": asset_symbol,
            "market_symbol": latest["market_symbol"],
            "latest_window_end": latest["window_end_utc"],
            "latest_market_close": latest["market_close"],
            "latest_market_return_24h_pct": float(latest["market_return_24h"]) * 100,
            "latest_market_volatility_20": latest["market_volatility_20"],
            "latest_onchain_regime_label": latest_onchain_label,
            "latest_onchain_regime_score": latest_onchain_score,
            "latest_onchain_regime_reason": latest_onchain_reason,
            "latest_onchain_confidence_score": latest.get("onchain_confidence_score", 0.0),
            "latest_onchain_participation_breadth_score": latest.get("onchain_participation_breadth_score", 0.0),
            "latest_onchain_structural_fragility_score": latest.get("onchain_structural_fragility_score", 0.0),
            "latest_market_onchain_alignment_score": latest.get("market_onchain_alignment_score", 0.0),
            "latest_market_onchain_divergence_score": latest.get("market_onchain_divergence_score", 0.0),
            "latest_market_onchain_structural_label": latest.get("market_onchain_structural_label", "mixed"),
            "latest_onchain_primary_support_driver": support_driver,
            "latest_onchain_primary_risk_driver": risk_driver,
            "latest_market_onchain_driver_summary": driver_summary,
            "onchain_data_available": onchain_available,
            "combined_view": combined_view,
            "latest_market_onchain_summary": summary_text,
        }]
    )

    output_path = get_market_onchain_summary_path(asset_symbol, frequency=frequency)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def build_market_onchain_summary_overview(frequency=ONCHAIN_FREQUENCY):
    """Build combined overview rows for all supported on-chain assets."""
    ensure_dirs()
    rows = []
    for asset_symbol in get_supported_onchain_assets():
        rows.append(build_market_onchain_summary_for_asset(asset_symbol, frequency=frequency))

    overview_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    overview_path = get_market_onchain_overview_path(frequency=frequency)
    overview_df.to_csv(overview_path, index=False)

    print("built market + on-chain summary overview")
    print(f"rows saved: {len(overview_df)}")
    print(f"summary overview saved to: {overview_path}")

    return overview_df


if __name__ == "__main__":
    build_market_onchain_summary_overview()
