"""Create app-facing latest on-chain summary outputs."""

import pandas as pd

from src.config import (
    ONCHAIN_FREQUENCY,
    get_onchain_features_path,
    get_onchain_overview_path,
    get_onchain_summary_path,
    get_supported_onchain_assets,
)
from src.io_paths import ensure_dirs


def build_onchain_summary_for_asset(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    """Create the latest latest on-chain summary for one asset."""
    features_path = get_onchain_features_path(asset_symbol, frequency=frequency)
    if not features_path.exists():
        summary_df = pd.DataFrame(
            [{
                "asset_symbol": asset_symbol,
                "latest_window_end": "",
                "latest_onchain_regime_score": 0.0,
                "latest_onchain_regime_label": "unavailable",
                "latest_activity_change": 0.0,
                "latest_valuation_ratio": 0.0,
                "onchain_data_available": False,
                "latest_onchain_summary": f"On-chain data for {asset_symbol} is currently unavailable.",
            }]
        )
        output_path = get_onchain_summary_path(asset_symbol, frequency=frequency)
        summary_df.to_csv(output_path, index=False)
        return summary_df

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        summary_df = pd.DataFrame(
            [{
                "asset_symbol": asset_symbol,
                "latest_window_end": "",
                "latest_onchain_regime_score": 0.0,
                "latest_onchain_regime_label": "unavailable",
                "latest_activity_change": 0.0,
                "latest_valuation_ratio": 0.0,
                "onchain_data_available": False,
                "latest_onchain_summary": f"On-chain data for {asset_symbol} is currently unavailable.",
            }]
        )
    else:
        latest = features_df.sort_values("window_end_utc").iloc[-1]
        if not bool(latest["onchain_data_available"]):
            summary_text = f"On-chain data for {asset_symbol} is currently unavailable."
        else:
            summary_text = (
                f"On-chain conditions for {asset_symbol} are currently "
                f"{latest['onchain_regime_label']}, with regime score "
                f"{float(latest['onchain_regime_score']):.2f}."
            )

        summary_df = pd.DataFrame(
            [{
                "asset_symbol": asset_symbol,
                "latest_window_end": latest["window_end_utc"],
                "latest_onchain_regime_score": latest["onchain_regime_score"],
                "latest_onchain_regime_label": latest["onchain_regime_label"],
                "latest_activity_change": latest.get("economic_activity_change_1d", 0.0),
                "latest_valuation_ratio": latest.get("valuation_ratio", 0.0),
                "onchain_data_available": bool(latest["onchain_data_available"]),
                "latest_onchain_summary": summary_text,
            }]
        )

    output_path = get_onchain_summary_path(asset_symbol, frequency=frequency)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def build_onchain_summary_overview(frequency=ONCHAIN_FREQUENCY):
    """Build combined latest on-chain summary outputs."""
    ensure_dirs()
    rows = []
    for asset_symbol in get_supported_onchain_assets():
        rows.append(build_onchain_summary_for_asset(asset_symbol, frequency=frequency))

    overview_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    overview_path = get_onchain_overview_path(frequency=frequency)
    overview_df.to_csv(overview_path, index=False)

    print("built on-chain summary overview")
    print(f"rows saved: {len(overview_df)}")
    print(f"summary overview saved to: {overview_path}")

    return overview_df


if __name__ == "__main__":
    build_onchain_summary_overview()
