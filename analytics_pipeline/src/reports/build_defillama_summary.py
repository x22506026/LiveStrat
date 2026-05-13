"""Build an app-facing DeFiLlama ecosystem context summary."""

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    ASSET_REGISTRY,
    DEFILLAMA_CHAIN_MAP,
    DEFILLAMA_FREQUENCY,
    get_defillama_features_path,
    get_defillama_summary_path,
)
from src.io_paths import ensure_dirs


def _safe_read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=["window_end_utc"])
    except (OSError, EmptyDataError):
        return pd.DataFrame()


def _describe_row(row):
    label = str(row.get("defi_regime_label", "unavailable") or "unavailable").replace("_", " ")
    chain = row.get("chain_name", "this chain")
    tvl = row.get("latest_defi_tvl_usd")
    change_30d = row.get("defi_tvl_change_pct_30d")
    if pd.isna(tvl):
        return f"{chain} DeFi ecosystem context is unavailable."
    return (
        f"{chain} DeFi TVL is ${tvl:,.0f}; the 30-day TVL move is {change_30d:+.1%}, "
        f"leaving the ecosystem context {label}."
    )


def build_defillama_summary(frequency=DEFILLAMA_FREQUENCY):
    """Summarize the latest DeFiLlama feature row per configured asset."""
    ensure_dirs()
    rows = []
    for asset_symbol, chain_name in DEFILLAMA_CHAIN_MAP.items():
        market_symbol = ASSET_REGISTRY[asset_symbol]["market_symbol"]
        features_df = _safe_read_csv(get_defillama_features_path(asset_symbol, frequency))
        if features_df.empty:
            row = {
                "asset_symbol": asset_symbol,
                "symbol": market_symbol,
                "chain_name": chain_name,
                "latest_defi_window_end": pd.NA,
                "latest_defi_tvl_usd": pd.NA,
                "defi_tvl_change_pct_1d": pd.NA,
                "defi_tvl_change_pct_7d": pd.NA,
                "defi_tvl_change_pct_30d": pd.NA,
                "defi_tvl_zscore_30d": pd.NA,
                "defi_tvl_drawdown_30d": pd.NA,
                "defi_regime_score": pd.NA,
                "defi_regime_label": "unavailable",
                "defi_context_available": False,
            }
        else:
            latest = features_df.sort_values("window_end_utc").iloc[-1]
            row = {
                "asset_symbol": asset_symbol,
                "symbol": market_symbol,
                "chain_name": chain_name,
                "latest_defi_window_end": latest["window_end_utc"],
                "latest_defi_tvl_usd": latest.get("defi_tvl_usd"),
                "defi_tvl_change_pct_1d": latest.get("defi_tvl_change_pct_1d"),
                "defi_tvl_change_pct_7d": latest.get("defi_tvl_change_pct_7d"),
                "defi_tvl_change_pct_30d": latest.get("defi_tvl_change_pct_30d"),
                "defi_tvl_zscore_30d": latest.get("defi_tvl_zscore_30d"),
                "defi_tvl_drawdown_30d": latest.get("defi_tvl_drawdown_30d"),
                "defi_regime_score": latest.get("defi_regime_score"),
                "defi_regime_label": latest.get("defi_regime_label", "unavailable"),
                "defi_context_available": bool(latest.get("defi_context_available", False)),
            }
        row["defi_summary"] = _describe_row(row)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    output_path = get_defillama_summary_path(frequency)
    summary_df.to_csv(output_path, index=False)
    print("DeFiLlama summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {output_path}")
    return summary_df


if __name__ == "__main__":
    build_defillama_summary()
