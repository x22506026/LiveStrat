"""Build DeFiLlama chain TVL features as ecosystem context."""

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    ASSET_REGISTRY,
    DEFILLAMA_CHAIN_MAP,
    DEFILLAMA_FREQUENCY,
    get_defillama_features_path,
    get_raw_defillama_chain_tvl_path,
)
from src.io_paths import ensure_dirs


def _rolling_zscore(series, window=30, min_periods=5):
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    return ((series - rolling_mean) / rolling_std.replace(0, pd.NA)).fillna(0.0)


def _label_defi_regime(row):
    tvl_change_30d = pd.to_numeric(pd.Series([row.get("defi_tvl_change_pct_30d", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
    tvl_zscore = pd.to_numeric(pd.Series([row.get("defi_tvl_zscore_30d", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
    if tvl_change_30d >= 0.10 and tvl_zscore >= -0.25:
        return "expanding"
    if tvl_change_30d <= -0.10 and tvl_zscore <= 0.25:
        return "contracting"
    return "stable"


def _score_defi_regime(df):
    tvl_momentum = pd.to_numeric(df["defi_tvl_change_pct_30d"], errors="coerce").fillna(0.0).clip(-0.35, 0.35) / 0.35
    short_momentum = pd.to_numeric(df["defi_tvl_change_pct_7d"], errors="coerce").fillna(0.0).clip(-0.20, 0.20) / 0.20
    zscore = pd.to_numeric(df["defi_tvl_zscore_30d"], errors="coerce").fillna(0.0).clip(-2.0, 2.0) / 2.0
    return (0.50 * tvl_momentum + 0.30 * short_momentum + 0.20 * zscore).clip(-1.0, 1.0)


def _empty_feature_frame(asset_symbol, chain_name):
    return pd.DataFrame(
        columns=[
            "asset_symbol",
            "market_symbol",
            "chain_name",
            "window_end_utc",
            "defi_tvl_usd",
            "defi_tvl_change_1d",
            "defi_tvl_change_pct_1d",
            "defi_tvl_change_pct_7d",
            "defi_tvl_change_pct_30d",
            "defi_tvl_zscore_30d",
            "defi_tvl_drawdown_30d",
            "defi_regime_score",
            "defi_regime_label",
            "defi_context_available",
        ]
    )


def build_defillama_features_for_asset(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    """Convert raw DeFiLlama TVL rows into daily ecosystem context features."""
    ensure_dirs()
    chain_name = DEFILLAMA_CHAIN_MAP[asset_symbol]
    market_symbol = ASSET_REGISTRY[asset_symbol]["market_symbol"]
    raw_path = get_raw_defillama_chain_tvl_path(asset_symbol, frequency)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw DeFiLlama data not found: {raw_path}")

    try:
        raw_df = pd.read_csv(raw_path)
    except EmptyDataError:
        raw_df = pd.DataFrame()

    if raw_df.empty or "date" not in raw_df.columns or "tvl" not in raw_df.columns:
        features_df = _empty_feature_frame(asset_symbol, chain_name)
    else:
        features_df = pd.DataFrame(
            {
                "asset_symbol": asset_symbol,
                "market_symbol": market_symbol,
                "chain_name": chain_name,
                "window_end_utc": pd.to_datetime(raw_df["date"], unit="s", utc=True),
                "defi_tvl_usd": pd.to_numeric(raw_df["tvl"], errors="coerce"),
            }
        ).dropna(subset=["window_end_utc"]).sort_values("window_end_utc").reset_index(drop=True)

        features_df["defi_tvl_change_1d"] = features_df["defi_tvl_usd"].diff()
        features_df["defi_tvl_change_pct_1d"] = features_df["defi_tvl_usd"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf],
            pd.NA,
        )
        features_df["defi_tvl_change_pct_7d"] = features_df["defi_tvl_usd"].pct_change(
            periods=7,
            fill_method=None,
        ).replace([np.inf, -np.inf], pd.NA)
        features_df["defi_tvl_change_pct_30d"] = features_df["defi_tvl_usd"].pct_change(
            periods=30,
            fill_method=None,
        ).replace([np.inf, -np.inf], pd.NA)
        features_df["defi_tvl_zscore_30d"] = _rolling_zscore(features_df["defi_tvl_usd"])
        rolling_max = features_df["defi_tvl_usd"].rolling(30, min_periods=5).max()
        features_df["defi_tvl_drawdown_30d"] = (
            features_df["defi_tvl_usd"] / rolling_max.replace(0, pd.NA) - 1.0
        ).replace([np.inf, -np.inf], pd.NA).fillna(0.0)
        features_df["defi_regime_score"] = _score_defi_regime(features_df)
        features_df["defi_regime_label"] = features_df.apply(_label_defi_regime, axis=1)
        features_df["defi_context_available"] = features_df["defi_tvl_usd"].notna()
        features_df["window_end_utc"] = features_df["window_end_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = get_defillama_features_path(asset_symbol, frequency)
    features_df.to_csv(output_path, index=False)
    print(f"built DeFiLlama features for {asset_symbol}")
    print(f"rows saved: {len(features_df)}")
    print(f"processed features saved to: {output_path}")
    return features_df


def build_defillama_features_for_assets(asset_symbols=None, frequency=DEFILLAMA_FREQUENCY):
    """Build DeFiLlama features for all configured assets by default."""
    outputs = {}
    for asset_symbol in asset_symbols or list(DEFILLAMA_CHAIN_MAP):
        outputs[asset_symbol] = build_defillama_features_for_asset(asset_symbol, frequency=frequency)
    return outputs


if __name__ == "__main__":
    build_defillama_features_for_assets()
