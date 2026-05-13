"""Build processed daily on-chain features from Coin Metrics raw data."""

import math

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    ONCHAIN_FREQUENCY,
    get_onchain_features_path,
    get_raw_onchain_path,
    get_supported_onchain_assets,
)
from src.io_paths import ensure_dirs


RAW_TO_FEATURE_COLUMNS = {
    "AdrActCnt": "active_addresses_value",
    "TxCnt": "transaction_count_value",
    "TxTfrValAdjUSD": "economic_activity_value",
    "CapMrktCurUSD": "market_value_value",
    "CapMVRVCur": "valuation_ratio",
    "FlowInExUSD": "exchange_inflow_value",
    "FlowOutExUSD": "exchange_outflow_value",
}


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _rolling_zscore(series, window=30, min_periods=5):
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    return ((series - rolling_mean) / rolling_std.replace(0, pd.NA)).fillna(0.0)


def _bounded_score(series, divisor=2.0):
    return (series / divisor).clip(-1.5, 1.5)


def _valuation_score(series):
    score = pd.Series(0.0, index=series.index)
    score = score.mask(series >= 2.4, -1.25)
    score = score.mask((series >= 1.8) & (series < 2.4), -0.5)
    score = score.mask((series >= 1.2) & (series < 1.8), 0.25)
    score = score.mask((series > 0) & (series < 1.2), 0.75)
    return score.fillna(0.0)


def _label_regime(score):
    if score >= 0.35:
        return "supportive"
    if score <= -0.35:
        return "weakening"
    return "neutral"


def _describe_regime(row):
    component_descriptions = []

    if row.get("network_activity_score", 0.0) >= 0.35:
        component_descriptions.append("network activity is strengthening")
    elif row.get("network_activity_score", 0.0) <= -0.35:
        component_descriptions.append("network activity is weakening")

    if row.get("economic_activity_score", 0.0) >= 0.35:
        component_descriptions.append("economic transfer activity is improving")
    elif row.get("economic_activity_score", 0.0) <= -0.35:
        component_descriptions.append("economic transfer activity is softening")

    if row.get("valuation_state_score", 0.0) >= 0.35:
        component_descriptions.append("valuation looks relatively supportive")
    elif row.get("valuation_state_score", 0.0) <= -0.35:
        component_descriptions.append("valuation looks stretched")

    if row.get("exchange_flow_score", 0.0) >= 0.35:
        component_descriptions.append("exchange flow pressure is supportive")
    elif row.get("exchange_flow_score", 0.0) <= -0.35:
        component_descriptions.append("exchange flow pressure is risk-off")

    if not component_descriptions:
        return "on-chain structure is mixed"

    return "; ".join(component_descriptions)


def build_onchain_features_for_asset(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    """Convert raw Coin Metrics time series into aligned daily on-chain features."""
    ensure_dirs()
    raw_path = get_raw_onchain_path(asset_symbol, frequency=frequency)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw Coin Metrics data not found: {raw_path}")

    def empty_features_dataframe():
        return pd.DataFrame(
            columns=[
                "asset_symbol",
                "window_end_utc",
                "active_addresses_value",
                "active_addresses_change_1d",
                "active_addresses_zscore_30d",
                "transaction_count_value",
                "transaction_count_change_1d",
                "transaction_count_zscore_30d",
                "economic_activity_value",
                "economic_activity_change_1d",
                "economic_activity_zscore_30d",
                "market_value_value",
                "exchange_inflow_value",
                "exchange_outflow_value",
                "exchange_netflow_value",
                "address_to_tx_ratio",
                "economic_value_to_market_cap_ratio",
                "valuation_ratio",
                "valuation_ratio_change_1d",
                "active_addresses_change_pct_1d",
                "transaction_count_change_pct_1d",
                "economic_activity_change_pct_1d",
                "exchange_netflow_zscore_30d",
                "network_activity_score",
                "economic_activity_score",
                "valuation_state_score",
                "exchange_flow_score",
                "onchain_component_count",
                "onchain_regime_score",
                "onchain_regime_label",
                "onchain_regime_reason",
                "onchain_data_available",
            ]
        )

    try:
        raw_df = pd.read_csv(raw_path)
    except EmptyDataError:
        raw_df = pd.DataFrame()

    if raw_df.empty:
        features_df = empty_features_dataframe()
    else:
        features_df = raw_df[["asset_symbol", "window_end_utc"]].copy()
        for raw_column, feature_column in RAW_TO_FEATURE_COLUMNS.items():
            if raw_column in raw_df.columns:
                features_df[feature_column] = _safe_numeric(raw_df[raw_column])
            else:
                features_df[feature_column] = math.nan

        features_df = features_df.sort_values("window_end_utc").reset_index(drop=True)

        for base_name in [
            "active_addresses",
            "transaction_count",
            "economic_activity",
        ]:
            value_column = f"{base_name}_value"
            features_df[f"{base_name}_change_1d"] = features_df[value_column].diff()
            features_df[f"{base_name}_change_pct_1d"] = (
                features_df[value_column].pct_change(fill_method=None).replace([math.inf, -math.inf], pd.NA).fillna(0.0)
            )
            features_df[f"{base_name}_zscore_30d"] = _rolling_zscore(features_df[value_column])

        features_df["exchange_netflow_value"] = (
            features_df["exchange_inflow_value"].fillna(0.0) -
            features_df["exchange_outflow_value"].fillna(0.0)
        )
        features_df["valuation_ratio_change_1d"] = features_df["valuation_ratio"].diff().fillna(0.0)
        features_df["exchange_netflow_zscore_30d"] = _rolling_zscore(features_df["exchange_netflow_value"])
        features_df["address_to_tx_ratio"] = (
            features_df["active_addresses_value"] /
            features_df["transaction_count_value"].replace(0, pd.NA)
        ).replace([math.inf, -math.inf], pd.NA)
        features_df["economic_value_to_market_cap_ratio"] = (
            features_df["economic_activity_value"] /
            features_df["market_value_value"].replace(0, pd.NA)
        ).replace([math.inf, -math.inf], pd.NA)

        features_df["network_activity_score"] = (
            0.6 * _bounded_score(features_df["active_addresses_zscore_30d"].fillna(0.0)) +
            0.4 * _bounded_score(features_df["transaction_count_zscore_30d"].fillna(0.0))
        )
        features_df["economic_activity_score"] = _bounded_score(
            features_df["economic_activity_zscore_30d"].fillna(0.0)
        )
        features_df["valuation_state_score"] = _valuation_score(features_df["valuation_ratio"])
        features_df["exchange_flow_score"] = _bounded_score(
            -features_df["exchange_netflow_zscore_30d"].fillna(0.0)
        )

        component_columns = [
            ("network_activity_score", ["active_addresses_value", "transaction_count_value"]),
            ("economic_activity_score", ["economic_activity_value"]),
            ("valuation_state_score", ["valuation_ratio"]),
            ("exchange_flow_score", ["exchange_inflow_value", "exchange_outflow_value"]),
        ]

        for component_name, source_columns in component_columns:
            features_df[f"{component_name}_available"] = features_df[source_columns].notna().any(axis=1)

        weighted_sum = (
            features_df["network_activity_score"] * features_df["network_activity_score_available"].astype(int) +
            features_df["economic_activity_score"] * features_df["economic_activity_score_available"].astype(int) +
            features_df["valuation_state_score"] * features_df["valuation_state_score_available"].astype(int) +
            features_df["exchange_flow_score"] * features_df["exchange_flow_score_available"].astype(int)
        )
        features_df["onchain_component_count"] = (
            features_df["network_activity_score_available"].astype(int) +
            features_df["economic_activity_score_available"].astype(int) +
            features_df["valuation_state_score_available"].astype(int) +
            features_df["exchange_flow_score_available"].astype(int)
        )
        features_df["onchain_regime_score"] = (
            weighted_sum / features_df["onchain_component_count"].replace(0, pd.NA)
        ).fillna(0.0)
        features_df["onchain_regime_label"] = features_df["onchain_regime_score"].apply(_label_regime)
        features_df["onchain_regime_reason"] = features_df.apply(_describe_regime, axis=1)
        required_columns = ["active_addresses_value", "transaction_count_value", "valuation_ratio"]
        features_df["onchain_data_available"] = features_df[required_columns].notna().any(axis=1)
        features_df.loc[~features_df["onchain_data_available"], "onchain_regime_label"] = "unavailable"
        features_df.loc[~features_df["onchain_data_available"], "onchain_regime_reason"] = "on-chain data is unavailable"

    output_path = get_onchain_features_path(asset_symbol, frequency=frequency)
    features_df.to_csv(output_path, index=False)

    print(f"built on-chain features for {asset_symbol}")
    print(f"rows saved: {len(features_df)}")
    print(f"processed features saved to: {output_path}")

    return features_df


def build_onchain_features_for_all_supported_assets(frequency=ONCHAIN_FREQUENCY):
    """Build on-chain features for all configured on-chain assets."""
    outputs = {}
    for asset_symbol in get_supported_onchain_assets():
        outputs[asset_symbol] = build_onchain_features_for_asset(asset_symbol, frequency=frequency)
    return outputs


if __name__ == "__main__":
    build_onchain_features_for_all_supported_assets()
