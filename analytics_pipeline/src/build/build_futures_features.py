"""Build processed Binance futures market-structure features."""

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    FUTURES_CONTRACT_TYPE,
    get_futures_features_path,
    get_raw_futures_basis_path,
    get_raw_futures_funding_rate_path,
    get_raw_futures_long_short_ratio_path,
    get_raw_futures_mark_price_path,
    get_raw_futures_open_interest_path,
    get_raw_futures_taker_volume_path,
)
from src.io_paths import ensure_dirs


SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def add_rolling_zscore(df, source_column, output_column, window=21, min_periods=5):
    """Add a rolling z-score for one numeric column."""
    rolling_mean = df[source_column].rolling(window, min_periods=min_periods).mean()
    rolling_std = df[source_column].rolling(window, min_periods=min_periods).std()
    df[output_column] = (
        (df[source_column] - rolling_mean) / rolling_std.replace(0, pd.NA)
    ).fillna(0.0)
    return df


def read_optional_csv(path, parse_dates=None):
    """Read a CSV if it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
    except EmptyDataError:
        return pd.DataFrame()
    for column in parse_dates or []:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df


def build_futures_features_for_symbol(symbol, timeframe, start_date, end_date,
                                      contract_type=FUTURES_CONTRACT_TYPE):
    """Create aligned futures features from raw Binance futures datasets."""
    ensure_dirs()

    mark_price_path = get_raw_futures_mark_price_path(symbol, timeframe, start_date, end_date)
    if not mark_price_path.exists():
        raise FileNotFoundError(
            f"raw futures mark-price file not found: {mark_price_path}. "
            "run the Binance futures audit step first."
        )

    mark_df = pd.read_csv(mark_price_path, parse_dates=["open_time", "close_time"])
    mark_df["open_time"] = pd.to_datetime(mark_df["open_time"], utc=True, errors="coerce")
    mark_df["close_time"] = pd.to_datetime(mark_df["close_time"], utc=True, errors="coerce")
    mark_df = mark_df.sort_values("open_time").reset_index(drop=True)
    mark_df["mark_return"] = mark_df["close"].pct_change()

    periods_per_24h = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)
    periods_per_3d = periods_per_24h * 3
    mark_df["mark_return_24h"] = mark_df["close"].pct_change(periods=periods_per_24h)
    mark_df["mark_return_3d"] = mark_df["close"].pct_change(periods=periods_per_3d)

    features_df = pd.DataFrame(
        {
            "symbol": symbol,
            "open_time": mark_df["open_time"],
            "close_time": mark_df["close_time"],
            "mark_open": pd.to_numeric(mark_df["open"], errors="coerce"),
            "mark_high": pd.to_numeric(mark_df["high"], errors="coerce"),
            "mark_low": pd.to_numeric(mark_df["low"], errors="coerce"),
            "mark_close": pd.to_numeric(mark_df["close"], errors="coerce"),
            "mark_return": mark_df["mark_return"],
            "mark_return_24h": mark_df["mark_return_24h"],
            "mark_return_3d": mark_df["mark_return_3d"],
            "futures_number_of_trades": pd.to_numeric(mark_df["number_of_trades"], errors="coerce"),
        }
    )

    funding_path = get_raw_futures_funding_rate_path(symbol, start_date, end_date)
    funding_df = read_optional_csv(funding_path, parse_dates=["funding_time"])
    if not funding_df.empty:
        funding_df = funding_df.sort_values("funding_time").reset_index(drop=True)
        funding_df = funding_df.dropna(subset=["funding_time"]).reset_index(drop=True)
        funding_df["funding_rate"] = pd.to_numeric(funding_df["funding_rate"], errors="coerce")
        funding_df["funding_rate_change"] = funding_df["funding_rate"].diff()
        funding_df["funding_rate_rolling_mean_21"] = (
            funding_df["funding_rate"].rolling(21, min_periods=5).mean()
        )
        funding_df = add_rolling_zscore(funding_df, "funding_rate", "funding_rate_zscore_21")
        funding_df["funding_mark_price"] = pd.to_numeric(funding_df["mark_price"], errors="coerce")
        features_df = pd.merge_asof(
            features_df.sort_values("close_time"),
            funding_df[
                [
                    "funding_time",
                    "funding_rate",
                    "funding_rate_change",
                    "funding_rate_rolling_mean_21",
                    "funding_rate_zscore_21",
                    "funding_mark_price",
                ]
            ].sort_values("funding_time"),
            left_on="close_time",
            right_on="funding_time",
            direction="backward",
        )
        features_df = features_df.drop(columns=["funding_time"])
    else:
        features_df["funding_rate"] = pd.NA
        features_df["funding_rate_change"] = pd.NA
        features_df["funding_rate_rolling_mean_21"] = pd.NA
        features_df["funding_rate_zscore_21"] = pd.NA
        features_df["funding_mark_price"] = pd.NA
    features_df["funding_feature_available"] = features_df["funding_rate"].notna()

    open_interest_path = get_raw_futures_open_interest_path(symbol, timeframe, start_date, end_date)
    open_interest_df = read_optional_csv(open_interest_path, parse_dates=["event_time"])
    if not open_interest_df.empty:
        open_interest_df = open_interest_df.sort_values("event_time").reset_index(drop=True)
        open_interest_df = open_interest_df.dropna(subset=["event_time"]).reset_index(drop=True)
        for column in ["open_interest", "open_interest_value", "circulating_supply"]:
            if column in open_interest_df.columns:
                open_interest_df[column] = pd.to_numeric(open_interest_df[column], errors="coerce")
        open_interest_df["open_interest_change_pct"] = open_interest_df["open_interest"].pct_change()
        open_interest_df = add_rolling_zscore(
            open_interest_df,
            "open_interest_value",
            "open_interest_value_zscore_21",
        )
        open_interest_df = add_rolling_zscore(
            open_interest_df,
            "open_interest_change_pct",
            "open_interest_change_pct_zscore_21",
        )
        features_df = pd.merge_asof(
            features_df.sort_values("close_time"),
            open_interest_df[
                [
                    "event_time",
                    "open_interest",
                    "open_interest_value",
                    "circulating_supply",
                    "open_interest_change_pct",
                    "open_interest_value_zscore_21",
                    "open_interest_change_pct_zscore_21",
                ]
            ].sort_values("event_time"),
            left_on="close_time",
            right_on="event_time",
            direction="backward",
        )
        features_df = features_df.drop(columns=["event_time"])
    else:
        features_df["open_interest"] = pd.NA
        features_df["open_interest_value"] = pd.NA
        features_df["circulating_supply"] = pd.NA
        features_df["open_interest_change_pct"] = pd.NA
        features_df["open_interest_value_zscore_21"] = pd.NA
        features_df["open_interest_change_pct_zscore_21"] = pd.NA
    features_df["open_interest_feature_available"] = features_df["open_interest"].notna()

    long_short_path = get_raw_futures_long_short_ratio_path(symbol, timeframe, start_date, end_date)
    long_short_df = read_optional_csv(long_short_path, parse_dates=["event_time"])
    if not long_short_df.empty:
        long_short_df = long_short_df.sort_values("event_time").reset_index(drop=True)
        long_short_df = long_short_df.dropna(subset=["event_time"]).reset_index(drop=True)
        for column in ["long_short_ratio", "long_account_share", "short_account_share"]:
            long_short_df[column] = pd.to_numeric(long_short_df[column], errors="coerce")
        long_short_df = add_rolling_zscore(
            long_short_df,
            "long_short_ratio",
            "long_short_ratio_zscore_21",
        )
        features_df = pd.merge_asof(
            features_df.sort_values("close_time"),
            long_short_df[
                [
                    "event_time",
                    "long_short_ratio",
                    "long_account_share",
                    "short_account_share",
                    "long_short_ratio_zscore_21",
                ]
            ].sort_values("event_time"),
            left_on="close_time",
            right_on="event_time",
            direction="backward",
        )
        features_df = features_df.drop(columns=["event_time"])
    else:
        features_df["long_short_ratio"] = pd.NA
        features_df["long_account_share"] = pd.NA
        features_df["short_account_share"] = pd.NA
        features_df["long_short_ratio_zscore_21"] = pd.NA
    features_df["positioning_feature_available"] = features_df["long_short_ratio"].notna()

    taker_path = get_raw_futures_taker_volume_path(symbol, timeframe, start_date, end_date)
    taker_df = read_optional_csv(taker_path, parse_dates=["event_time"])
    if not taker_df.empty:
        taker_df = taker_df.sort_values("event_time").reset_index(drop=True)
        taker_df = taker_df.dropna(subset=["event_time"]).reset_index(drop=True)
        for column in ["taker_buy_sell_ratio", "taker_buy_volume", "taker_sell_volume"]:
            taker_df[column] = pd.to_numeric(taker_df[column], errors="coerce")
        taker_df = add_rolling_zscore(
            taker_df,
            "taker_buy_sell_ratio",
            "taker_buy_sell_ratio_zscore_21",
        )
        features_df = pd.merge_asof(
            features_df.sort_values("close_time"),
            taker_df[
                [
                    "event_time",
                    "taker_buy_sell_ratio",
                    "taker_buy_volume",
                    "taker_sell_volume",
                    "taker_buy_sell_ratio_zscore_21",
                ]
            ].sort_values("event_time"),
            left_on="close_time",
            right_on="event_time",
            direction="backward",
        )
        features_df = features_df.drop(columns=["event_time"])
    else:
        features_df["taker_buy_sell_ratio"] = pd.NA
        features_df["taker_buy_volume"] = pd.NA
        features_df["taker_sell_volume"] = pd.NA
        features_df["taker_buy_sell_ratio_zscore_21"] = pd.NA
    features_df["taker_flow_feature_available"] = features_df["taker_buy_sell_ratio"].notna()

    basis_path = get_raw_futures_basis_path(symbol, timeframe, start_date, end_date, contract_type=contract_type)
    basis_df = read_optional_csv(basis_path, parse_dates=["event_time"])
    if not basis_df.empty:
        basis_df = basis_df.sort_values("event_time").reset_index(drop=True)
        basis_df = basis_df.dropna(subset=["event_time"]).reset_index(drop=True)
        for column in ["basis_rate", "basis_value", "annualized_basis_rate", "futures_price", "index_price"]:
            if column in basis_df.columns:
                basis_df[column] = pd.to_numeric(basis_df[column], errors="coerce")
        basis_df = add_rolling_zscore(
            basis_df,
            "basis_rate",
            "basis_rate_zscore_21",
        )
        features_df = pd.merge_asof(
            features_df.sort_values("close_time"),
            basis_df[
                [
                    "event_time",
                    "basis_rate",
                    "basis_value",
                    "annualized_basis_rate",
                    "basis_rate_zscore_21",
                    "futures_price",
                    "index_price",
                ]
            ].sort_values("event_time"),
            left_on="close_time",
            right_on="event_time",
            direction="backward",
        )
        features_df = features_df.drop(columns=["event_time"])
    else:
        features_df["basis_rate"] = pd.NA
        features_df["basis_value"] = pd.NA
        features_df["annualized_basis_rate"] = pd.NA
        features_df["basis_rate_zscore_21"] = pd.NA
        features_df["futures_price"] = pd.NA
        features_df["index_price"] = pd.NA
    features_df["basis_feature_available"] = features_df["basis_rate"].notna()

    crowding_components = [
        "funding_rate_zscore_21",
        "long_short_ratio_zscore_21",
        "taker_buy_sell_ratio_zscore_21",
        "basis_rate_zscore_21",
    ]
    activity_components = [
        "open_interest_value_zscore_21",
        "open_interest_change_pct_zscore_21",
    ]
    features_df["futures_crowding_score"] = features_df[crowding_components].mean(axis=1, skipna=True)
    features_df["futures_activity_score"] = features_df[activity_components].mean(axis=1, skipna=True)
    sublayer_columns = [
        "funding_feature_available",
        "open_interest_feature_available",
        "positioning_feature_available",
        "taker_flow_feature_available",
        "basis_feature_available",
    ]
    features_df["futures_feature_completeness_score"] = (
        features_df[sublayer_columns].astype(float).mean(axis=1)
    )
    features_df["futures_completeness_label"] = "partial"
    features_df.loc[features_df["futures_feature_completeness_score"] >= 0.99, "futures_completeness_label"] = "full"
    features_df.loc[features_df["futures_feature_completeness_score"] < 0.40, "futures_completeness_label"] = "thin"

    recent_structure_columns = [
        "open_interest",
        "open_interest_value",
        "long_short_ratio",
        "taker_buy_sell_ratio",
        "basis_rate",
    ]
    features_df["futures_structure_data_available"] = features_df[recent_structure_columns].notna().any(axis=1)
    features_df["funding_data_available"] = features_df["funding_rate"].notna()
    features_df["futures_data_available"] = (
        features_df["futures_structure_data_available"] | features_df["funding_data_available"]
    )

    output_path = get_futures_features_path(symbol, timeframe, start_date, end_date)
    features_df.to_csv(output_path, index=False)

    print(f"futures feature construction completed for {symbol}")
    print(f"timeframe: {timeframe}")
    print(f"features saved to: {output_path}")
    print(f"total rows: {len(features_df)}")

    return features_df


def build_futures_features():
    return build_futures_features_for_symbol(SYMBOL, TIMEFRAME, START_DATE, END_DATE)


if __name__ == "__main__":
    build_futures_features()
