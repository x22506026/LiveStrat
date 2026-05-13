import pandas as pd

from src.io_paths import ensure_dirs
from src.config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    get_all_symbols,
    get_market_features_path,
    get_raw_binance_path,
)


# this file builds market behaviour features from raw binance ohlcv data
# features include momentum, trend, volatility, and participation measures

SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
RUN_ALL_SYMBOLS = False


def get_timeframe_return_periods(timeframe):
    """Map a timeframe to the number of steps needed for 24h and 3d returns."""
    periods_per_24h = {
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }.get(timeframe)
    if periods_per_24h is None:
        raise ValueError(f"unsupported timeframe for market features: {timeframe}")
    return periods_per_24h, periods_per_24h * 3


def build_market_features_for_symbol(symbol, timeframe, start_date, end_date):
    # ensure output directories exist
    ensure_dirs()

    # load the exact raw dataset produced by the binance audit step
    raw_file = get_raw_binance_path(symbol, timeframe, start_date, end_date)
    if not raw_file.exists():
        raise FileNotFoundError(
            f"raw market file not found: {raw_file}. run the binance audit step first."
        )

    print("loading raw market data...")
    df = pd.read_csv(raw_file, parse_dates=["open_time", "close_time"])

    # sort by time to avoid leakage
    df = df.sort_values("open_time").reset_index(drop=True)

    # Momentum features describe how the price has been moving recently.

    # simple return over one candle
    df["return_4h"] = df["close"].pct_change()

    # return over longer windows, adjusted to the selected candle interval
    periods_24h, periods_3d = get_timeframe_return_periods(timeframe)
    df["return_24h"] = df["close"].pct_change(periods=periods_24h)
    df["return_3d"] = df["close"].pct_change(periods=periods_3d)

    # Trend features describe the medium-term direction of price.

    # moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # price relative to moving averages
    df["price_sma_20_diff"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["price_sma_50_diff"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # Risk and volatility features describe how noisy and how active the market is.

    # rolling volatility of returns
    df["volatility_20"] = df["return_4h"].rolling(window=20).std()
    df["volatility_50"] = df["return_4h"].rolling(window=50).std()

    # volume anomaly (z-score)
    volume_mean = df["volume"].rolling(window=20).mean()
    volume_std = df["volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["volume"] - volume_mean) / volume_std

    # candle range and body help describe intraperiod market behavior
    df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]

    # taker buy volume ratio acts as a simple directional participation signal
    df["taker_buy_volume_ratio"] = df["taker_buy_base_volume"] / df["volume"]

    # trade count anomaly shows unusual market activity versus recent history
    trades_mean = df["number_of_trades"].rolling(window=20).mean()
    trades_std = df["number_of_trades"].rolling(window=20).std()
    df["trade_count_zscore"] = (df["number_of_trades"] - trades_mean) / trades_std

    # Final cleanup before saving the feature dataset.

    # drop early rows with incomplete rolling windows
    df = df.dropna().reset_index(drop=True)

    # save processed feature dataset
    output_file = get_market_features_path(symbol, timeframe, start_date, end_date)
    df.to_csv(output_file, index=False)

    print("market feature construction completed")
    print(f"symbol: {symbol}")
    print(f"features saved to: {output_file}")
    print(f"total rows: {len(df)}")


def build_market_features():
    symbols = get_all_symbols() if RUN_ALL_SYMBOLS else [SYMBOL]
    for symbol in symbols:
        build_market_features_for_symbol(symbol, TIMEFRAME, START_DATE, END_DATE)


if __name__ == "__main__":
    build_market_features()
