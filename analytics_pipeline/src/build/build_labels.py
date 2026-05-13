import pandas as pd

from src.io_paths import PROCESSED_DIR, ensure_dirs
from src.config import (
    DEFAULT_TIMEFRAME,
    DEFAULT_SYMBOL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    build_labeled_market_filename,
    get_all_symbols,
    get_market_features_path,
)
from src.models.market_futures_targets import describe_target_for_timeframe


# this file creates future-based labels from the market feature dataset
# labels are derived from future price movement while features stay historical

SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE

DEFAULT_TARGET_NAME = "fixed_h24"
RUN_ALL_SYMBOLS = False


def build_labels_for_symbol(symbol, timeframe, start_date=None, end_date=None):
    # ensure directories exist
    ensure_dirs()

    input_file = get_market_features_path(symbol, timeframe, start_date, end_date)
    df = pd.read_csv(input_file, parse_dates=["open_time"])
    target_definition = describe_target_for_timeframe(DEFAULT_TARGET_NAME, timeframe)
    horizon_steps = target_definition["horizon_steps"]
    buy_threshold = target_definition.get("buy_threshold", 0.0)
    dont_buy_threshold = target_definition.get("dont_buy_threshold", -buy_threshold)

    # compute future return over the horizon
    # future_close is shifted backwards so it represents the close at t+horizon
    df["future_close"] = df["close"].shift(-horizon_steps)
    df["future_return"] = (df["future_close"] / df["close"]) - 1.0

    # build label based on future_return thresholds
    df["label"] = "hold"
    df.loc[df["future_return"] >= buy_threshold, "label"] = "buy"
    df.loc[df["future_return"] <= dont_buy_threshold, "label"] = "dont_buy"
    df["target_name"] = DEFAULT_TARGET_NAME
    df["target_horizon_hours"] = target_definition["effective_horizon_hours"]
    df["target_exact_horizon_match"] = target_definition["exact_horizon_match"]
    df["target_resolution_note"] = target_definition["horizon_resolution_note"]

    # drop rows where future is not known (last horizon rows)
    df = df.dropna(subset=["future_close", "future_return"]).reset_index(drop=True)

    # save output
    output_file = PROCESSED_DIR / build_labeled_market_filename(symbol, timeframe, start_date, end_date)
    df.to_csv(output_file, index=False)

    # simple label distribution output for sanity check
    print("label building completed")
    print(f"symbol: {symbol}")
    print(f"saved to: {output_file}")
    print("label counts:")
    print(df["label"].value_counts())


def build_labels():
    symbols = get_all_symbols() if RUN_ALL_SYMBOLS else [SYMBOL]
    for symbol in symbols:
        build_labels_for_symbol(symbol, TIMEFRAME, START_DATE, END_DATE)


if __name__ == "__main__":
    build_labels()
