# this file evaluates a simple rule based trading strategy
# the strategy is compared against the true future based labels
# this provides a transparent baseline to compare with ml models

import pandas as pd

from src.config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_labeled_market_path,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)


SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
MODEL_NAME = "rule_based"


def apply_rule_based_strategy(df):
    # default action is hold
    df["rule_signal"] = "hold"

    # buy condition: trend up and positive momentum
    buy_condition = (
        (df["close"] > df["sma_50"]) &
        (df["return_24h"] > 0)
    )

    # dont buy condition: trend down and negative momentum
    dont_buy_condition = (
        (df["close"] < df["sma_50"]) &
        (df["return_24h"] < 0)
    )

    df.loc[buy_condition, "rule_signal"] = "buy"
    df.loc[dont_buy_condition, "rule_signal"] = "dont_buy"

    return df


def evaluate_rule_strategy():
    # load labeled dataset
    input_file = get_labeled_market_path(SYMBOL, TIMEFRAME)
    df = pd.read_csv(input_file, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # apply rule based logic
    df = apply_rule_based_strategy(df)

    # use the same held-out test split as the model baselines
    _, _, _, y_true = make_time_based_split(df[["close"]], df["label"], TRAIN_RATIO)
    split_idx = int(len(df) * TRAIN_RATIO)
    y_pred = df["rule_signal"].iloc[split_idx:]

    metrics_df = build_metrics_dataframe(y_true, y_pred, MODEL_NAME, SYMBOL, TIMEFRAME)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)

    metrics_df.to_csv(get_evaluation_metrics_path(SYMBOL, TIMEFRAME, MODEL_NAME), index=False)
    confusion_df.to_csv(get_confusion_matrix_path(SYMBOL, TIMEFRAME, MODEL_NAME))

    print_evaluation_summary("rule based strategy evaluation", y_true, y_pred)


if __name__ == "__main__":
    evaluate_rule_strategy()
