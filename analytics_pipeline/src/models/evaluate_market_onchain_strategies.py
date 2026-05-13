"""Evaluate first-pass market + on-chain strategies on the aligned daily datasets."""

import pandas as pd

from src.config import (
    BUY_THRESHOLD,
    DONT_BUY_THRESHOLD,
    ONCHAIN_FREQUENCY,
    TRAIN_RATIO,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_onchain_dataset_path,
    get_strategy_summary_path,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)


SUPPORTED_STRATEGY_ASSETS = ["BTC", "ETH"]
TIMEFRAME = ONCHAIN_FREQUENCY


def build_daily_labels(df):
    """Create future daily labels from the combined market + on-chain dataset."""
    df = df.copy()
    df["future_market_close"] = df["market_close"].shift(-1)
    df["future_return_1d"] = (df["future_market_close"] / df["market_close"]) - 1.0
    df["label"] = "hold"
    df.loc[df["future_return_1d"] >= BUY_THRESHOLD, "label"] = "buy"
    df.loc[df["future_return_1d"] <= DONT_BUY_THRESHOLD, "label"] = "dont_buy"
    return df.dropna(subset=["future_market_close", "future_return_1d"]).reset_index(drop=True)


def apply_market_onchain_regime_filter(df):
    """Use on-chain regime as a gate on top of a market trend rule."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_close"] > df["market_sma_50"]) &
        (df["market_return_24h"] > 0) &
        (df["onchain_regime_score"] > 0)
    )
    dont_buy_condition = (
        (df["market_close"] < df["market_sma_50"]) &
        (df["market_return_24h"] < 0) &
        (df["onchain_regime_score"] < 0)
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_market_onchain_confirmation(df):
    """Require stronger agreement between market structure and on-chain state."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_close"] > df["market_sma_20"]) &
        (df["market_sma_20"] > df["market_sma_50"]) &
        (df["market_return_24h"] > 0) &
        (df["market_price_sma_20_diff"] > 0) &
        (df["onchain_regime_label"] == "supportive")
    )
    dont_buy_condition = (
        (df["market_close"] < df["market_sma_20"]) &
        (df["market_sma_20"] < df["market_sma_50"]) &
        (df["market_return_24h"] < 0) &
        (df["market_price_sma_20_diff"] < 0) &
        (df["onchain_regime_label"] == "weakening")
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_market_onchain_divergence_guard(df):
    """Use structural divergence and distribution pressure as a caution layer."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_trend_label"] == "bullish") &
        (df["market_momentum_label"] == "positive") &
        (df["market_onchain_alignment_score"] >= 0.20) &
        (df["market_onchain_divergence_score"] < 0.25) &
        (df["onchain_distribution_risk_score"] < 0.35)
    )
    dont_buy_condition = (
        (df["market_onchain_structural_label"].isin(["distribution_risk", "divergence_watch"])) |
        (
            (df["market_trend_label"] == "bearish") &
            (df["onchain_risk_bias"] > 0.20)
        )
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def evaluate_strategy(df, asset_symbol, model_name):
    """Evaluate one strategy using the shared time-based evaluation helpers."""
    _, _, _, y_true = make_time_based_split(df[["market_close"]], df["label"], TRAIN_RATIO)
    split_idx = int(len(df) * TRAIN_RATIO)
    y_pred = df["strategy_signal"].iloc[split_idx:]

    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, asset_symbol, TIMEFRAME)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)

    metrics_df.to_csv(get_evaluation_metrics_path(asset_symbol, TIMEFRAME, model_name), index=False)
    confusion_df.to_csv(get_confusion_matrix_path(asset_symbol, TIMEFRAME, model_name))
    print_evaluation_summary(f"{asset_symbol} {model_name} evaluation", y_true, y_pred)

    accuracy = float(metrics_df.loc[metrics_df["metric"] == "accuracy", "value"].iloc[0])
    macro_f1 = float(metrics_df.loc[metrics_df["metric"] == "macro_f1", "value"].iloc[0])
    return accuracy, macro_f1


def build_strategy_summary_row(asset_symbol, strategy_name, df, accuracy, macro_f1):
    """Create a compact latest summary row for app/research use."""
    latest = df.iloc[-1]
    return {
        "asset_symbol": asset_symbol,
        "strategy_name": strategy_name,
        "latest_window_end": latest["window_end_utc"],
        "latest_signal": latest["strategy_signal"],
        "latest_market_return_24h_pct": float(latest["market_return_24h"]) * 100,
        "latest_onchain_regime_label": latest["onchain_regime_label"],
        "latest_onchain_regime_score": latest["onchain_regime_score"],
        "latest_onchain_confidence_score": latest.get("onchain_confidence_score", 0.0),
        "latest_market_onchain_alignment_score": latest.get("market_onchain_alignment_score", 0.0),
        "latest_market_onchain_divergence_score": latest.get("market_onchain_divergence_score", 0.0),
        "latest_market_onchain_structural_label": latest.get("market_onchain_structural_label", "mixed"),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "strategy_summary": (
            f"{strategy_name} on {asset_symbol} currently suggests {latest['strategy_signal']}. "
            f"Market 24h return is {float(latest['market_return_24h']) * 100:.2f}% "
            f"and the on-chain regime is {latest['onchain_regime_label']}. "
            f"Structural label is {str(latest.get('market_onchain_structural_label', 'mixed')).replace('_', ' ')}."
        ),
    }


def evaluate_market_onchain_strategies():
    """Run the first two market + on-chain strategies on supported assets."""
    summary_rows = []

    for asset_symbol in SUPPORTED_STRATEGY_ASSETS:
        dataset_path = get_market_onchain_dataset_path(asset_symbol, ONCHAIN_FREQUENCY)
        df = pd.read_csv(dataset_path)
        df = df[df["onchain_data_available"] == True].copy()
        df = build_daily_labels(df)

        regime_filter_df = apply_market_onchain_regime_filter(df)
        regime_accuracy, regime_macro_f1 = evaluate_strategy(
            regime_filter_df,
            asset_symbol,
            "market_onchain_regime_filter",
        )
        summary_rows.append(
            build_strategy_summary_row(
                asset_symbol,
                "market_onchain_regime_filter",
                regime_filter_df,
                regime_accuracy,
                regime_macro_f1,
            )
        )

        confirmation_df = apply_market_onchain_confirmation(df)
        confirmation_accuracy, confirmation_macro_f1 = evaluate_strategy(
            confirmation_df,
            asset_symbol,
            "market_onchain_confirmation",
        )
        summary_rows.append(
            build_strategy_summary_row(
                asset_symbol,
                "market_onchain_confirmation",
                confirmation_df,
                confirmation_accuracy,
                confirmation_macro_f1,
            )
        )

        divergence_guard_df = apply_market_onchain_divergence_guard(df)
        divergence_accuracy, divergence_macro_f1 = evaluate_strategy(
            divergence_guard_df,
            asset_symbol,
            "market_onchain_divergence_guard",
        )
        summary_rows.append(
            build_strategy_summary_row(
                asset_symbol,
                "market_onchain_divergence_guard",
                divergence_guard_df,
                divergence_accuracy,
                divergence_macro_f1,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = get_strategy_summary_path("market_onchain", ONCHAIN_FREQUENCY)
    summary_df.to_csv(summary_path, index=False)

    print("market + on-chain strategy summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")

    return summary_df


if __name__ == "__main__":
    evaluate_market_onchain_strategies()
