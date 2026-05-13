"""Evaluate multiple market + futures target definitions using the logistic benchmark."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_all_symbols,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_futures_dataset_path,
    get_market_futures_target_variant_summary_path,
)
from src.models.market_futures_targets import (
    build_target_configs,
    build_target_labels,
    resolve_target_config_for_timeframe,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)
from src.models.evaluate_market_futures_strategies import (
    build_combined_feature_matrix,
    train_scaled_logistic_baseline,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_SYMBOLS = get_all_symbols()


def evaluate_target_variant(df, symbol, target_config, timeframe, start_date, end_date):
    """Train and evaluate the logistic benchmark on one target definition."""
    resolved_target = resolve_target_config_for_timeframe(target_config, timeframe)
    labeled_df = build_target_labels(df, resolved_target)
    X, feature_names = build_combined_feature_matrix(labeled_df)
    y = labeled_df["label"]

    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {
            "symbol": symbol,
            "target_name": target_config["target_name"],
            "label_mode": resolved_target["label_mode"],
            "horizon_steps": resolved_target["horizon_steps"],
            "target_horizon_hours": resolved_target.get("effective_horizon_hours"),
            "target_exact_horizon_match": resolved_target.get("exact_horizon_match"),
            "target_resolution_note": resolved_target.get("horizon_resolution_note"),
            "status": "skipped_low_class_diversity",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_class_count": int(y_train.nunique()),
            "test_class_count": int(y_test.nunique()),
        }

    model = train_scaled_logistic_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name = f"market_futures_logistic_{target_config['target_name']}"

    metrics_df = build_metrics_dataframe(y_test, y_pred, model_name, symbol, timeframe)
    confusion_df = build_confusion_matrix_dataframe(y_test, y_pred)

    metrics_path = get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date)
    confusion_path = get_confusion_matrix_path(symbol, timeframe, model_name, start_date, end_date)
    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path)
    print_evaluation_summary(f"{symbol} {model_name} evaluation", y_test, y_pred)

    latest_prediction = model.predict(X.iloc[[-1]])[0]
    latest_probability = float(model.predict_proba(X.iloc[[-1]]).max())
    accuracy = float(metrics_df.loc[metrics_df["metric"] == "accuracy", "value"].iloc[0])
    macro_f1 = float(metrics_df.loc[metrics_df["metric"] == "macro_f1", "value"].iloc[0])
    balanced_accuracy = float(metrics_df.loc[metrics_df["metric"] == "balanced_accuracy", "value"].iloc[0])

    threshold_description = (
        f"volatility-adjusted x{resolved_target['vol_multiplier']}"
        if resolved_target["label_mode"] == "vol_adjusted"
        else f"fixed +/-{resolved_target['buy_threshold']:.4f}"
    )

    feature_names = list(feature_names)
    return {
        "symbol": symbol,
        "target_name": resolved_target["target_name"],
        "label_mode": resolved_target["label_mode"],
        "horizon_steps": resolved_target["horizon_steps"],
        "target_horizon_hours": resolved_target.get("effective_horizon_hours"),
        "target_exact_horizon_match": resolved_target.get("exact_horizon_match"),
        "target_resolution_note": resolved_target.get("horizon_resolution_note"),
        "threshold_rule": threshold_description,
        "status": "evaluated",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "latest_signal": latest_prediction,
        "latest_signal_confidence": latest_probability,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": balanced_accuracy,
        "buy_count": int((y == "buy").sum()),
        "dont_buy_count": int((y == "dont_buy").sum()),
        "hold_count": int((y == "hold").sum()),
        "feature_count": len(feature_names),
        "top_feature_hint": feature_names[0] if feature_names else "n/a",
    }


def evaluate_market_futures_target_variants(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Run the balanced logistic benchmark across multiple target definitions."""
    summary_rows = []
    target_configs = build_target_configs()

    for symbol in SUPPORTED_SYMBOLS:
        dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
        df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
        df = df[df["futures_data_available"] == True].copy()
        df = df.sort_values("open_time").reset_index(drop=True)

        for target_config in target_configs:
            summary_rows.append(
                evaluate_target_variant(
                    df,
                    symbol,
                    target_config,
                    timeframe,
                    start_date,
                    end_date,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = get_market_futures_target_variant_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(summary_path, index=False)

    print("market + futures target variant summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_target_variants()
