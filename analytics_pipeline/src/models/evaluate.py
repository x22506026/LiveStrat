"""Shared evaluation helpers for baseline model reporting."""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def make_time_based_split(X, y, train_ratio):
    """Split features and labels chronologically to avoid lookahead leakage."""
    split_idx = int(len(X) * train_ratio)
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def build_metrics_dataframe(y_true, y_pred, model_name, symbol, timeframe):
    """Create a compact evaluation table that is easy to save and reuse later."""
    metrics_rows = [
        {
            "model_name": model_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "metric": "accuracy",
            "value": accuracy_score(y_true, y_pred),
        },
        {
            "model_name": model_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "metric": "macro_f1",
            "value": f1_score(y_true, y_pred, average="macro", zero_division=0),
        },
        {
            "model_name": model_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "metric": "balanced_accuracy",
            "value": balanced_accuracy_score(y_true, y_pred),
        },
        {
            "model_name": model_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "metric": "weighted_f1",
            "value": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        },
    ]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for label, label_metrics in report.items():
        if not isinstance(label_metrics, dict):
            continue

        for metric_name in ["precision", "recall", "f1-score", "support"]:
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "metric": f"{label}_{metric_name}",
                    "value": label_metrics[metric_name],
                }
            )

    return pd.DataFrame(metrics_rows)


def build_confusion_matrix_dataframe(y_true, y_pred):
    """Return a labeled confusion matrix dataframe for downstream reporting."""
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)).tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def print_evaluation_summary(title, y_true, y_pred):
    """Print a readable summary while keeping the saved outputs machine-friendly."""
    print(title)
    print("-" * len(title))
    print(classification_report(y_true, y_pred, zero_division=0))
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
