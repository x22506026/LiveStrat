"""Evaluate binary directional market + futures models for collapse-prone cases."""

import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_all_symbols,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_futures_binary_summary_path,
    get_market_futures_binary_walkforward_detail_path,
    get_market_futures_binary_walkforward_summary_path,
    get_market_futures_dataset_path,
)
from src.models.evaluate import build_confusion_matrix_dataframe, build_metrics_dataframe, make_time_based_split
from src.models.evaluate_market_futures_backtests import build_lagged_feature_frames, train_lagged_random_forest_baseline
from src.models.evaluate_market_futures_strategies import build_combined_feature_matrix, train_scaled_logistic_baseline
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_SYMBOLS = get_all_symbols()
LAG_WINDOW = 12


def _build_binary_labels(df, target_config, timeframe):
    """Convert the preferred 3-class target into a binary long-vs-flat label."""
    labeled_df = build_target_labels(df, target_config, timeframe=timeframe)
    labeled_df = labeled_df.copy()
    labeled_df["binary_label"] = "flat"
    labeled_df.loc[labeled_df["future_return"] >= labeled_df["buy_threshold"], "binary_label"] = "long"
    return labeled_df


def _safe_balanced_accuracy(y_true, y_pred):
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)).tolist())
    recalls = []
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    for label in labels:
        true_mask = y_true_series == label
        support = int(true_mask.sum())
        if support == 0:
            continue
        recalls.append(float((y_pred_series[true_mask] == label).mean()))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(_safe_balanced_accuracy(y_true, y_pred)),
        "long_precision": float(precision_score(y_true, y_pred, pos_label="long", zero_division=0)),
        "long_recall": float(recall_score(y_true, y_pred, pos_label="long", zero_division=0)),
        "long_f1": float(f1_score(y_true, y_pred, pos_label="long", zero_division=0)),
    }


def _persist_outputs(symbol, timeframe, model_name, start_date, end_date, y_true, y_pred):
    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, symbol, timeframe)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)
    metrics_df.to_csv(get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date), index=False)
    confusion_df.to_csv(get_confusion_matrix_path(symbol, timeframe, model_name, start_date, end_date))


def _evaluate_static_logistic(df, symbol, timeframe, start_date, end_date):
    X, _ = build_combined_feature_matrix(df)
    y = df["binary_label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    model = train_scaled_logistic_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name = "market_futures_binary_logistic"
    _persist_outputs(symbol, timeframe, model_name, start_date, end_date, y_test, y_pred)
    metrics = _binary_metrics(y_test, y_pred)
    latest_signal = model.predict(X.iloc[[-1]])[0]
    latest_confidence = float(model.predict_proba(X.iloc[[-1]]).max())
    return {
        "model_name": model_name,
        "latest_signal": latest_signal,
        "latest_signal_confidence": latest_confidence,
        **metrics,
    }


def _evaluate_lagged_models(df, symbol, timeframe, start_date, end_date):
    X, _ = build_combined_feature_matrix(df)
    y = df["binary_label"].reset_index(drop=True)
    split_idx = int(len(X) * TRAIN_RATIO)
    lagged_X, lagged_y, lagged_indices = build_lagged_feature_frames(X, y, window_size=LAG_WINDOW)
    train_mask = lagged_indices < split_idx
    test_mask = lagged_indices >= split_idx

    rows = []
    specs = [
        ("market_futures_binary_lagged_logistic", train_scaled_logistic_baseline),
        ("market_futures_binary_lagged_forest", train_lagged_random_forest_baseline),
    ]
    for model_name, trainer in specs:
        X_train = lagged_X.loc[train_mask].reset_index(drop=True)
        y_train = lagged_y.loc[train_mask].reset_index(drop=True)
        X_test = lagged_X.loc[test_mask].reset_index(drop=True)
        y_test = lagged_y.loc[test_mask].reset_index(drop=True)
        if min(len(X_train), len(X_test)) == 0:
            continue

        model = trainer(X_train, y_train)
        y_pred = model.predict(X_test)
        _persist_outputs(symbol, timeframe, model_name, start_date, end_date, y_test, y_pred)
        metrics = _binary_metrics(y_test, y_pred)
        latest_signal = model.predict(lagged_X.iloc[[-1]])[0]
        latest_confidence = float(model.predict_proba(lagged_X.iloc[[-1]]).max())
        rows.append(
            {
                "model_name": model_name,
                "latest_signal": latest_signal,
                "latest_signal_confidence": latest_confidence,
                **metrics,
            }
        )

    return rows


def _walkforward_folds(total_rows, timeframe):
    min_train_rows = 72 if timeframe == "4h" else 120
    test_rows = 12 if timeframe == "4h" else 24
    step_rows = test_rows
    folds = []
    fold_number = 1
    train_end = min_train_rows

    while train_end + test_rows <= total_rows:
        folds.append({"fold_number": fold_number, "train_end": train_end, "test_end": train_end + test_rows})
        train_end += step_rows
        fold_number += 1

    return folds


def _evaluate_walkforward(df, symbol, timeframe, start_date, end_date, selected_model_name):
    X, _ = build_combined_feature_matrix(df)
    y = df["binary_label"].reset_index(drop=True)
    detail_rows = []

    for fold in _walkforward_folds(len(df), timeframe):
        X_train = X.iloc[:fold["train_end"]]
        X_test = X.iloc[fold["train_end"]:fold["test_end"]]
        y_train = y.iloc[:fold["train_end"]]
        y_test = y.iloc[fold["train_end"]:fold["test_end"]]

        if selected_model_name == "market_futures_binary_logistic":
            model = train_scaled_logistic_baseline(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            lagged_X, lagged_y, lagged_indices = build_lagged_feature_frames(X, y, window_size=LAG_WINDOW)
            train_mask = lagged_indices < fold["train_end"]
            test_mask = (lagged_indices >= fold["train_end"]) & (lagged_indices < fold["test_end"])
            if not train_mask.any() or not test_mask.any():
                continue
            trainer = (
                train_scaled_logistic_baseline
                if selected_model_name == "market_futures_binary_lagged_logistic"
                else train_lagged_random_forest_baseline
            )
            model = trainer(
                lagged_X.loc[train_mask].reset_index(drop=True),
                lagged_y.loc[train_mask].reset_index(drop=True),
            )
            y_test = lagged_y.loc[test_mask].reset_index(drop=True)
            y_pred = model.predict(lagged_X.loc[test_mask].reset_index(drop=True))

        metrics = _binary_metrics(y_test, y_pred)
        detail_rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": selected_model_name,
                "window_start": start_date,
                "window_end": end_date,
                "fold_number": fold["fold_number"],
                "fold_accuracy": metrics["accuracy"],
                "fold_macro_f1": metrics["macro_f1"],
                "fold_balanced_accuracy": metrics["balanced_accuracy"],
                "fold_long_precision": metrics["long_precision"],
                "fold_long_recall": metrics["long_recall"],
                "fold_long_f1": metrics["long_f1"],
            }
        )

    return pd.DataFrame(detail_rows)


def evaluate_market_futures_binary_directional(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Run a binary directional market + futures family for long-vs-flat research."""
    symbols = tuple(symbols or SUPPORTED_SYMBOLS)
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    summary_rows = []
    walkforward_frames = []

    for symbol in symbols:
        dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
        df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
        df = df[df["futures_data_available"] == True].copy()
        df = df.sort_values("open_time").reset_index(drop=True)
        df = _build_binary_labels(df, preferred_targets[symbol], timeframe)

        rows = [_evaluate_static_logistic(df, symbol, timeframe, start_date, end_date)]
        rows.extend(_evaluate_lagged_models(df, symbol, timeframe, start_date, end_date))

        summary_df = pd.DataFrame(rows)
        summary_df.insert(0, "symbol", symbol)
        summary_df.insert(1, "timeframe", timeframe)
        summary_df.insert(2, "window_start", start_date)
        summary_df.insert(3, "window_end", end_date)
        summary_df["target_name"] = preferred_targets[symbol]["target_name"]
        summary_df["target_horizon_hours"] = preferred_targets[symbol].get("effective_horizon_hours")
        summary_df["long_share"] = float((df["binary_label"] == "long").mean())
        summary_df["flat_share"] = float((df["binary_label"] == "flat").mean())
        summary_rows.append(summary_df)

        best_row = summary_df.sort_values(["macro_f1", "accuracy", "balanced_accuracy"], ascending=False).iloc[0]
        walkforward_frames.append(
            _evaluate_walkforward(df, symbol, timeframe, start_date, end_date, best_row["model_name"])
        )

    combined_summary = pd.concat(summary_rows, ignore_index=True)
    walkforward_detail = pd.concat(walkforward_frames, ignore_index=True)

    if walkforward_detail.empty:
        selected_rows = (
            combined_summary.sort_values(["symbol", "macro_f1", "accuracy", "balanced_accuracy"], ascending=[True, False, False, False])
            .drop_duplicates("symbol")
        )
        walkforward_summary = pd.DataFrame(
            [
                {
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "model_name": row["model_name"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "walkforward_fold_count": 0,
                    "walkforward_avg_accuracy": 0.0,
                    "walkforward_avg_macro_f1": 0.0,
                    "walkforward_avg_balanced_accuracy": 0.0,
                    "walkforward_avg_long_precision": 0.0,
                    "walkforward_avg_long_recall": 0.0,
                    "walkforward_avg_long_f1": 0.0,
                    "walkforward_summary": (
                        f"{row['symbol']} binary market+futures walk-forward was skipped on {row['timeframe']} "
                        "because the available futures-aligned window is too short for reliable rolling folds."
                    ),
                }
                for _, row in selected_rows.iterrows()
            ]
        )
    else:
        walkforward_summary = (
            walkforward_detail.groupby(["symbol", "timeframe", "model_name", "window_start", "window_end"], as_index=False)
            .agg(
                walkforward_fold_count=("fold_number", "count"),
                walkforward_avg_accuracy=("fold_accuracy", "mean"),
                walkforward_avg_macro_f1=("fold_macro_f1", "mean"),
                walkforward_avg_balanced_accuracy=("fold_balanced_accuracy", "mean"),
                walkforward_avg_long_precision=("fold_long_precision", "mean"),
                walkforward_avg_long_recall=("fold_long_recall", "mean"),
                walkforward_avg_long_f1=("fold_long_f1", "mean"),
            )
        )
        walkforward_summary["walkforward_summary"] = walkforward_summary.apply(
            lambda row: (
                f"{row['symbol']} binary market+futures walk-forward used {int(row['walkforward_fold_count'])} folds with "
                f"accuracy {row['walkforward_avg_accuracy'] * 100:.1f}%, macro-F1 {row['walkforward_avg_macro_f1'] * 100:.1f}%, "
                f"and long-class F1 {row['walkforward_avg_long_f1'] * 100:.1f}%."
            ),
            axis=1,
        )

    combined_summary.to_csv(get_market_futures_binary_summary_path(timeframe, start_date, end_date), index=False)
    walkforward_detail.to_csv(get_market_futures_binary_walkforward_detail_path(timeframe, start_date, end_date), index=False)
    walkforward_summary.to_csv(get_market_futures_binary_walkforward_summary_path(timeframe, start_date, end_date), index=False)

    print("market + futures binary summary generated")
    print(f"rows saved: {len(combined_summary)}")
    print(f"summary saved to: {get_market_futures_binary_summary_path(timeframe, start_date, end_date)}")
    print("market + futures binary walk-forward summary generated")
    print(f"rows saved: {len(walkforward_summary)}")
    print(f"summary saved to: {get_market_futures_binary_walkforward_summary_path(timeframe, start_date, end_date)}")
    return {
        "summary": combined_summary,
        "walkforward_summary": walkforward_summary,
    }


if __name__ == "__main__":
    evaluate_market_futures_binary_directional()
