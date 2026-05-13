"""Evaluate the market-only benchmark family for LiveStrat."""

import math

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BUY_THRESHOLD,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    DONT_BUY_THRESHOLD,
    MARKET_FEATURE_COLUMNS,
    TRAIN_RATIO,
    get_all_symbols,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_labeled_market_path,
    get_market_trend_feature_review_path,
    get_market_trend_forecast_summary_path,
    get_market_trend_regression_summary_path,
    get_market_trend_walkforward_detail_path,
    get_market_trend_walkforward_summary_path,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_SYMBOLS = get_all_symbols()

CLASSIFIER_BUILDERS = {
    "market_trend_logistic": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced", C=0.5)),
        ]
    ),
    "market_trend_random_forest": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
            )),
        ]
    ),
    "market_trend_gradient_boosting": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            )),
        ]
    ),
}

REGRESSOR_BUILDERS = {
    "market_trend_linear_regression": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    ),
    "market_trend_random_forest_regression": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
            )),
        ]
    ),
    "market_trend_gradient_boosting_regression": lambda: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            )),
        ]
    ),
}


def _prediction_to_posture(predicted_return):
    if predicted_return >= BUY_THRESHOLD:
        return "buy"
    if predicted_return <= DONT_BUY_THRESHOLD:
        return "dont_buy"
    return "hold"


def _feature_matrix(df):
    available_columns = [
        column for column in MARKET_FEATURE_COLUMNS
        if column in df.columns and df[column].notna().any()
    ]
    return df[available_columns].copy(), available_columns


def _load_labeled_market(symbol, timeframe, start_date, end_date):
    path = get_labeled_market_path(symbol, timeframe, start_date, end_date)
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return df.sort_values("open_time").reset_index(drop=True)


def _persist_classification_outputs(symbol, timeframe, model_name, start_date, end_date, y_true, y_pred):
    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, symbol, timeframe)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)
    metrics_df.to_csv(get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date), index=False)
    confusion_df.to_csv(get_confusion_matrix_path(symbol, timeframe, model_name, start_date, end_date))

    metric_map = dict(zip(metrics_df["metric"], metrics_df["value"]))
    return {
        "accuracy": float(metric_map["accuracy"]),
        "macro_f1": float(metric_map["macro_f1"]),
        "balanced_accuracy": float(metric_map["balanced_accuracy"]),
        "weighted_f1": float(metric_map["weighted_f1"]),
    }


def _extract_feature_review_rows(model_name, pipeline, feature_names, top_n=10):
    if model_name == "market_trend_logistic":
        classifier = pipeline.named_steps["classifier"]
        scores = pd.Series(abs(classifier.coef_).mean(axis=0), index=feature_names).sort_values(ascending=False)
    elif model_name == "market_trend_random_forest":
        classifier = pipeline.named_steps["classifier"]
        scores = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif model_name == "market_trend_gradient_boosting":
        classifier = pipeline.named_steps["classifier"]
        scores = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
    else:
        return []

    return [
        {
            "model_name": model_name,
            "feature_name": feature_name,
            "importance_score": float(score),
            "feature_rank": rank,
        }
        for rank, (feature_name, score) in enumerate(scores.head(top_n).items(), start=1)
    ]


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


def _evaluate_classifiers_for_symbol(df, symbol, timeframe, start_date, end_date):
    X, feature_names = _feature_matrix(df)
    y = df["label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    rows = []
    feature_rows = []

    for model_name, builder in CLASSIFIER_BUILDERS.items():
        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = _persist_classification_outputs(symbol, timeframe, model_name, start_date, end_date, y_test, y_pred)
        latest_prediction = model.predict(X.iloc[[-1]])[0]
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "window_start": start_date,
                "window_end": end_date,
                **metrics,
                "latest_prediction": latest_prediction,
                "latest_close": float(df.iloc[-1]["close"]),
                "family_role": "market_only_benchmark",
            }
        )
        feature_rows.extend(_extract_feature_review_rows(model_name, model, feature_names))

    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def _evaluate_regressors_for_symbol(df, symbol, timeframe, start_date, end_date):
    X, _ = _feature_matrix(df)
    y = pd.to_numeric(df["future_return"], errors="coerce")
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    rows = []

    for model_name, builder in REGRESSOR_BUILDERS.items():
        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        latest_predicted_return = float(model.predict(X.iloc[[-1]])[0])
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": model_name,
                "window_start": start_date,
                "window_end": end_date,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": float(r2_score(y_test, y_pred)),
                "directional_accuracy": float(((pd.Series(y_pred) > 0) == (pd.Series(y_test).reset_index(drop=True) > 0)).mean()),
                "latest_predicted_return": latest_predicted_return,
                "latest_predicted_posture": _prediction_to_posture(latest_predicted_return),
            }
        )

    return pd.DataFrame(rows)


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


def _evaluate_walkforward_for_symbol(df, symbol, timeframe, start_date, end_date, selected_model_name):
    X, _ = _feature_matrix(df)
    y = df["label"].reset_index(drop=True)
    detail_rows = []

    for fold in _walkforward_folds(len(df), timeframe):
        X_train = X.iloc[:fold["train_end"]]
        X_test = X.iloc[fold["train_end"]:fold["test_end"]]
        y_train = y.iloc[:fold["train_end"]]
        y_test = y.iloc[fold["train_end"]:fold["test_end"]]
        model = CLASSIFIER_BUILDERS[selected_model_name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        detail_rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_name": selected_model_name,
                "window_start": start_date,
                "window_end": end_date,
                "fold_number": fold["fold_number"],
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "fold_accuracy": float(accuracy_score(y_test, y_pred)),
                "fold_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "fold_balanced_accuracy": float(_safe_balanced_accuracy(y_test, y_pred)),
                "test_start_time": df.iloc[fold["train_end"]]["open_time"],
                "test_end_time": df.iloc[fold["test_end"] - 1]["open_time"],
            }
        )

    return pd.DataFrame(detail_rows)


def evaluate_market_trend_forecast(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Run the market-only benchmark family with classification, regression, and walk-forward checks."""
    symbols = tuple(symbols or SUPPORTED_SYMBOLS)
    summary_frames = []
    regression_frames = []
    feature_review_frames = []
    walkforward_frames = []

    for symbol in symbols:
        df = _load_labeled_market(symbol, timeframe, start_date, end_date)
        classifier_df, feature_review_df = _evaluate_classifiers_for_symbol(df, symbol, timeframe, start_date, end_date)
        regression_df = _evaluate_regressors_for_symbol(df, symbol, timeframe, start_date, end_date)

        best_classifier = (
            classifier_df.sort_values(["macro_f1", "accuracy", "balanced_accuracy"], ascending=False)
            .iloc[0]
        )
        walkforward_df = _evaluate_walkforward_for_symbol(
            df,
            symbol,
            timeframe,
            start_date,
            end_date,
            best_classifier["model_name"],
        )

        feature_review_df.insert(0, "symbol", symbol)
        feature_review_df.insert(1, "timeframe", timeframe)
        feature_review_df.insert(2, "window_start", start_date)
        feature_review_df.insert(3, "window_end", end_date)
        feature_review_df.to_csv(
            get_market_trend_feature_review_path(symbol, timeframe, start_date, end_date),
            index=False,
        )

        summary_frames.append(classifier_df)
        regression_frames.append(regression_df)
        feature_review_frames.append(feature_review_df)
        walkforward_frames.append(walkforward_df)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    regression_df = pd.concat(regression_frames, ignore_index=True)
    walkforward_detail_df = pd.concat(walkforward_frames, ignore_index=True)

    if walkforward_detail_df.empty:
        walkforward_summary_df = pd.DataFrame(
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
                    "walkforward_summary": (
                        f"{row['symbol']} market-only walk-forward was skipped on {row['timeframe']} "
                        "because the available labelled window is too short for reliable rolling folds."
                    ),
                }
                for _, row in summary_df.sort_values(["symbol", "macro_f1"], ascending=[True, False])
                .drop_duplicates("symbol")
                .iterrows()
            ]
        )
    else:
        walkforward_summary_df = (
            walkforward_detail_df.groupby(["symbol", "timeframe", "model_name", "window_start", "window_end"], as_index=False)
            .agg(
                walkforward_fold_count=("fold_number", "count"),
                walkforward_avg_accuracy=("fold_accuracy", "mean"),
                walkforward_avg_macro_f1=("fold_macro_f1", "mean"),
                walkforward_avg_balanced_accuracy=("fold_balanced_accuracy", "mean"),
            )
        )
        walkforward_summary_df["walkforward_summary"] = walkforward_summary_df.apply(
            lambda row: (
                f"{row['symbol']} market-only walk-forward used {int(row['walkforward_fold_count'])} folds with "
                f"average accuracy {row['walkforward_avg_accuracy'] * 100:.1f}% and macro-F1 "
                f"{row['walkforward_avg_macro_f1'] * 100:.1f}%."
            ),
            axis=1,
        )

    summary_df.to_csv(get_market_trend_forecast_summary_path(timeframe, start_date, end_date), index=False)
    regression_df.to_csv(get_market_trend_regression_summary_path(timeframe, start_date, end_date), index=False)
    walkforward_detail_df.to_csv(get_market_trend_walkforward_detail_path(timeframe, start_date, end_date), index=False)
    walkforward_summary_df.to_csv(get_market_trend_walkforward_summary_path(timeframe, start_date, end_date), index=False)

    print("market trend forecast summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {get_market_trend_forecast_summary_path(timeframe, start_date, end_date)}")
    print("market trend regression summary generated")
    print(f"rows saved: {len(regression_df)}")
    print(f"summary saved to: {get_market_trend_regression_summary_path(timeframe, start_date, end_date)}")
    print("market trend walk-forward detail generated")
    print(f"rows saved: {len(walkforward_detail_df)}")
    print(f"detail saved to: {get_market_trend_walkforward_detail_path(timeframe, start_date, end_date)}")
    print("market trend walk-forward summary generated")
    print(f"rows saved: {len(walkforward_summary_df)}")
    print(f"summary saved to: {get_market_trend_walkforward_summary_path(timeframe, start_date, end_date)}")
    return {
        "classification_summary": summary_df,
        "regression_summary": regression_df,
        "walkforward_summary": walkforward_summary_df,
    }


if __name__ == "__main__":
    evaluate_market_trend_forecast()
