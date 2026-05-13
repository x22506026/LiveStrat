"""Evaluate first-pass market + futures strategies on aligned recent datasets."""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    FUTURES_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    get_all_symbols,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_futures_dataset_path,
    get_market_futures_strategy_summary_path,
    TRAIN_RATIO,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_STRATEGY_SYMBOLS = get_all_symbols()

MARKET_FUTURES_FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS + FUTURES_FEATURE_COLUMNS + [
    "futures_number_of_trades",
    "funding_mark_price",
    "circulating_supply",
    "futures_price",
    "index_price",
    "mark_spot_spread_pct",
    "futures_price_spot_spread_pct",
    "funding_mark_spot_spread_pct",
    "mark_spot_spread_zscore_20",
    "futures_price_spot_spread_zscore_20",
    "funding_mark_spot_spread_zscore_20",
    "funding_oi_pressure",
    "basis_momentum_agreement",
    "taker_pressure_return_alignment",
    "market_futures_alignment_score",
]


def apply_market_futures_regime_filter(df):
    """Follow trend only when futures positioning supports the move."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["close"] > df["sma_50"]) &
        (df["return_24h"] > 0) &
        (df["mark_return_24h"] > 0) &
        (df["futures_activity_score"] > -0.25) &
        (df["taker_buy_sell_ratio_zscore_21"] > -0.25) &
        (df["market_futures_alignment_score"] > -0.25)
    )
    dont_buy_condition = (
        (df["close"] < df["sma_50"]) &
        (df["return_24h"] < 0) &
        (df["mark_return_24h"] < 0) &
        (df["futures_activity_score"] > -0.25) &
        (df["taker_buy_sell_ratio_zscore_21"] < 0.0) &
        (df["market_futures_alignment_score"] < 0.0)
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_market_futures_crowding_reversal(df):
    """Look for price/futures disagreement as a cautionary reversal signal."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["close"] > df["sma_50"]) &
        (df["return_4h"] < 0) &
        (df["funding_rate_zscore_21"] < 0) &
        (df["basis_rate_zscore_21"] < 0) &
        (df["taker_buy_sell_ratio_zscore_21"] > 0)
    )
    dont_buy_condition = (
        (df["close"] < df["sma_50"]) &
        (df["return_4h"] > 0) &
        (df["funding_rate_zscore_21"] > 0) &
        (df["long_short_ratio_zscore_21"] > 0) &
        (df["taker_buy_sell_ratio_zscore_21"] < 0)
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def build_combined_feature_matrix(df):
    """Select only features that exist in the current merged dataset."""
    available_columns = [
        column
        for column in MARKET_FUTURES_FEATURE_COLUMNS
        if column in df.columns and df[column].notna().any()
    ]
    return df[available_columns].copy(), available_columns


def train_scaled_logistic_baseline(X_train, y_train):
    """Train a simple interpretable linear baseline on the combined feature set."""
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                class_weight="balanced",
                C=0.5,
            )),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_baseline(X_train, y_train):
    """Train a nonlinear market + futures baseline."""
    model = Pipeline(
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
    )
    model.fit(X_train, y_train)
    return model


def evaluate_prediction_vectors(y_true, y_pred, symbol, model_name, timeframe, start_date, end_date):
    """Persist evaluation outputs and return headline metrics."""
    metrics_df = build_metrics_dataframe(
        y_true,
        y_pred,
        model_name,
        symbol,
        timeframe,
    )
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)

    metrics_df.to_csv(
        get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date),
        index=False,
    )
    confusion_df.to_csv(
        get_confusion_matrix_path(symbol, timeframe, model_name, start_date, end_date)
    )
    print_evaluation_summary(f"{symbol} {model_name} evaluation", y_true, y_pred)

    accuracy = float(metrics_df.loc[metrics_df["metric"] == "accuracy", "value"].iloc[0])
    macro_f1 = float(metrics_df.loc[metrics_df["metric"] == "macro_f1", "value"].iloc[0])
    return accuracy, macro_f1


def evaluate_rule_strategy(df, symbol, model_name, timeframe, start_date, end_date):
    """Evaluate a rule-based signal column using the shared time-based split."""
    _, _, _, y_true = make_time_based_split(df[["close"]], df["label"], TRAIN_RATIO)
    split_idx = int(len(df) * TRAIN_RATIO)
    y_pred = df["strategy_signal"].iloc[split_idx:]
    return evaluate_prediction_vectors(y_true, y_pred, symbol, model_name, timeframe, start_date, end_date)


def evaluate_model_strategy(df, symbol, model_name, trainer, timeframe, start_date, end_date):
    """Evaluate one learned model on the combined market + futures feature set."""
    X, feature_names = build_combined_feature_matrix(df)
    y = df["label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    model = trainer(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, macro_f1 = evaluate_prediction_vectors(
        y_test,
        y_pred,
        symbol,
        model_name,
        timeframe,
        start_date,
        end_date,
    )

    latest_features = X.iloc[[-1]]
    latest_prediction = model.predict(latest_features)[0]

    top_feature_name = "n/a"
    if model_name == "market_futures_random_forest":
        classifier = model.named_steps["classifier"]
        importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
        if not importances.empty:
            top_feature_name = importances.index[0]

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "latest_prediction": latest_prediction,
        "top_feature_name": top_feature_name,
    }


def build_strategy_summary_row(symbol, strategy_name, latest_signal, accuracy, macro_f1,
                               extra_note, timeframe, start_date, end_date, target_config):
    """Create one compact summary row for app/research use."""
    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "timeframe": timeframe,
        "window_start": start_date,
        "window_end": end_date,
        "target_name": target_config["target_name"],
        "horizon_steps": target_config["horizon_steps"],
        "target_horizon_hours": target_config.get("effective_horizon_hours"),
        "target_exact_horizon_match": target_config.get("exact_horizon_match"),
        "target_resolution_note": target_config.get("horizon_resolution_note"),
        "latest_signal": latest_signal,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "strategy_summary": extra_note,
    }


def evaluate_market_futures_strategies(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Run the first market + futures strategy suite on the recent aligned datasets."""
    summary_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    symbols = tuple(symbols or SUPPORTED_STRATEGY_SYMBOLS)

    for symbol in symbols:
        dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
        if not dataset_path.exists():
            print(f"skipping {symbol}: market + futures dataset not found at {dataset_path}")
            continue
        df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
        df = df[df["futures_data_available"] == True].copy()
        if df.empty:
            print(f"skipping {symbol}: futures-aligned rows are unavailable")
            continue
        df = df.sort_values("open_time").reset_index(drop=True)
        target_config = preferred_targets[symbol]
        df = build_target_labels(df, target_config, timeframe=timeframe)

        regime_filter_df = apply_market_futures_regime_filter(df)
        regime_accuracy, regime_macro_f1 = evaluate_rule_strategy(
            regime_filter_df,
            symbol,
            "market_futures_regime_filter",
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_strategy_summary_row(
                symbol,
                "market_futures_regime_filter",
                regime_filter_df.iloc[-1]["strategy_signal"],
                regime_accuracy,
                regime_macro_f1,
                (
                    f"{symbol} regime filter currently suggests {regime_filter_df.iloc[-1]['strategy_signal']}. "
                    "It follows spot trend only when futures positioning confirms the move."
                ),
                timeframe,
                start_date,
                end_date,
                target_config,
            )
        )

        reversal_df = apply_market_futures_crowding_reversal(df)
        reversal_accuracy, reversal_macro_f1 = evaluate_rule_strategy(
            reversal_df,
            symbol,
            "market_futures_crowding_reversal",
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_strategy_summary_row(
                symbol,
                "market_futures_crowding_reversal",
                reversal_df.iloc[-1]["strategy_signal"],
                reversal_accuracy,
                reversal_macro_f1,
                (
                    f"{symbol} crowding reversal currently suggests {reversal_df.iloc[-1]['strategy_signal']}. "
                    "It looks for crowded futures positioning that may justify caution or reversal."
                ),
                timeframe,
                start_date,
                end_date,
                target_config,
            )
        )

        logistic_results = evaluate_model_strategy(
            df,
            symbol,
            "market_futures_logistic",
            train_scaled_logistic_baseline,
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_strategy_summary_row(
                symbol,
                "market_futures_logistic",
                logistic_results["latest_prediction"],
                logistic_results["accuracy"],
                logistic_results["macro_f1"],
                (
                    f"{symbol} logistic baseline currently suggests {logistic_results['latest_prediction']}. "
                    "This is the interpretable linear benchmark on the combined market + futures features."
                ),
                timeframe,
                start_date,
                end_date,
                target_config,
            )
        )

        forest_results = evaluate_model_strategy(
            df,
            symbol,
            "market_futures_random_forest",
            train_random_forest_baseline,
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_strategy_summary_row(
                symbol,
                "market_futures_random_forest",
                forest_results["latest_prediction"],
                forest_results["accuracy"],
                forest_results["macro_f1"],
                (
                    f"{symbol} random forest currently suggests {forest_results['latest_prediction']}. "
                    f"The top recent feature is {forest_results['top_feature_name']}."
                ),
                timeframe,
                start_date,
                end_date,
                target_config,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = get_market_futures_strategy_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(summary_path, index=False)

    print("market + futures strategy summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_strategies()
