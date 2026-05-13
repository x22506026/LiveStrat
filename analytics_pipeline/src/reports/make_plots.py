"""Create lightweight market summary outputs for dashboard and analytics use."""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    MARKET_FEATURE_COLUMNS,
    TRAIN_RATIO,
    get_all_symbols,
    get_market_features_path,
    get_labeled_market_path,
    get_market_overview_path,
    get_market_summary_path,
)
from src.models.evaluate import make_time_based_split
from src.models.evaluate_rule_based_strategy import apply_rule_based_strategy


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def classify_trend(latest_row):
    """Describe the latest market trend in a simple explainable way."""
    if (
        latest_row["close"] > latest_row["sma_20"] > latest_row["sma_50"]
        and latest_row["return_24h"] > 0
    ):
        return "bullish"

    if (
        latest_row["close"] < latest_row["sma_20"] < latest_row["sma_50"]
        and latest_row["return_24h"] < 0
    ):
        return "bearish"

    return "neutral"


def classify_volatility(latest_volatility, series):
    """Bucket current volatility against the asset's own historical range."""
    low_threshold = series.quantile(0.33)
    high_threshold = series.quantile(0.66)

    if latest_volatility >= high_threshold:
        return "high"
    if latest_volatility <= low_threshold:
        return "low"
    return "medium"


def train_unscaled_baseline(X_train, y_train):
    return LogisticRegression(max_iter=5000, solver="lbfgs").fit(X_train, y_train)


def train_scaled_baseline(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def build_feature_importance_from_scaled_model(model, feature_names):
    """Extract average absolute logistic-regression importance from the scaled model."""
    classifier = model.named_steps["classifier"]
    coef_df = pd.DataFrame(classifier.coef_, columns=feature_names, index=classifier.classes_)
    return coef_df.abs().mean(axis=0).sort_values(ascending=False)


def build_analysis_summary(symbol, timeframe, latest_row, trend_status, volatility_status,
                           rule_signal, model_signal, top_feature_name):
    """Create a human-readable summary sentence for future app use."""
    latest_return_pct = latest_row["return_24h"] * 100
    return (
        f"{symbol} on {timeframe} is currently {trend_status} with {volatility_status} volatility. "
        f"The latest 24h return is {latest_return_pct:.2f}%. "
        f"The rule-based view is {rule_signal} while the scaled baseline suggests {model_signal}. "
        f"The most influential feature right now is {top_feature_name}."
    )


def generate_market_summary_for_symbol(symbol, timeframe=TIMEFRAME, start_date=None, end_date=None):
    """Build a compact market summary table for one symbol."""
    labeled_input_file = get_labeled_market_path(symbol, timeframe, start_date, end_date)
    labeled_df = pd.read_csv(labeled_input_file, parse_dates=["open_time"])
    labeled_df = labeled_df.sort_values("open_time").reset_index(drop=True)
    labeled_df = apply_rule_based_strategy(labeled_df)

    latest_input_file = get_market_features_path(symbol, timeframe, start_date, end_date)
    latest_df = pd.read_csv(latest_input_file, parse_dates=["open_time"])
    latest_df = latest_df.sort_values("open_time").reset_index(drop=True)
    latest_df = apply_rule_based_strategy(latest_df)

    X = labeled_df[MARKET_FEATURE_COLUMNS].copy()
    y = labeled_df["label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)

    unscaled_model = train_unscaled_baseline(X_train, y_train)
    scaled_model = train_scaled_baseline(X_train, y_train)

    unscaled_accuracy = accuracy_score(y_test, unscaled_model.predict(X_test))
    scaled_accuracy = accuracy_score(y_test, scaled_model.predict(X_test))

    split_idx = int(len(labeled_df) * TRAIN_RATIO)
    rule_accuracy = accuracy_score(y_test, labeled_df["rule_signal"].iloc[split_idx:])

    latest_row = latest_df.iloc[-1]
    latest_features = latest_df[MARKET_FEATURE_COLUMNS].iloc[[-1]]
    latest_scaled_prediction = scaled_model.predict(latest_features)[0]
    latest_scaled_proba = float(scaled_model.predict_proba(latest_features).max())
    latest_rule_signal = latest_row["rule_signal"]

    trend_status = classify_trend(latest_row)
    volatility_status = classify_volatility(latest_row["volatility_20"], latest_df["volatility_20"])

    feature_importance = build_feature_importance_from_scaled_model(scaled_model, MARKET_FEATURE_COLUMNS)
    top_feature_name = feature_importance.index[0]
    top_feature_importance = float(feature_importance.iloc[0])

    summary_row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_start": start_date or "",
        "window_end": end_date or "",
        "latest_open_time": latest_row["open_time"],
        "latest_close": float(latest_row["close"]),
        "latest_return_4h_pct": float(latest_row["return_4h"] * 100),
        "latest_return_24h_pct": float(latest_row["return_24h"] * 100),
        "latest_return_3d_pct": float(latest_row["return_3d"] * 100),
        "latest_volatility_20": float(latest_row["volatility_20"]),
        "latest_volume_zscore": float(latest_row["volume_zscore"]),
        "latest_high_low_range_pct": float(latest_row["high_low_range_pct"] * 100),
        "latest_taker_buy_ratio": float(latest_row["taker_buy_volume_ratio"]),
        "trend_status": trend_status,
        "volatility_status": volatility_status,
        "rule_signal": latest_rule_signal,
        "scaled_model_signal": latest_scaled_prediction,
        "scaled_model_confidence": latest_scaled_proba,
        "rule_based_test_accuracy": float(rule_accuracy),
        "baseline_unscaled_test_accuracy": float(unscaled_accuracy),
        "baseline_scaled_test_accuracy": float(scaled_accuracy),
        "top_feature_name": top_feature_name,
        "top_feature_importance": top_feature_importance,
    }
    summary_row["analysis_summary"] = build_analysis_summary(
        symbol,
        timeframe,
        latest_row,
        trend_status,
        volatility_status,
        latest_rule_signal,
        latest_scaled_prediction,
        top_feature_name,
    )

    summary_df = pd.DataFrame([summary_row])
    summary_path = get_market_summary_path(symbol, timeframe, start_date, end_date)
    summary_df.to_csv(summary_path, index=False)

    print("market summary generated")
    print(f"symbol: {symbol}")
    print(f"summary saved to: {summary_path}")

    return summary_df


def generate_market_reports(timeframe=TIMEFRAME, start_date=None, end_date=None, symbols=None):
    """Generate one summary per configured symbol plus a combined overview file."""
    selected_symbols = symbols or get_all_symbols()
    summaries = []
    for symbol in selected_symbols:
        summaries.append(generate_market_summary_for_symbol(symbol, timeframe, start_date, end_date))

    overview_df = pd.concat(summaries, ignore_index=True)
    overview_path = get_market_overview_path(timeframe, start_date, end_date)
    overview_df.to_csv(overview_path, index=False)

    print("combined market overview generated")
    print(f"overview saved to: {overview_path}")
    return overview_df


if __name__ == "__main__":
    generate_market_reports()
