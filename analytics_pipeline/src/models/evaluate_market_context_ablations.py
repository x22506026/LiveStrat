"""Evaluate how each context source changes market + futures benchmark quality."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_all_symbols,
    get_market_context_ablation_summary_path,
    get_market_multimodal_dataset_path,
)
from src.models.evaluate import build_metrics_dataframe, make_time_based_split
from src.models.evaluate_market_futures_strategies import (
    build_combined_feature_matrix,
    train_scaled_logistic_baseline,
)
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_SYMBOLS = get_all_symbols()

FEAR_GREED_COLUMNS = [
    "sentiment_value",
    "sentiment_change_1d",
    "sentiment_change_7d",
    "sentiment_rolling_mean_7d",
    "sentiment_zscore_30d",
    "sentiment_regime_score",
    "sentiment_supportive_flag",
    "sentiment_risk_off_flag",
    "sentiment_context_available",
]

GDELT_COLUMNS = [
    "gdelt_article_count",
    "gdelt_unique_domain_count",
    "gdelt_sentiment_mean",
    "gdelt_sentiment_std",
    "gdelt_positive_share",
    "gdelt_negative_share",
    "gdelt_neutral_share",
    "gdelt_sentiment_momentum_3d",
    "gdelt_sentiment_momentum_7d",
    "gdelt_article_count_zscore_30d",
    "gdelt_article_count_3d_sum",
    "gdelt_article_count_7d_sum",
    "gdelt_positive_keyword_share",
    "gdelt_negative_keyword_share",
    "gdelt_keyword_intensity_mean",
    "gdelt_source_concentration",
    "gdelt_coverage_quality_score",
    "gdelt_event_intensity_score",
    "news_source_count",
    "news_rss_share",
    "news_gdelt_share",
    "gdelt_regime_score",
    "gdelt_supportive_flag",
    "gdelt_risk_off_flag",
    "gdelt_context_available",
]

ONCHAIN_COLUMNS = [
    "active_addresses_value",
    "transaction_count_value",
    "economic_activity_value",
    "valuation_ratio",
    "exchange_netflow_value",
    "exchange_netflow_zscore_30d",
    "onchain_regime_score",
    "onchain_supportive_flag",
    "onchain_risk_off_flag",
    "onchain_context_available",
]

DEFI_COLUMNS = [
    "defi_tvl_usd",
    "defi_tvl_change_1d",
    "defi_tvl_change_pct_1d",
    "defi_tvl_change_pct_7d",
    "defi_tvl_change_pct_30d",
    "defi_tvl_zscore_30d",
    "defi_tvl_drawdown_30d",
    "defi_regime_score",
    "defi_supportive_flag",
    "defi_risk_off_flag",
    "defi_context_available",
]


def get_metric_value(metrics_df, metric_name):
    """Read a single metric value from the generic evaluation dataframe."""
    return float(metrics_df.loc[metrics_df["metric"] == metric_name, "value"].iloc[0])


def unique_present_columns(df, columns):
    """Keep only context columns that exist and have at least one non-null value."""
    return [
        column
        for column in columns
        if column in df.columns and df[column].notna().any()
    ]


def build_ablation_feature_sets(df):
    """Create the feature subsets used by the context ablation study."""
    base_X, base_columns = build_combined_feature_matrix(df)
    sentiment_available = (
        "sentiment_context_available" in df.columns and df["sentiment_context_available"].eq(True).any()
    )
    gdelt_available = (
        "gdelt_context_available" in df.columns and df["gdelt_context_available"].eq(True).any()
    )
    onchain_available = (
        "onchain_context_available" in df.columns and df["onchain_context_available"].eq(True).any()
    )
    defi_available = (
        "defi_context_available" in df.columns and df["defi_context_available"].eq(True).any()
    )

    fear_greed_columns = unique_present_columns(df, FEAR_GREED_COLUMNS) if sentiment_available else []
    gdelt_columns = unique_present_columns(df, GDELT_COLUMNS) if gdelt_available else []
    onchain_columns = unique_present_columns(df, ONCHAIN_COLUMNS) if onchain_available else []
    defi_columns = unique_present_columns(df, DEFI_COLUMNS) if defi_available else []

    full_context_columns = []
    for column in fear_greed_columns + gdelt_columns + onchain_columns + defi_columns:
        if column not in full_context_columns:
            full_context_columns.append(column)

    feature_sets = {"market_futures_only": base_columns}

    if fear_greed_columns:
        feature_sets["market_futures_plus_fear_greed"] = (
            base_columns + [column for column in fear_greed_columns if column not in base_columns]
        )
    if gdelt_columns:
        feature_sets["market_futures_plus_gdelt"] = (
            base_columns + [column for column in gdelt_columns if column not in base_columns]
        )
    if onchain_columns:
        feature_sets["market_futures_plus_onchain"] = (
            base_columns + [column for column in onchain_columns if column not in base_columns]
        )
    if defi_columns:
        feature_sets["market_futures_plus_defi"] = (
            base_columns + [column for column in defi_columns if column not in base_columns]
        )
    if full_context_columns:
        feature_sets["full_multimodal"] = (
            base_columns + [column for column in full_context_columns if column not in base_columns]
        )

    return df[base_columns].copy(), feature_sets


def evaluate_feature_variant(df, feature_columns):
    """Train and evaluate the balanced logistic benchmark for one feature subset."""
    X = df[feature_columns].copy()
    y = df["label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    model = train_scaled_logistic_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_df = build_metrics_dataframe(y_test, y_pred, "context_ablation", "asset", TIMEFRAME)
    return {
        "accuracy": get_metric_value(metrics_df, "accuracy"),
        "macro_f1": get_metric_value(metrics_df, "macro_f1"),
        "balanced_accuracy": get_metric_value(metrics_df, "balanced_accuracy"),
        "latest_signal": model.predict(X.iloc[[-1]])[0],
        "latest_signal_confidence": float(model.predict_proba(X.iloc[[-1]]).max()),
        "feature_count": len(feature_columns),
    }


def evaluate_market_context_ablations(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Compare market + futures against each added context source and the full multimodal set."""
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    summary_rows = []

    for symbol in SUPPORTED_SYMBOLS:
        dataset_path = get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date)
        df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        df = build_target_labels(df, preferred_targets[symbol], timeframe=timeframe)

        _, feature_sets = build_ablation_feature_sets(df)
        variant_results = {}
        for variant_name, feature_columns in feature_sets.items():
            variant_results[variant_name] = evaluate_feature_variant(df, feature_columns)

        base_macro_f1 = variant_results["market_futures_only"]["macro_f1"]
        base_balanced_accuracy = variant_results["market_futures_only"]["balanced_accuracy"]

        for variant_name, result in variant_results.items():
            summary_rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "window_start": start_date,
                    "window_end": end_date,
                    "variant_name": variant_name,
                    "feature_count": result["feature_count"],
                    "latest_signal": result["latest_signal"],
                    "latest_signal_confidence": result["latest_signal_confidence"],
                    "test_accuracy": result["accuracy"],
                    "test_macro_f1": result["macro_f1"],
                    "test_balanced_accuracy": result["balanced_accuracy"],
                    "delta_macro_f1_vs_market_futures": result["macro_f1"] - base_macro_f1,
                    "delta_balanced_accuracy_vs_market_futures": result["balanced_accuracy"] - base_balanced_accuracy,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    output_path = get_market_context_ablation_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(output_path, index=False)

    print("market context ablation summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {output_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_context_ablations()
