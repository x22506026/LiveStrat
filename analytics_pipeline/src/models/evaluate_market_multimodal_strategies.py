"""Evaluate first-pass market + futures + sentiment + on-chain strategies."""

from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_all_symbols,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_multimodal_dataset_path,
    get_market_multimodal_strategy_summary_path,
    get_strategy_summary_path,
)
from src.models.evaluate_market_context_ablations import build_ablation_feature_sets
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
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SUPPORTED_STRATEGY_SYMBOLS = get_all_symbols()
SELECTION_TRAIN_RATIO = 0.55
SELECTION_VALIDATION_RATIO = 0.15

MULTIMODAL_CONTEXT_COLUMNS = [
    "sentiment_value",
    "sentiment_change_1d",
    "sentiment_change_7d",
    "sentiment_rolling_mean_7d",
    "sentiment_zscore_30d",
    "sentiment_regime_score",
    "sentiment_supportive_flag",
    "sentiment_risk_off_flag",
    "sentiment_context_available",
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
    "news_event_risk_score",
    "news_event_supportive_score",
    "news_event_risk_flag",
    "news_event_supportive_flag",
    "gdelt_regime_score",
    "gdelt_supportive_flag",
    "gdelt_risk_off_flag",
    "gdelt_context_available",
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
    "multimodal_confirmation_count",
    "multimodal_risk_off_count",
    "multimodal_context_score",
    "multimodal_context_available",
]


def load_context_reliability_map():
    """Load the daily context-reliability summary if it exists."""
    path = get_strategy_summary_path("context_reliability", "1d")
    if not Path(path).exists():
        return {}

    reliability_df = pd.read_csv(path)
    if reliability_df.empty:
        return {}
    return {
        str(row["market_symbol"]): row.to_dict()
        for _, row in reliability_df.iterrows()
    }


def get_metric_value(metrics_df, metric_name):
    """Read one scalar metric value from the shared metrics dataframe."""
    return float(metrics_df.loc[metrics_df["metric"] == metric_name, "value"].iloc[0])


def make_train_validation_test_split(X, y,
                                     train_ratio=SELECTION_TRAIN_RATIO,
                                     validation_ratio=SELECTION_VALIDATION_RATIO):
    """Create chronological train/validation/test splits for model selection."""
    train_end_idx = int(len(X) * train_ratio)
    validation_end_idx = int(len(X) * (train_ratio + validation_ratio))
    return (
        X.iloc[:train_end_idx],
        X.iloc[train_end_idx:validation_end_idx],
        X.iloc[validation_end_idx:],
        y.iloc[:train_end_idx],
        y.iloc[train_end_idx:validation_end_idx],
        y.iloc[validation_end_idx:],
    )


def rank_result_key(result):
    """Rank strategies using the same headline order throughout the project."""
    return (
        result["macro_f1"],
        result["balanced_accuracy"],
        result["accuracy"],
        result.get("latest_signal_confidence", 0.0),
    )


def apply_multimodal_confirmation_gate(df):
    """Require market/futures structure and context layers to agree before acting."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["close"] > df["sma_50"]) &
        (df["return_24h"] > 0) &
        (df["market_futures_alignment_score"] > 0.0) &
        (df["multimodal_confirmation_count"] >= 1) &
        (df["multimodal_risk_off_count"] == 0)
    )
    dont_buy_condition = (
        (df["close"] < df["sma_50"]) &
        (df["return_24h"] < 0) &
        (df["market_futures_alignment_score"] < 0.0) &
        (df["multimodal_risk_off_count"] >= 1)
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_multimodal_context_veto(df):
    """Let broad context veto otherwise bullish conditions when sentiment/on-chain deteriorate."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_futures_alignment_score"] > 0.35) &
        (df["multimodal_context_score"] >= 0.0) &
        (df["multimodal_risk_off_count"] == 0)
    )
    dont_buy_condition = (
        (df["multimodal_context_score"] <= -0.5) |
        (
            (df["market_futures_alignment_score"] < -0.25) &
            (df["multimodal_risk_off_count"] >= 1)
        )
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_news_event_risk_overlay(df):
    """Use asset-news as an event-risk veto or selective confirmation layer."""
    df = df.copy()
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_futures_alignment_score"] > 0.30) &
        (df["news_event_risk_flag"] == False) &
        (
            (df["news_event_supportive_flag"] == True) |
            (df["onchain_supportive_flag"] == True)
        )
    )
    dont_buy_condition = (
        (df["news_event_risk_flag"] == True) |
        (
            (df["market_futures_alignment_score"] < -0.20) &
            (df["multimodal_risk_off_count"] >= 1)
        )
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def build_multimodal_feature_matrix(df):
    """Extend the market + futures feature block with context features when available."""
    base_X, base_columns = build_combined_feature_matrix(df)
    extra_columns = [
        column for column in MULTIMODAL_CONTEXT_COLUMNS
        if column in df.columns and df[column].notna().any()
    ]
    feature_columns = base_columns + [column for column in extra_columns if column not in base_columns]
    return df[feature_columns].copy(), feature_columns


def evaluate_prediction_vectors(y_true, y_pred, symbol, model_name, timeframe, start_date, end_date):
    """Persist evaluation outputs and return headline metrics."""
    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, symbol, timeframe)
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
    balanced_accuracy = float(
        metrics_df.loc[metrics_df["metric"] == "balanced_accuracy", "value"].iloc[0]
    )
    return accuracy, macro_f1, balanced_accuracy


def evaluate_rule_strategy(df, symbol, model_name, timeframe, start_date, end_date):
    """Evaluate a multimodal rule strategy chronologically."""
    _, _, _, y_true = make_time_based_split(df[["close"]], df["label"], TRAIN_RATIO)
    split_idx = int(len(df) * TRAIN_RATIO)
    y_pred = df["strategy_signal"].iloc[split_idx:]
    return evaluate_prediction_vectors(y_true, y_pred, symbol, model_name, timeframe, start_date, end_date)


def evaluate_logistic_strategy(df, symbol, target_name, timeframe, start_date, end_date):
    """Evaluate a logistic benchmark on the full multimodal feature set."""
    X, _ = build_multimodal_feature_matrix(df)
    y = df["label"]
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)
    model = train_scaled_logistic_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name = f"market_multimodal_logistic_{target_name}"
    accuracy, macro_f1, balanced_accuracy = evaluate_prediction_vectors(
        y_test,
        y_pred,
        symbol,
        model_name,
        timeframe,
        start_date,
        end_date,
    )
    latest_signal = model.predict(X.iloc[[-1]])[0]
    latest_signal_confidence = float(model.predict_proba(X.iloc[[-1]]).max())
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "latest_signal": latest_signal,
        "latest_signal_confidence": latest_signal_confidence,
    }


def evaluate_context_specialist_logistics(df, symbol, timeframe, start_date, end_date):
    """Evaluate direct logistic specialists for each context family on the held-out test split."""
    _, feature_sets = build_ablation_feature_sets(df)
    y = df["label"]
    results = []

    for variant_name, feature_columns in feature_sets.items():
        if variant_name == "market_futures_only":
            continue

        X_variant = df[feature_columns].copy()
        X_train, X_test, y_train, y_test = make_time_based_split(X_variant, y, TRAIN_RATIO)
        model = train_scaled_logistic_baseline(X_train, y_train)
        y_pred = model.predict(X_test)
        model_name = f"market_multimodal_specialist_{variant_name}"
        accuracy, macro_f1, balanced_accuracy = evaluate_prediction_vectors(
            y_test,
            y_pred,
            symbol,
            model_name,
            timeframe,
            start_date,
            end_date,
        )
        results.append(
            {
                "model_name": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_accuracy,
                "latest_signal": model.predict(X_variant.iloc[[-1]])[0],
                "latest_signal_confidence": float(model.predict_proba(X_variant.iloc[[-1]]).max()),
                "selected_context_variant": variant_name,
                "selected_context_feature_count": len(feature_columns),
            }
        )

    return results


def evaluate_validation_selected_logistic(df, symbol, timeframe, start_date, end_date):
    """Choose the best context family on validation, then test that selected logistic model."""
    _, feature_sets = build_ablation_feature_sets(df)
    y = df["label"]

    validation_results = {}
    for variant_name, feature_columns in feature_sets.items():
        X_variant = df[feature_columns].copy()
        (
            X_train,
            X_val,
            _,
            y_train,
            y_val,
            _,
        ) = make_train_validation_test_split(X_variant, y)
        model = train_scaled_logistic_baseline(X_train, y_train)
        y_val_pred = model.predict(X_val)
        metrics_df = build_metrics_dataframe(
            y_val,
            y_val_pred,
            f"validation_{variant_name}",
            symbol,
            timeframe,
        )
        validation_results[variant_name] = {
            "feature_columns": feature_columns,
            "accuracy": get_metric_value(metrics_df, "accuracy"),
            "macro_f1": get_metric_value(metrics_df, "macro_f1"),
            "balanced_accuracy": get_metric_value(metrics_df, "balanced_accuracy"),
            "feature_count": len(feature_columns),
        }

    selected_variant_name, selected_variant_result = max(
        validation_results.items(),
        key=lambda item: rank_result_key(item[1]),
    )
    selected_feature_columns = selected_variant_result["feature_columns"]
    X_selected = df[selected_feature_columns].copy()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = make_train_validation_test_split(X_selected, y)
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    model = train_scaled_logistic_baseline(X_train_final, y_train_final)
    y_test_pred = model.predict(X_test)
    model_name = "market_multimodal_validation_selected"
    accuracy, macro_f1, balanced_accuracy = evaluate_prediction_vectors(
        y_test,
        y_test_pred,
        symbol,
        model_name,
        timeframe,
        start_date,
        end_date,
    )
    latest_signal = model.predict(X_selected.iloc[[-1]])[0]
    latest_signal_confidence = float(model.predict_proba(X_selected.iloc[[-1]]).max())
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "latest_signal": latest_signal,
        "latest_signal_confidence": latest_signal_confidence,
        "selected_context_variant": selected_variant_name,
        "selected_context_feature_count": selected_variant_result["feature_count"],
        "validation_macro_f1": selected_variant_result["macro_f1"],
        "validation_balanced_accuracy": selected_variant_result["balanced_accuracy"],
    }


def build_summary_row(symbol, strategy_name, df, accuracy, macro_f1, balanced_accuracy,
                      timeframe, start_date, end_date,
                      latest_signal=None, latest_signal_confidence=None,
                      selected_context_variant=None, selected_context_feature_count=None,
                      validation_macro_f1=None, validation_balanced_accuracy=None,
                      context_selection_method=None, context_reliability_row=None):
    """Create one compact multimodal summary row per strategy."""
    latest = df.iloc[-1]
    resolved_signal = latest_signal or latest.get("strategy_signal", "hold")
    resolved_confidence = latest_signal_confidence if latest_signal_confidence is not None else 1.0
    latest_sentiment_value = latest.get("sentiment_value")
    latest_gdelt_sentiment_value = latest.get("gdelt_sentiment_mean")
    latest_gdelt_article_count = latest.get("gdelt_article_count", 0)
    latest_gdelt_regime_label = latest.get("gdelt_regime_label", "unavailable")
    latest_gdelt_attention_label = latest.get("gdelt_attention_label", "quiet")
    latest_gdelt_coverage_quality = latest.get("gdelt_coverage_quality_score", 0.0)
    latest_gdelt_dominant_event_theme = latest.get("gdelt_dominant_event_theme", "none")
    latest_gdelt_risk_event_theme = latest.get("gdelt_risk_event_theme", "none")
    latest_gdelt_supportive_event_theme = latest.get("gdelt_supportive_event_theme", "none")
    latest_news_event_label = latest.get("news_event_regime_label", "quiet")
    latest_news_event_risk_score = latest.get("news_event_risk_score", 0.0)
    latest_effective_sentiment_source = latest.get("effective_sentiment_source", "unavailable")
    latest_effective_sentiment_label = latest.get("effective_sentiment_label", "unavailable")
    latest_onchain_label = latest.get("onchain_regime_label", "unavailable")
    latest_context_label = latest.get("multimodal_context_label", "mixed")
    context_reliability_row = context_reliability_row or {}
    context_readiness = context_reliability_row.get("combined_context_readiness", "limited_context_confirmation")
    gdelt_reliability_label = context_reliability_row.get("gdelt_reliability_label", "unavailable")
    onchain_reliability_label = context_reliability_row.get("onchain_reliability_label", "unavailable")
    effective_sentiment_role = context_reliability_row.get("effective_sentiment_role", "unavailable")

    if pd.isna(latest_sentiment_value):
        latest_sentiment_value = pd.NA
    if pd.isna(latest_gdelt_sentiment_value):
        latest_gdelt_sentiment_value = pd.NA
    if pd.isna(latest_gdelt_article_count):
        latest_gdelt_article_count = 0
    if pd.isna(latest_gdelt_regime_label):
        latest_gdelt_regime_label = "unavailable"
    if pd.isna(latest_gdelt_attention_label):
        latest_gdelt_attention_label = "quiet"
    if pd.isna(latest_gdelt_coverage_quality):
        latest_gdelt_coverage_quality = 0.0
    if pd.isna(latest_gdelt_dominant_event_theme):
        latest_gdelt_dominant_event_theme = "none"
    if pd.isna(latest_gdelt_risk_event_theme):
        latest_gdelt_risk_event_theme = "none"
    if pd.isna(latest_gdelt_supportive_event_theme):
        latest_gdelt_supportive_event_theme = "none"
    if pd.isna(latest_news_event_label):
        latest_news_event_label = "quiet"
    if pd.isna(latest_news_event_risk_score):
        latest_news_event_risk_score = 0.0
    if pd.isna(latest_effective_sentiment_source):
        latest_effective_sentiment_source = "unavailable"
    if pd.isna(latest_effective_sentiment_label):
        latest_effective_sentiment_label = "unavailable"
    if pd.isna(latest_onchain_label):
        latest_onchain_label = "unavailable"
    if pd.isna(latest_context_label):
        latest_context_label = "mixed"
    if latest_gdelt_regime_label == "unavailable":
        gdelt_summary_text = "asset-specific GDELT news sentiment is unavailable for this asset right now"
    else:
        gdelt_summary_text = (
            f"GDELT news is {latest_gdelt_regime_label} from {int(latest_gdelt_article_count)} articles "
            f"with {str(latest_gdelt_attention_label).replace('_', ' ')} attention and coverage quality "
            f"{float(latest_gdelt_coverage_quality):.2f}, dominant theme "
            f"{str(latest_gdelt_dominant_event_theme).replace('_', ' ')}"
        )
    context_selection_note = ""
    if selected_context_variant:
        if context_selection_method == "validation_selected":
            context_selection_note = (
                f", and the validation-selected context family is "
                f"{selected_context_variant.replace('_', ' ')}"
            )
        elif context_selection_method == "direct_specialist":
            context_selection_note = (
                f", and the current specialist context family is "
                f"{selected_context_variant.replace('_', ' ')}"
            )
        else:
            context_selection_note = (
                f", and the active context family is "
                f"{selected_context_variant.replace('_', ' ')}"
            )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_start": start_date,
        "window_end": end_date,
        "strategy_name": strategy_name,
        "latest_signal": resolved_signal,
        "latest_signal_confidence": resolved_confidence,
        "latest_multimodal_context_label": latest_context_label,
        "latest_sentiment_value": latest_sentiment_value,
        "latest_gdelt_sentiment_mean": latest_gdelt_sentiment_value,
        "latest_gdelt_article_count": latest_gdelt_article_count,
        "latest_gdelt_regime_label": latest_gdelt_regime_label,
        "latest_gdelt_attention_label": latest_gdelt_attention_label,
        "latest_gdelt_coverage_quality_score": latest_gdelt_coverage_quality,
        "latest_gdelt_dominant_event_theme": latest_gdelt_dominant_event_theme,
        "latest_gdelt_risk_event_theme": latest_gdelt_risk_event_theme,
        "latest_gdelt_supportive_event_theme": latest_gdelt_supportive_event_theme,
        "latest_news_event_regime_label": latest_news_event_label,
        "latest_news_event_risk_score": latest_news_event_risk_score,
        "latest_effective_sentiment_source": latest_effective_sentiment_source,
        "latest_effective_sentiment_label": latest_effective_sentiment_label,
        "latest_onchain_regime_label": latest_onchain_label,
        "effective_sentiment_role": effective_sentiment_role,
        "gdelt_reliability_label": gdelt_reliability_label,
        "onchain_reliability_label": onchain_reliability_label,
        "combined_context_readiness": context_readiness,
        "selected_context_variant": selected_context_variant,
        "selected_context_feature_count": selected_context_feature_count,
        "context_selection_method": context_selection_method,
        "validation_macro_f1": validation_macro_f1,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": balanced_accuracy,
        "multimodal_summary": (
            f"{symbol} {strategy_name} currently suggests {resolved_signal}. "
            f"Context is {latest_context_label}, "
            f"effective sentiment is {latest_effective_sentiment_label} via {latest_effective_sentiment_source.replace('_', ' ')}, "
            f"{gdelt_summary_text}, "
            f"news event mode is {str(latest_news_event_label).replace('_', ' ')} "
            f"with risk score {float(latest_news_event_risk_score):.2f}, "
            f"on-chain regime is {latest_onchain_label}, and current context readiness is "
            f"{str(context_readiness).replace('_', ' ')}{context_selection_note}."
        ),
    }


def evaluate_market_multimodal_strategies(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Run the first multimodal strategies over the aligned market context datasets."""
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    context_reliability_map = load_context_reliability_map()
    summary_rows = []

    for symbol in SUPPORTED_STRATEGY_SYMBOLS:
        dataset_path = get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date)
        df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        df = build_target_labels(df, preferred_targets[symbol], timeframe=timeframe)
        context_reliability_row = context_reliability_map.get(symbol, {})

        confirmation_df = apply_multimodal_confirmation_gate(df)
        confirmation_accuracy, confirmation_macro_f1, confirmation_balanced = evaluate_rule_strategy(
            confirmation_df,
            symbol,
            "market_multimodal_confirmation_gate",
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_summary_row(
                symbol,
                "market_multimodal_confirmation_gate",
                confirmation_df,
                confirmation_accuracy,
                confirmation_macro_f1,
                confirmation_balanced,
                timeframe,
                start_date,
                end_date,
                context_reliability_row=context_reliability_row,
            )
        )

        veto_df = apply_multimodal_context_veto(df)
        veto_accuracy, veto_macro_f1, veto_balanced = evaluate_rule_strategy(
            veto_df,
            symbol,
            "market_multimodal_context_veto",
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_summary_row(
                symbol,
                "market_multimodal_context_veto",
                veto_df,
                veto_accuracy,
                veto_macro_f1,
                veto_balanced,
                timeframe,
                start_date,
                end_date,
                context_reliability_row=context_reliability_row,
            )
        )

        news_overlay_df = apply_news_event_risk_overlay(df)
        news_overlay_accuracy, news_overlay_macro_f1, news_overlay_balanced = evaluate_rule_strategy(
            news_overlay_df,
            symbol,
            "market_multimodal_news_event_veto",
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_summary_row(
                symbol,
                "market_multimodal_news_event_veto",
                news_overlay_df,
                news_overlay_accuracy,
                news_overlay_macro_f1,
                news_overlay_balanced,
                timeframe,
                start_date,
                end_date,
                context_reliability_row=context_reliability_row,
            )
        )

        logistic_result = evaluate_logistic_strategy(
            df,
            symbol,
            preferred_targets[symbol]["target_name"],
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_summary_row(
                symbol,
                logistic_result["model_name"],
                df,
                logistic_result["accuracy"],
                logistic_result["macro_f1"],
                logistic_result["balanced_accuracy"],
                timeframe,
                start_date,
                end_date,
                latest_signal=logistic_result["latest_signal"],
                latest_signal_confidence=logistic_result["latest_signal_confidence"],
                context_reliability_row=context_reliability_row,
            )
        )

        specialist_results = evaluate_context_specialist_logistics(
            df,
            symbol,
            timeframe,
            start_date,
            end_date,
        )
        for specialist_result in specialist_results:
            summary_rows.append(
                build_summary_row(
                    symbol,
                    specialist_result["model_name"],
                    df,
                    specialist_result["accuracy"],
                    specialist_result["macro_f1"],
                    specialist_result["balanced_accuracy"],
                    timeframe,
                    start_date,
                    end_date,
                    latest_signal=specialist_result["latest_signal"],
                    latest_signal_confidence=specialist_result["latest_signal_confidence"],
                    selected_context_variant=specialist_result["selected_context_variant"],
                    selected_context_feature_count=specialist_result["selected_context_feature_count"],
                    context_selection_method="direct_specialist",
                    context_reliability_row=context_reliability_row,
                )
            )

        validation_selected_result = evaluate_validation_selected_logistic(
            df,
            symbol,
            timeframe,
            start_date,
            end_date,
        )
        summary_rows.append(
            build_summary_row(
                symbol,
                validation_selected_result["model_name"],
                df,
                validation_selected_result["accuracy"],
                validation_selected_result["macro_f1"],
                validation_selected_result["balanced_accuracy"],
                timeframe,
                start_date,
                end_date,
                latest_signal=validation_selected_result["latest_signal"],
                latest_signal_confidence=validation_selected_result["latest_signal_confidence"],
                selected_context_variant=validation_selected_result["selected_context_variant"],
                selected_context_feature_count=validation_selected_result["selected_context_feature_count"],
                validation_macro_f1=validation_selected_result["validation_macro_f1"],
                validation_balanced_accuracy=validation_selected_result["validation_balanced_accuracy"],
                context_selection_method="validation_selected",
                context_reliability_row=context_reliability_row,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    ranked_df = summary_df.sort_values(
        ["symbol", "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "latest_signal_confidence"],
        ascending=[True, False, False, False, False],
    )
    best_df = ranked_df.groupby("symbol", as_index=False).head(1).reset_index(drop=True)
    output_path = get_market_multimodal_strategy_summary_path(timeframe, start_date, end_date)
    best_df.to_csv(output_path, index=False)

    print("market multimodal strategy summary generated")
    print(f"rows saved: {len(best_df)}")
    print(f"summary saved to: {output_path}")
    return best_df


if __name__ == "__main__":
    evaluate_market_multimodal_strategies()
