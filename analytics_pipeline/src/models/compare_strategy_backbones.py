"""Compare the market-only benchmark family against the market + futures backbone."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_market_futures_backtest_summary_path,
    get_market_futures_binary_backtest_summary_path,
    get_market_futures_binary_summary_path,
    get_market_futures_binary_walkforward_summary_path,
    get_market_futures_preferred_model_summary_path,
    get_market_futures_walkforward_summary_path,
    get_market_trend_forecast_summary_path,
    get_market_trend_regression_summary_path,
    get_market_trend_walkforward_summary_path,
    get_strategy_backbone_comparison_path,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_csv(path):
    return pd.read_csv(path)


def _pick_best_benchmark_classifier(summary_df):
    return (
        summary_df.sort_values(
            ["symbol", "macro_f1", "accuracy", "balanced_accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _pick_best_benchmark_regressor(regression_df):
    return (
        regression_df.sort_values(
            ["symbol", "directional_accuracy", "r2", "mae"],
            ascending=[True, False, False, True],
        )
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _pick_best_market_futures_model(preferred_df):
    filtered_df = preferred_df.dropna(subset=["test_macro_f1", "test_accuracy", "test_balanced_accuracy"]).copy()
    return (
        filtered_df.sort_values(
            ["symbol", "test_macro_f1", "test_accuracy", "test_balanced_accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _pick_best_binary_market_futures_model(binary_df):
    filtered_df = binary_df.dropna(subset=["macro_f1", "accuracy", "balanced_accuracy"]).copy()
    return (
        filtered_df.sort_values(
            ["symbol", "macro_f1", "balanced_accuracy", "accuracy", "long_f1"],
            ascending=[True, False, False, False, False],
        )
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _select_backbone_candidate(row):
    ternary_score = (
        _safe_float(row.get("ternary_backbone_best_macro_f1"))
        + 0.30 * _safe_float(row.get("ternary_backbone_best_balanced_accuracy"))
        + 0.20 * _safe_float(row.get("ternary_backbone_walkforward_avg_macro_f1"))
        + 0.10 * _safe_float(row.get("ternary_backbone_walkforward_avg_balanced_accuracy"))
    )
    binary_score = (
        _safe_float(row.get("binary_backbone_best_macro_f1"))
        + 0.30 * _safe_float(row.get("binary_backbone_best_balanced_accuracy"))
        + 0.20 * _safe_float(row.get("binary_backbone_walkforward_avg_macro_f1"))
        + 0.10 * _safe_float(row.get("binary_backbone_walkforward_avg_balanced_accuracy"))
        + 0.10 * _safe_float(row.get("binary_backbone_walkforward_avg_long_f1"))
    )
    ternary_available = bool(str(row.get("ternary_backbone_best_model_name", "")).strip())
    binary_available = bool(str(row.get("binary_backbone_best_model_name", "")).strip())

    if binary_available and (not ternary_available or binary_score > ternary_score + 0.02):
        return "binary_directional"
    return "three_class"


def _apply_selected_backbone_fields(row):
    selected_family = row["backbone_selected_family"]
    prefix = "binary_backbone_" if selected_family == "binary_directional" else "ternary_backbone_"

    row["backbone_best_family"] = selected_family
    row["backbone_best_model_name"] = row.get(f"{prefix}best_model_name")
    row["backbone_best_accuracy"] = row.get(f"{prefix}best_accuracy")
    row["backbone_best_macro_f1"] = row.get(f"{prefix}best_macro_f1")
    row["backbone_best_balanced_accuracy"] = row.get(f"{prefix}best_balanced_accuracy")
    row["backbone_best_latest_signal"] = row.get(f"{prefix}best_latest_signal")
    row["backbone_best_latest_confidence"] = row.get(f"{prefix}best_latest_confidence")
    row["backbone_walkforward_avg_accuracy"] = row.get(f"{prefix}walkforward_avg_accuracy")
    row["backbone_walkforward_avg_macro_f1"] = row.get(f"{prefix}walkforward_avg_macro_f1")
    row["backbone_walkforward_avg_balanced_accuracy"] = row.get(f"{prefix}walkforward_avg_balanced_accuracy")
    row["backbone_walkforward_summary"] = row.get(f"{prefix}walkforward_summary")
    row["backbone_futures_feature_completeness_score"] = row.get(f"{prefix}futures_feature_completeness_score")
    row["backbone_futures_completeness_label"] = row.get(f"{prefix}futures_completeness_label")
    row["backbone_futures_context_resilience_score"] = row.get(f"{prefix}futures_context_resilience_score")
    row["backbone_futures_context_resilience_label"] = row.get(f"{prefix}futures_context_resilience_label")
    row["backbone_futures_basis_reliance_score"] = row.get(f"{prefix}futures_basis_reliance_score")
    row["backbone_basis_feature_available"] = row.get(f"{prefix}basis_feature_available")

    if selected_family == "binary_directional":
        row["backbone_target_name"] = row.get("binary_backbone_target_name")
        row["backbone_long_f1"] = row.get("binary_backbone_long_f1")
        row["backbone_walkforward_avg_long_f1"] = row.get("binary_backbone_walkforward_avg_long_f1")
        row["backbone_deployed_model_name"] = row.get("binary_backbone_deployed_model_name")
        row["backbone_validation_policy_active"] = row.get("binary_backbone_validation_policy_active")
        row["backbone_deployment_active"] = row.get("binary_backbone_deployment_active")
        row["backbone_excess_return"] = row.get("binary_backbone_excess_return")
        row["backbone_backtest_summary"] = row.get("binary_backbone_backtest_summary")
        row["backbone_walkforward_avg_strategy_total_return"] = row.get("binary_backbone_walkforward_avg_strategy_total_return")
        row["backbone_walkforward_avg_buy_hold_return"] = row.get("binary_backbone_walkforward_avg_buy_hold_return")
        row["backbone_walkforward_avg_excess_return"] = row.get("binary_backbone_walkforward_avg_excess_return")
        row["backbone_walkforward_avg_sharpe"] = row.get("binary_backbone_walkforward_avg_sharpe")
        row["backbone_walkforward_avg_max_drawdown"] = row.get("binary_backbone_walkforward_avg_max_drawdown")
        row["backbone_walkforward_selected_policy"] = row.get("binary_backbone_walkforward_selected_policy")
        row["backbone_walkforward_deployment_active_rate"] = row.get("binary_backbone_walkforward_deployment_active_rate")
    else:
        row["backbone_target_name"] = row.get("ternary_backbone_target_name")
        row["backbone_deployed_model_name"] = row.get("ternary_backbone_deployed_model_name")
        row["backbone_validation_policy_active"] = row.get("ternary_backbone_validation_policy_active")
        row["backbone_deployment_active"] = row.get("ternary_backbone_deployment_active")
        row["backbone_excess_return"] = row.get("ternary_backbone_excess_return")
        row["backbone_backtest_summary"] = row.get("ternary_backbone_backtest_summary")
        row["backbone_walkforward_avg_long_f1"] = None
        row["backbone_walkforward_avg_strategy_total_return"] = row.get("ternary_backbone_walkforward_avg_strategy_total_return")
        row["backbone_walkforward_avg_buy_hold_return"] = row.get("ternary_backbone_walkforward_avg_buy_hold_return")
        row["backbone_walkforward_avg_excess_return"] = row.get("ternary_backbone_walkforward_avg_excess_return")
        row["backbone_walkforward_avg_sharpe"] = row.get("ternary_backbone_walkforward_avg_sharpe")
        row["backbone_walkforward_avg_max_drawdown"] = row.get("ternary_backbone_walkforward_avg_max_drawdown")
        row["backbone_walkforward_selected_policy"] = row.get("ternary_backbone_walkforward_selected_policy")
        row["backbone_walkforward_deployment_active_rate"] = 1.0 if str(row.get("ternary_backbone_deployment_active", "")).lower() == "true" else 0.0

    return row


def _coalesce_suffix_columns(df, base_name):
    """Coalesce duplicated merge columns like <name>_x / <name>_y back into <name>."""
    if base_name in df.columns:
        return df

    suffix_columns = [column for column in (f"{base_name}_x", f"{base_name}_y") if column in df.columns]
    if not suffix_columns:
        return df

    df[base_name] = df[suffix_columns].bfill(axis=1).iloc[:, 0]
    return df


def _family_recommendation(row):
    deployed_excess = _safe_float(row.get("backbone_walkforward_avg_excess_return"))
    deployed_macro_f1 = _safe_float(row.get("backbone_walkforward_avg_macro_f1"))
    benchmark_macro_f1 = _safe_float(row.get("benchmark_walkforward_avg_macro_f1"))
    heldout_uplift = _safe_float(row.get("classification_macro_f1_uplift"))
    deployment_active = str(row.get("backbone_deployment_active", "")).lower() == "true"
    policy_excess = _safe_float(row.get("backbone_excess_return"))
    selected_family = str(row.get("backbone_best_family", "three_class") or "three_class")

    if selected_family == "binary_directional":
        binary_long_f1 = _safe_float(row.get("backbone_walkforward_avg_long_f1"))
        deployment_active_rate = _safe_float(row.get("backbone_walkforward_deployment_active_rate"))
        if (
            deployment_active
            and policy_excess > 0
            and deployed_excess > 0
            and heldout_uplift > 0.03
            and deployed_macro_f1 >= benchmark_macro_f1
            and binary_long_f1 > 0.20
            and deployment_active_rate >= 0.50
        ):
            return "promote_market_futures_backbone"
        if (not deployment_active or deployment_active_rate < 0.30) and policy_excess <= 0:
            return "mixed_evidence_keep_both_visible"
        if deployment_active and (policy_excess <= 0 or deployed_excess <= 0):
            return "mixed_evidence_keep_both_visible"
        if benchmark_macro_f1 > deployed_macro_f1 + 0.08 and binary_long_f1 <= 0.15:
            return "keep_market_only_as_benchmark_lead"
        return "mixed_evidence_keep_both_visible"

    if deployment_active and deployed_excess > 0 and (deployed_macro_f1 >= benchmark_macro_f1 or heldout_uplift >= 0):
        return "promote_market_futures_backbone"
    if not deployment_active and deployed_excess <= 0:
        return "mixed_evidence_keep_both_visible"
    if benchmark_macro_f1 > deployed_macro_f1 + 0.08 and deployed_excess <= 0:
        return "keep_market_only_as_benchmark_lead"
    return "mixed_evidence_keep_both_visible"


def _family_reason(row):
    benchmark_name = row["benchmark_best_model_name"]
    backbone_name = row["backbone_best_model_name"]
    recommendation = row["recommended_family"]
    deployment_active = str(row.get("backbone_deployment_active", "")).lower() == "true"
    selected_family = str(row.get("backbone_best_family", "three_class") or "three_class")

    if selected_family == "binary_directional":
        if recommendation == "promote_market_futures_backbone":
            return (
                f"{row['symbol']} currently looks strongest under the binary directional market + futures family. "
                f"It is separating long vs flat more cleanly than {benchmark_name} on recent evaluation windows."
            )
        if recommendation == "keep_market_only_as_benchmark_lead":
            return (
                f"{row['symbol']} still shows cleaner evidence for the market-only benchmark {benchmark_name} "
                f"than for the binary directional market + futures candidate {backbone_name}."
            )
        return (
            f"{row['symbol']} has promising binary directional evidence, but it is not yet strong enough to replace "
            f"{benchmark_name} as the lead family."
        )

    if recommendation == "promote_market_futures_backbone":
        return (
            f"{row['symbol']} still supports the market + futures backbone. "
            f"The deployed backbone keeps policy-level value while remaining competitive with {benchmark_name}."
        )
    if recommendation == "keep_market_only_as_benchmark_lead":
        return (
            f"{row['symbol']} currently shows cleaner evidence for the market-only benchmark {benchmark_name} "
            f"than for the deployed market + futures path using {backbone_name}."
        )
    if not deployment_active:
        return (
            f"{row['symbol']} has improving backbone evidence, but the deployed policy layer is still inactive or weak. "
            f"{benchmark_name} and the market + futures path should both remain visible until deployment becomes credible."
        )
    return (
        f"{row['symbol']} has mixed evidence. {benchmark_name} and the deployed market + futures path "
        f"should both remain visible until more windows resolve the disagreement."
    )


def compare_strategy_backbones(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Build one comparison table between the benchmark family and the backbone family."""
    benchmark_summary = _load_csv(get_market_trend_forecast_summary_path(timeframe, start_date, end_date))
    benchmark_regression = _load_csv(get_market_trend_regression_summary_path(timeframe, start_date, end_date))
    benchmark_walkforward = _load_csv(get_market_trend_walkforward_summary_path(timeframe, start_date, end_date))
    futures_preferred = _load_csv(get_market_futures_preferred_model_summary_path(timeframe, start_date, end_date))
    futures_backtest = _load_csv(get_market_futures_backtest_summary_path(timeframe, start_date, end_date))
    futures_walkforward = _load_csv(get_market_futures_walkforward_summary_path(timeframe, start_date, end_date))
    binary_summary = _load_csv(get_market_futures_binary_summary_path(timeframe, start_date, end_date))
    binary_backtest = _load_csv(get_market_futures_binary_backtest_summary_path(timeframe, start_date, end_date))
    binary_walkforward = _load_csv(get_market_futures_binary_walkforward_summary_path(timeframe, start_date, end_date))

    best_benchmark = _pick_best_benchmark_classifier(benchmark_summary).rename(
        columns={
            "model_name": "benchmark_best_model_name",
            "accuracy": "benchmark_best_accuracy",
            "macro_f1": "benchmark_best_macro_f1",
            "balanced_accuracy": "benchmark_best_balanced_accuracy",
            "latest_prediction": "benchmark_latest_prediction",
        }
    )
    best_benchmark_regression = _pick_best_benchmark_regressor(benchmark_regression).rename(
        columns={
            "model_name": "benchmark_best_regression_model_name",
            "mae": "benchmark_best_regression_mae",
            "rmse": "benchmark_best_regression_rmse",
            "r2": "benchmark_best_regression_r2",
            "directional_accuracy": "benchmark_best_regression_directional_accuracy",
            "latest_predicted_posture": "benchmark_regression_latest_posture",
        }
    )
    benchmark_walkforward = benchmark_walkforward.rename(
        columns={
            "model_name": "benchmark_walkforward_model_name",
            "walkforward_avg_accuracy": "benchmark_walkforward_avg_accuracy",
            "walkforward_avg_macro_f1": "benchmark_walkforward_avg_macro_f1",
            "walkforward_avg_balanced_accuracy": "benchmark_walkforward_avg_balanced_accuracy",
            "walkforward_summary": "benchmark_walkforward_summary",
        }
    )
    best_futures_model = _pick_best_market_futures_model(futures_preferred).rename(
        columns={
            "model_name": "ternary_backbone_best_model_name",
            "test_accuracy": "ternary_backbone_best_accuracy",
            "test_macro_f1": "ternary_backbone_best_macro_f1",
            "test_balanced_accuracy": "ternary_backbone_best_balanced_accuracy",
            "latest_signal": "ternary_backbone_best_latest_signal",
            "latest_signal_confidence": "ternary_backbone_best_latest_confidence",
            "target_name": "ternary_backbone_target_name",
        }
    )
    best_binary_model = _pick_best_binary_market_futures_model(binary_summary).rename(
        columns={
            "model_name": "binary_backbone_best_model_name",
            "accuracy": "binary_backbone_best_accuracy",
            "macro_f1": "binary_backbone_best_macro_f1",
            "balanced_accuracy": "binary_backbone_best_balanced_accuracy",
            "latest_signal": "binary_backbone_best_latest_signal",
            "latest_signal_confidence": "binary_backbone_best_latest_confidence",
            "target_name": "binary_backbone_target_name",
            "long_f1": "binary_backbone_long_f1",
        }
    )
    futures_backtest = futures_backtest.rename(
        columns={
            "selected_backend_model": "ternary_backbone_deployed_model_name",
            "selected_target_name": "ternary_backbone_target_name",
            "policy_name": "ternary_backbone_policy_name",
            "probability_mode": "ternary_backbone_probability_mode",
            "strategy_total_return": "ternary_backbone_strategy_total_return",
            "buy_hold_total_return": "ternary_backbone_buy_hold_total_return",
            "excess_return": "ternary_backbone_excess_return",
            "sharpe_ratio": "ternary_backbone_sharpe_ratio",
            "max_drawdown": "ternary_backbone_max_drawdown",
            "latest_signal": "ternary_backbone_latest_signal",
            "latest_action": "ternary_backbone_latest_action",
            "latest_position": "ternary_backbone_latest_position",
            "latest_signal_confidence": "ternary_backbone_latest_signal_confidence",
            "validation_policy_active": "ternary_backbone_validation_policy_active",
            "deployment_active": "ternary_backbone_deployment_active",
            "prediction_concentration_flag": "ternary_backbone_prediction_concentration_flag",
            "dominant_predicted_label": "ternary_backbone_dominant_predicted_label",
            "dominant_predicted_share": "ternary_backbone_dominant_predicted_share",
            "futures_feature_completeness_score": "ternary_backbone_futures_feature_completeness_score",
            "futures_completeness_label": "ternary_backbone_futures_completeness_label",
            "futures_context_resilience_score": "ternary_backbone_futures_context_resilience_score",
            "futures_context_resilience_label": "ternary_backbone_futures_context_resilience_label",
            "futures_basis_reliance_score": "ternary_backbone_futures_basis_reliance_score",
            "basis_feature_available": "ternary_backbone_basis_feature_available",
            "backtest_summary": "ternary_backbone_backtest_summary",
        }
    )
    futures_walkforward = futures_walkforward.rename(
        columns={
            "walkforward_avg_accuracy": "ternary_backbone_walkforward_avg_accuracy",
            "walkforward_avg_macro_f1": "ternary_backbone_walkforward_avg_macro_f1",
            "walkforward_avg_balanced_accuracy": "ternary_backbone_walkforward_avg_balanced_accuracy",
            "walkforward_avg_strategy_total_return": "ternary_backbone_walkforward_avg_strategy_total_return",
            "walkforward_avg_buy_hold_return": "ternary_backbone_walkforward_avg_buy_hold_return",
            "walkforward_avg_excess_return": "ternary_backbone_walkforward_avg_excess_return",
            "walkforward_avg_sharpe": "ternary_backbone_walkforward_avg_sharpe",
            "walkforward_avg_max_drawdown": "ternary_backbone_walkforward_avg_max_drawdown",
            "walkforward_selected_policy": "ternary_backbone_walkforward_selected_policy",
            "walkforward_selected_probability_mode": "ternary_backbone_walkforward_selected_probability_mode",
            "futures_feature_completeness_score": "ternary_backbone_futures_feature_completeness_score",
            "futures_completeness_label": "ternary_backbone_futures_completeness_label",
            "futures_context_resilience_score": "ternary_backbone_futures_context_resilience_score",
            "futures_context_resilience_label": "ternary_backbone_futures_context_resilience_label",
            "futures_basis_reliance_score": "ternary_backbone_futures_basis_reliance_score",
            "basis_feature_available": "ternary_backbone_basis_feature_available",
            "walkforward_summary": "ternary_backbone_walkforward_summary",
        }
    )
    binary_walkforward = binary_walkforward.rename(
        columns={
            "model_name": "binary_backbone_walkforward_model_name",
            "walkforward_avg_accuracy": "binary_backbone_walkforward_avg_accuracy",
            "walkforward_avg_macro_f1": "binary_backbone_walkforward_avg_macro_f1",
            "walkforward_avg_balanced_accuracy": "binary_backbone_walkforward_avg_balanced_accuracy",
            "walkforward_avg_strategy_total_return": "binary_backbone_walkforward_avg_strategy_total_return",
            "walkforward_avg_buy_hold_return": "binary_backbone_walkforward_avg_buy_hold_return",
            "walkforward_avg_excess_return": "binary_backbone_walkforward_avg_excess_return",
            "walkforward_avg_sharpe": "binary_backbone_walkforward_avg_sharpe",
            "walkforward_avg_max_drawdown": "binary_backbone_walkforward_avg_max_drawdown",
            "walkforward_avg_long_f1": "binary_backbone_walkforward_avg_long_f1",
            "walkforward_selected_policy": "binary_backbone_walkforward_selected_policy",
            "walkforward_deployment_active_rate": "binary_backbone_walkforward_deployment_active_rate",
            "futures_feature_completeness_score": "binary_backbone_futures_feature_completeness_score",
            "futures_completeness_label": "binary_backbone_futures_completeness_label",
            "futures_context_resilience_score": "binary_backbone_futures_context_resilience_score",
            "futures_context_resilience_label": "binary_backbone_futures_context_resilience_label",
            "futures_basis_reliance_score": "binary_backbone_futures_basis_reliance_score",
            "basis_feature_available": "binary_backbone_basis_feature_available",
            "walkforward_summary": "binary_backbone_walkforward_summary",
        }
    )
    binary_backtest = binary_backtest.rename(
        columns={
            "selected_backend_model": "binary_backbone_deployed_model_name",
            "selected_target_name": "binary_backbone_target_name",
            "policy_name": "binary_backbone_policy_name",
            "probability_mode": "binary_backbone_probability_mode",
            "strategy_total_return": "binary_backbone_strategy_total_return",
            "buy_hold_total_return": "binary_backbone_buy_hold_total_return",
            "excess_return": "binary_backbone_excess_return",
            "sharpe_ratio": "binary_backbone_sharpe_ratio",
            "max_drawdown": "binary_backbone_max_drawdown",
            "latest_signal": "binary_backbone_latest_signal",
            "latest_action": "binary_backbone_latest_action",
            "latest_position": "binary_backbone_latest_position",
            "latest_signal_confidence": "binary_backbone_latest_signal_confidence",
            "validation_policy_active": "binary_backbone_validation_policy_active",
            "deployment_active": "binary_backbone_deployment_active",
            "futures_feature_completeness_score": "binary_backbone_futures_feature_completeness_score",
            "futures_completeness_label": "binary_backbone_futures_completeness_label",
            "futures_context_resilience_score": "binary_backbone_futures_context_resilience_score",
            "futures_context_resilience_label": "binary_backbone_futures_context_resilience_label",
            "futures_basis_reliance_score": "binary_backbone_futures_basis_reliance_score",
            "basis_feature_available": "binary_backbone_basis_feature_available",
            "backtest_summary": "binary_backbone_backtest_summary",
        }
    )

    comparison_df = (
        best_benchmark
        .merge(
            best_benchmark_regression[
                [
                    "symbol",
                    "benchmark_best_regression_model_name",
                    "benchmark_best_regression_mae",
                    "benchmark_best_regression_rmse",
                    "benchmark_best_regression_r2",
                    "benchmark_best_regression_directional_accuracy",
                    "benchmark_regression_latest_posture",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            benchmark_walkforward[
                [
                    "symbol",
                    "benchmark_walkforward_model_name",
                    "benchmark_walkforward_avg_accuracy",
                    "benchmark_walkforward_avg_macro_f1",
                    "benchmark_walkforward_avg_balanced_accuracy",
                    "benchmark_walkforward_summary",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            best_futures_model[
                [
                    "symbol",
                    "ternary_backbone_best_model_name",
                    "ternary_backbone_best_accuracy",
                    "ternary_backbone_best_macro_f1",
                    "ternary_backbone_best_balanced_accuracy",
                    "ternary_backbone_best_latest_signal",
                    "ternary_backbone_best_latest_confidence",
                    "ternary_backbone_target_name",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            best_binary_model[
                [
                    "symbol",
                    "binary_backbone_best_model_name",
                    "binary_backbone_best_accuracy",
                    "binary_backbone_best_macro_f1",
                    "binary_backbone_best_balanced_accuracy",
                    "binary_backbone_best_latest_signal",
                    "binary_backbone_best_latest_confidence",
                    "binary_backbone_target_name",
                    "binary_backbone_long_f1",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            futures_backtest[
                [
                    "symbol",
                    "ternary_backbone_deployed_model_name",
                    "ternary_backbone_target_name",
                    "ternary_backbone_policy_name",
                    "ternary_backbone_probability_mode",
                    "ternary_backbone_strategy_total_return",
                    "ternary_backbone_buy_hold_total_return",
                    "ternary_backbone_excess_return",
                    "ternary_backbone_sharpe_ratio",
                    "ternary_backbone_max_drawdown",
                    "ternary_backbone_latest_signal",
                    "ternary_backbone_latest_action",
                    "ternary_backbone_latest_position",
                    "ternary_backbone_latest_signal_confidence",
                    "ternary_backbone_validation_policy_active",
                    "ternary_backbone_deployment_active",
                    "ternary_backbone_prediction_concentration_flag",
                    "ternary_backbone_dominant_predicted_label",
                    "ternary_backbone_dominant_predicted_share",
                    "ternary_backbone_futures_feature_completeness_score",
                    "ternary_backbone_futures_completeness_label",
                    "ternary_backbone_futures_context_resilience_score",
                    "ternary_backbone_futures_context_resilience_label",
                    "ternary_backbone_futures_basis_reliance_score",
                    "ternary_backbone_basis_feature_available",
                    "ternary_backbone_backtest_summary",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            futures_walkforward[
                [
                    "symbol",
                    "ternary_backbone_walkforward_avg_accuracy",
                    "ternary_backbone_walkforward_avg_macro_f1",
                    "ternary_backbone_walkforward_avg_balanced_accuracy",
                    "ternary_backbone_walkforward_avg_strategy_total_return",
                    "ternary_backbone_walkforward_avg_buy_hold_return",
                    "ternary_backbone_walkforward_avg_excess_return",
                    "ternary_backbone_walkforward_avg_sharpe",
                    "ternary_backbone_walkforward_avg_max_drawdown",
                    "ternary_backbone_walkforward_selected_policy",
                    "ternary_backbone_walkforward_selected_probability_mode",
                    "ternary_backbone_futures_feature_completeness_score",
                    "ternary_backbone_futures_completeness_label",
                    "ternary_backbone_futures_context_resilience_score",
                    "ternary_backbone_futures_context_resilience_label",
                    "ternary_backbone_futures_basis_reliance_score",
                    "ternary_backbone_basis_feature_available",
                    "ternary_backbone_walkforward_summary",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            binary_backtest[
                [
                    "symbol",
                    "binary_backbone_deployed_model_name",
                    "binary_backbone_target_name",
                    "binary_backbone_policy_name",
                    "binary_backbone_probability_mode",
                    "binary_backbone_strategy_total_return",
                    "binary_backbone_buy_hold_total_return",
                    "binary_backbone_excess_return",
                    "binary_backbone_sharpe_ratio",
                    "binary_backbone_max_drawdown",
                    "binary_backbone_latest_signal",
                    "binary_backbone_latest_action",
                    "binary_backbone_latest_position",
                    "binary_backbone_latest_signal_confidence",
                    "binary_backbone_validation_policy_active",
                    "binary_backbone_deployment_active",
                    "binary_backbone_futures_feature_completeness_score",
                    "binary_backbone_futures_completeness_label",
                    "binary_backbone_futures_context_resilience_score",
                    "binary_backbone_futures_context_resilience_label",
                    "binary_backbone_futures_basis_reliance_score",
                    "binary_backbone_basis_feature_available",
                    "binary_backbone_backtest_summary",
                ]
            ],
            on="symbol",
            how="left",
        )
        .merge(
            binary_walkforward[
                [
                    "symbol",
                    "binary_backbone_walkforward_model_name",
                    "binary_backbone_walkforward_avg_accuracy",
                    "binary_backbone_walkforward_avg_macro_f1",
                    "binary_backbone_walkforward_avg_balanced_accuracy",
                    "binary_backbone_walkforward_avg_strategy_total_return",
                    "binary_backbone_walkforward_avg_buy_hold_return",
                    "binary_backbone_walkforward_avg_excess_return",
                    "binary_backbone_walkforward_avg_sharpe",
                    "binary_backbone_walkforward_avg_max_drawdown",
                    "binary_backbone_walkforward_avg_long_f1",
                    "binary_backbone_walkforward_selected_policy",
                    "binary_backbone_walkforward_deployment_active_rate",
                    "binary_backbone_futures_feature_completeness_score",
                    "binary_backbone_futures_completeness_label",
                    "binary_backbone_futures_context_resilience_score",
                    "binary_backbone_futures_context_resilience_label",
                    "binary_backbone_futures_basis_reliance_score",
                    "binary_backbone_basis_feature_available",
                    "binary_backbone_walkforward_summary",
                ]
            ],
            on="symbol",
            how="left",
        )
    )

    for base_name in [
        "ternary_backbone_futures_feature_completeness_score",
        "ternary_backbone_futures_completeness_label",
        "ternary_backbone_futures_context_resilience_score",
        "ternary_backbone_futures_context_resilience_label",
        "ternary_backbone_futures_basis_reliance_score",
        "ternary_backbone_basis_feature_available",
        "binary_backbone_futures_feature_completeness_score",
        "binary_backbone_futures_completeness_label",
        "binary_backbone_futures_context_resilience_score",
        "binary_backbone_futures_context_resilience_label",
        "binary_backbone_futures_basis_reliance_score",
        "binary_backbone_basis_feature_available",
    ]:
        comparison_df = _coalesce_suffix_columns(comparison_df, base_name)

    comparison_df["backbone_selected_family"] = comparison_df.apply(_select_backbone_candidate, axis=1)
    comparison_df = comparison_df.apply(_apply_selected_backbone_fields, axis=1)
    comparison_df = comparison_df.copy()

    comparison_df["timeframe"] = timeframe
    comparison_df["window_start"] = start_date
    comparison_df["window_end"] = end_date
    comparison_df["classification_macro_f1_uplift"] = (
        comparison_df["backbone_best_macro_f1"] - comparison_df["benchmark_best_macro_f1"]
    )
    comparison_df["classification_accuracy_uplift"] = (
        comparison_df["backbone_best_accuracy"] - comparison_df["benchmark_best_accuracy"]
    )
    comparison_df["walkforward_macro_f1_uplift"] = (
        comparison_df["backbone_walkforward_avg_macro_f1"] - comparison_df["benchmark_walkforward_avg_macro_f1"]
    )
    comparison_df["walkforward_accuracy_uplift"] = (
        comparison_df["backbone_walkforward_avg_accuracy"] - comparison_df["benchmark_walkforward_avg_accuracy"]
    )
    comparison_df["deployed_policy_excess_return"] = comparison_df["backbone_excess_return"]
    comparison_df["recommended_family"] = comparison_df.apply(_family_recommendation, axis=1)
    comparison_df["comparison_summary"] = comparison_df.apply(_family_reason, axis=1)

    output_path = get_strategy_backbone_comparison_path(timeframe, start_date, end_date)
    comparison_df.to_csv(output_path, index=False)

    print("strategy backbone comparison generated")
    print(f"rows saved: {len(comparison_df)}")
    print(f"summary saved to: {output_path}")
    return comparison_df


if __name__ == "__main__":
    compare_strategy_backbones()
