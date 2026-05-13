"""Run rolling walk-forward evaluation for the preferred market + futures backend."""

from collections import Counter

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_market_futures_dataset_path,
    get_market_futures_walkforward_detail_path,
    get_market_futures_walkforward_summary_path,
)
from src.models.evaluate_market_futures_backtests import (
    CALIBRATION_RATIO,
    POLICY_VARIANTS,
    VALIDATION_RATIO,
    apply_prescriptive_policy,
    build_backtest_metrics,
    build_decision_frame,
    build_policy_score,
    is_active_policy,
    choose_thresholds,
    fit_temperature_on_calibration,
    should_replace_selected_policy,
    uses_calibrated_probabilities,
)
from src.models.evaluate_market_futures_strategies import (
    build_combined_feature_matrix,
    train_scaled_logistic_baseline,
)
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
MIN_TRAIN_ROWS = 72
TEST_WINDOW_ROWS = 12
STEP_ROWS = 12
WALKFORWARD_PROFILES = {
    "1h": {"min_train_rows": 120, "test_window_rows": 24, "step_rows": 24},
    "4h": {"min_train_rows": 72, "test_window_rows": 12, "step_rows": 12},
    "1d": {"min_train_rows": 90, "test_window_rows": 10, "step_rows": 10},
}


def get_walkforward_profile(timeframe):
    """Return one expanding-window fold profile per timeframe."""
    return WALKFORWARD_PROFILES.get(timeframe, WALKFORWARD_PROFILES["4h"])


def make_walkforward_folds(total_rows, timeframe=TIMEFRAME, min_train_rows=MIN_TRAIN_ROWS,
                           test_window_rows=TEST_WINDOW_ROWS, step_rows=STEP_ROWS):
    """Create expanding-window folds over one chronological dataset."""
    profile = get_walkforward_profile(timeframe)
    min_train_rows = profile["min_train_rows"]
    test_window_rows = profile["test_window_rows"]
    step_rows = profile["step_rows"]
    folds = []
    fold_number = 1
    train_end = min_train_rows

    while train_end + test_window_rows <= total_rows:
        folds.append(
            {
                "fold_number": fold_number,
                "train_end": train_end,
                "test_end": train_end + test_window_rows,
            }
        )
        train_end += step_rows
        fold_number += 1

    return folds


def split_train_window(train_df):
    """Split one training window into fit, calibration, and validation segments."""
    train_length = len(train_df)
    calibration_start = int(train_length * (1.0 - CALIBRATION_RATIO - VALIDATION_RATIO))
    validation_start = int(train_length * (1.0 - VALIDATION_RATIO))

    fit_df = train_df.iloc[:calibration_start].reset_index(drop=True)
    calibration_df = train_df.iloc[calibration_start:validation_start].reset_index(drop=True)
    validation_df = train_df.iloc[validation_start:].reset_index(drop=True)
    return fit_df, calibration_df, validation_df


def compute_safe_balanced_accuracy(y_true, y_pred):
    """Compute balanced accuracy without warnings when a fold is missing classes."""
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)).tolist())
    recalls = []
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)

    for label in labels:
        true_mask = y_true_series == label
        support = int(true_mask.sum())
        if support == 0:
            continue
        recalls.append(float((y_pred_series[true_mask] == label).mean()))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def summarize_walkforward_detail(detail_df):
    """Aggregate fold-level results into one summary row per symbol."""
    if detail_df.empty or "symbol" not in detail_df.columns:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "window_start",
                "window_end",
                "walkforward_fold_count",
                "walkforward_avg_accuracy",
                "walkforward_avg_macro_f1",
                "walkforward_avg_balanced_accuracy",
                "walkforward_avg_strategy_total_return",
                "walkforward_avg_buy_hold_return",
                "walkforward_avg_excess_return",
                "walkforward_avg_sharpe",
                "walkforward_avg_max_drawdown",
                "walkforward_deployment_activity_rate",
                "walkforward_selected_policy",
                "walkforward_selected_probability_mode",
                "futures_feature_completeness_score",
                "futures_completeness_label",
                "futures_context_resilience_score",
                "futures_context_resilience_label",
                "futures_basis_reliance_score",
                "basis_feature_available",
                "walkforward_summary",
            ]
        )

    summary_rows = []
    for symbol, symbol_df in detail_df.groupby("symbol"):
        selected_policy = Counter(symbol_df["selected_policy_name"]).most_common(1)[0][0]
        selected_probability_mode = Counter(symbol_df["selected_probability_mode"]).most_common(1)[0][0]
        robust_average_sharpe = float(symbol_df["selected_sharpe_ratio"].clip(-10, 10).mean())
        deployment_activity_rate = float(symbol_df["selected_deployment_active"].astype(float).mean())
        futures_resilience_label = str(symbol_df["futures_context_resilience_label"].iloc[0] or "unavailable")
        futures_completeness_label = str(symbol_df["futures_completeness_label"].iloc[0] or "unavailable")
        basis_feature_available = bool(symbol_df["basis_feature_available"].iloc[0])
        summary_rows.append(
            {
                "symbol": symbol,
                "timeframe": symbol_df["timeframe"].iloc[0],
                "window_start": symbol_df["window_start"].iloc[0],
                "window_end": symbol_df["window_end"].iloc[0],
                "walkforward_fold_count": int(len(symbol_df)),
                "walkforward_avg_accuracy": float(symbol_df["fold_accuracy"].mean()),
                "walkforward_avg_macro_f1": float(symbol_df["fold_macro_f1"].mean()),
                "walkforward_avg_balanced_accuracy": float(symbol_df["fold_balanced_accuracy"].mean()),
                "walkforward_avg_strategy_total_return": float(symbol_df["selected_strategy_total_return"].mean()),
                "walkforward_avg_buy_hold_return": float(symbol_df["selected_buy_hold_total_return"].mean()),
                "walkforward_avg_excess_return": float(symbol_df["selected_excess_return"].mean()),
                "walkforward_avg_sharpe": robust_average_sharpe,
                "walkforward_avg_max_drawdown": float(symbol_df["selected_max_drawdown"].mean()),
                "walkforward_deployment_activity_rate": deployment_activity_rate,
                "walkforward_selected_policy": selected_policy,
                "walkforward_selected_probability_mode": selected_probability_mode,
                "futures_feature_completeness_score": symbol_df["futures_feature_completeness_score"].iloc[0],
                "futures_completeness_label": futures_completeness_label,
                "futures_context_resilience_score": symbol_df["futures_context_resilience_score"].iloc[0],
                "futures_context_resilience_label": futures_resilience_label,
                "futures_basis_reliance_score": symbol_df["futures_basis_reliance_score"].iloc[0],
                "basis_feature_available": basis_feature_available,
                "walkforward_summary": (
                    f"{symbol} walk-forward evaluation ran across {len(symbol_df)} folds. "
                    f"Average accuracy is {symbol_df['fold_accuracy'].mean() * 100:.1f}%, "
                    f"macro-F1 is {symbol_df['fold_macro_f1'].mean() * 100:.1f}%, and average policy excess return "
                    f"is {symbol_df['selected_excess_return'].mean() * 100:.1f}% with robust average Sharpe "
                    f"{robust_average_sharpe:.2f}. Deployment was active in {deployment_activity_rate * 100:.1f}% "
                    f"of folds while futures support stayed {futures_resilience_label.replace('_', ' ')} with "
                    f"{futures_completeness_label.replace('_', ' ')} coverage"
                    f"{'' if basis_feature_available else ' and basis currently missing'}."
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def build_empty_walkforward_summary(preferred_targets, timeframe, start_date, end_date):
    """Record symbols that have strategy outputs but not enough rows for rolling folds."""
    rows = []
    for symbol in preferred_targets:
        dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
        df = pd.read_csv(dataset_path)
        df = df[df["futures_data_available"] == True].copy()
        latest_row = df.iloc[-1] if not df.empty else {}
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "window_start": start_date,
                "window_end": end_date,
                "walkforward_fold_count": 0,
                "walkforward_avg_accuracy": 0.0,
                "walkforward_avg_macro_f1": 0.0,
                "walkforward_avg_balanced_accuracy": 0.0,
                "walkforward_avg_strategy_total_return": 0.0,
                "walkforward_avg_buy_hold_return": 0.0,
                "walkforward_avg_excess_return": 0.0,
                "walkforward_avg_sharpe": 0.0,
                "walkforward_avg_max_drawdown": 0.0,
                "walkforward_deployment_activity_rate": 0.0,
                "walkforward_selected_policy": "not_selected_no_reliable_folds",
                "walkforward_selected_probability_mode": "not_selected_no_reliable_folds",
                "futures_feature_completeness_score": latest_row.get("futures_feature_completeness_score"),
                "futures_completeness_label": latest_row.get("futures_completeness_label"),
                "futures_context_resilience_score": latest_row.get("futures_context_resilience_score"),
                "futures_context_resilience_label": latest_row.get("futures_context_resilience_label"),
                "futures_basis_reliance_score": latest_row.get("futures_basis_reliance_score"),
                "basis_feature_available": latest_row.get("basis_feature_available"),
                "walkforward_summary": (
                    f"{symbol} market+futures walk-forward was skipped on {timeframe} because the available "
                    "futures-aligned window is too short for reliable rolling folds."
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_walkforward_for_symbol(symbol, target_config, timeframe, start_date, end_date):
    """Run anchored walk-forward folds for one symbol's preferred target."""
    dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
    df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
    df = df[df["futures_data_available"] == True].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df = build_target_labels(df, target_config, timeframe=timeframe)
    futures_context = {}
    if not df.empty:
        latest_source_row = df.iloc[-1]
        futures_context = {
            "futures_feature_completeness_score": latest_source_row.get("futures_feature_completeness_score"),
            "futures_completeness_label": latest_source_row.get("futures_completeness_label"),
            "futures_context_resilience_score": latest_source_row.get("futures_context_resilience_score"),
            "futures_context_resilience_label": latest_source_row.get("futures_context_resilience_label"),
            "futures_basis_reliance_score": latest_source_row.get("futures_basis_reliance_score"),
            "basis_feature_available": latest_source_row.get("basis_feature_available"),
        }

    X, _ = build_combined_feature_matrix(df)
    folds = make_walkforward_folds(len(df), timeframe=timeframe)
    detail_rows = []

    for fold in folds:
        train_df = df.iloc[:fold["train_end"]].reset_index(drop=True)
        test_df = df.iloc[fold["train_end"]:fold["test_end"]].reset_index(drop=True)
        fit_df, calibration_df, validation_df = split_train_window(train_df)

        fit_end = len(fit_df)
        calibration_end = fit_end + len(calibration_df)
        train_X = X.iloc[:fold["train_end"]].reset_index(drop=True)
        fit_X = train_X.iloc[:fit_end]
        calibration_X = train_X.iloc[fit_end:calibration_end]
        validation_X = train_X.iloc[calibration_end:]
        test_X = X.iloc[fold["train_end"]:fold["test_end"]].reset_index(drop=True)

        if min(len(fit_df), len(calibration_df), len(validation_df), len(test_df)) == 0:
            continue

        base_model = train_scaled_logistic_baseline(fit_X, fit_df["label"])
        calibration_temperature = fit_temperature_on_calibration(
            base_model,
            calibration_X,
            calibration_df["label"],
        )

        test_decisions_raw = build_decision_frame(test_df, base_model, test_X)
        fold_accuracy = accuracy_score(test_decisions_raw["label"], test_decisions_raw["predicted_label"])
        fold_macro_f1 = f1_score(
            test_decisions_raw["label"],
            test_decisions_raw["predicted_label"],
            average="macro",
            zero_division=0,
        )
        fold_balanced_accuracy = compute_safe_balanced_accuracy(
            test_decisions_raw["label"],
            test_decisions_raw["predicted_label"],
        )

        validation_decisions_raw = build_decision_frame(validation_df, base_model, validation_X)
        validation_decisions_calibrated = build_decision_frame(
            validation_df,
            base_model,
            validation_X,
            temperature=calibration_temperature,
        )
        test_decisions_calibrated = build_decision_frame(
            test_df,
            base_model,
            test_X,
            temperature=calibration_temperature,
        )

        best_policy_row = None
        best_policy_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
        selected_validation_metrics = None

        for policy_name in POLICY_VARIANTS:
            validation_decisions = (
                validation_decisions_calibrated
                if uses_calibrated_probabilities(policy_name)
                else validation_decisions_raw
            )
            test_decisions = (
                test_decisions_calibrated
                if uses_calibrated_probabilities(policy_name)
                else test_decisions_raw
            )
            thresholds = choose_thresholds(
                validation_decisions,
                timeframe,
                policy_name,
                symbol,
                target_config["target_name"],
            )
            test_backtest = apply_prescriptive_policy(
                test_decisions,
                policy_name,
                thresholds["buy_threshold"],
                thresholds["exit_threshold"],
                timeframe=timeframe,
            )
            test_metrics = build_backtest_metrics(test_backtest, timeframe)
            score = build_policy_score(thresholds["metrics"], timeframe=timeframe)
            if should_replace_selected_policy(
                best_policy_score,
                score,
                selected_validation_metrics,
                thresholds["metrics"],
                timeframe=timeframe,
            ):
                best_policy_score = score
                selected_validation_metrics = thresholds["metrics"]
                best_policy_row = {
                    "selected_policy_name": policy_name,
                    "selected_probability_mode": "temperature_scaled" if uses_calibrated_probabilities(policy_name) else "raw",
                    "calibration_temperature": calibration_temperature,
                    "selected_buy_threshold": thresholds["buy_threshold"],
                    "selected_exit_threshold": thresholds["exit_threshold"],
                    "selected_strategy_total_return": test_metrics["strategy_total_return"],
                    "selected_buy_hold_total_return": test_metrics["buy_hold_total_return"],
                    "selected_excess_return": test_metrics["excess_return"],
                    "selected_sharpe_ratio": test_metrics["sharpe_ratio"],
                    "selected_max_drawdown": test_metrics["max_drawdown"],
                    "selected_trade_count": test_metrics["trade_count"],
                    "selected_exposure_ratio": test_metrics["exposure_ratio"],
                    "selected_hit_rate": test_metrics["hit_rate"],
                    "selected_validation_trade_count": thresholds["metrics"]["trade_count"],
                    "selected_validation_exposure_ratio": thresholds["metrics"]["exposure_ratio"],
                    "selected_validation_policy_active": is_active_policy(thresholds["metrics"], timeframe=timeframe),
                    "selected_deployment_active": is_active_policy(test_metrics, timeframe=timeframe),
                    "selected_latest_action": test_backtest.iloc[-1]["action"],
                    "selected_latest_position": float(test_backtest.iloc[-1]["position"]),
                    **futures_context,
                }

        detail_rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "window_start": start_date,
                "window_end": end_date,
                "target_name": target_config["target_name"],
                "target_horizon_hours": target_config.get("effective_horizon_hours"),
                "target_exact_horizon_match": target_config.get("exact_horizon_match"),
                "target_resolution_note": target_config.get("horizon_resolution_note"),
                "fold_number": fold["fold_number"],
                "train_rows": int(fold["train_end"]),
                "test_rows": int(fold["test_end"] - fold["train_end"]),
                "train_end_time": train_df["open_time"].iloc[-1],
                "test_start_time": test_df["open_time"].iloc[0],
                "test_end_time": test_df["open_time"].iloc[-1],
                "fold_accuracy": float(fold_accuracy),
                "fold_macro_f1": float(fold_macro_f1),
                "fold_balanced_accuracy": float(fold_balanced_accuracy),
                **best_policy_row,
            }
        )

    return detail_rows


def evaluate_market_futures_walkforward(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Run walk-forward evaluation for the preferred targets."""
    detail_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    for symbol, target_config in preferred_targets.items():
        detail_rows.extend(
            evaluate_walkforward_for_symbol(
                symbol,
                target_config,
                timeframe,
                start_date,
                end_date,
            )
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(get_market_futures_walkforward_detail_path(timeframe, start_date, end_date), index=False)

    if detail_df.empty:
        summary_df = build_empty_walkforward_summary(preferred_targets, timeframe, start_date, end_date)
    else:
        summary_df = summarize_walkforward_detail(detail_df)
    summary_df.to_csv(get_market_futures_walkforward_summary_path(timeframe, start_date, end_date), index=False)

    print("market + futures walk-forward detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {get_market_futures_walkforward_detail_path(timeframe, start_date, end_date)}")
    print("market + futures walk-forward summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {get_market_futures_walkforward_summary_path(timeframe, start_date, end_date)}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_walkforward()
