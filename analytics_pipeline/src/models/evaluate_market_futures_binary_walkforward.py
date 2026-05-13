"""Run deployment-aware walk-forward evaluation for binary market + futures strategies."""

from collections import Counter

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_market_futures_binary_walkforward_detail_path,
    get_market_futures_binary_walkforward_summary_path,
    get_market_futures_dataset_path,
)
from src.models.evaluate_market_futures_binary_backtests import (
    BINARY_POLICY_VARIANTS,
    apply_binary_policy,
    build_backtest_metrics,
    build_binary_backend_candidate_frames,
    build_binary_decision_frame,
    choose_binary_thresholds,
    should_replace_selected_binary_candidate,
    split_train_calibration_validation,
)
from src.models.evaluate_market_futures_backtests import is_active_policy
from src.models.evaluate_market_futures_binary_directional import _build_binary_labels, _safe_balanced_accuracy
from src.models.evaluate_market_futures_strategies import build_combined_feature_matrix
from src.models.market_futures_targets import get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE

WALKFORWARD_PROFILES = {
    "1h": {"min_train_rows": 120, "test_window_rows": 24, "step_rows": 24},
    "4h": {"min_train_rows": 72, "test_window_rows": 12, "step_rows": 12},
    "1d": {"min_train_rows": 90, "test_window_rows": 10, "step_rows": 10},
}


def get_binary_walkforward_profile(timeframe):
    """Return a cadence profile suited to the requested timeframe."""
    return WALKFORWARD_PROFILES.get(timeframe, WALKFORWARD_PROFILES["4h"])


def make_binary_walkforward_folds(total_rows, timeframe):
    """Create expanding-window folds over one binary directional dataset."""
    profile = get_binary_walkforward_profile(timeframe)
    folds = []
    fold_number = 1
    train_end = profile["min_train_rows"]

    while train_end + profile["test_window_rows"] <= total_rows:
        folds.append(
            {
                "fold_number": fold_number,
                "train_end": train_end,
                "test_end": train_end + profile["test_window_rows"],
            }
        )
        train_end += profile["step_rows"]
        fold_number += 1

    return folds


def summarize_binary_walkforward_detail(detail_df):
    """Aggregate fold-level binary deployment results into one row per symbol."""
    summary_rows = []
    for symbol, symbol_df in detail_df.groupby("symbol"):
        selected_policy = Counter(symbol_df["selected_policy_name"]).most_common(1)[0][0]
        selected_model = Counter(symbol_df["selected_backend_model"]).most_common(1)[0][0]
        robust_average_sharpe = float(symbol_df["selected_sharpe_ratio"].clip(-10, 10).mean())
        summary_rows.append(
            {
                "symbol": symbol,
                "timeframe": symbol_df["timeframe"].iloc[0],
                "model_name": selected_model,
                "window_start": symbol_df["window_start"].iloc[0],
                "window_end": symbol_df["window_end"].iloc[0],
                "walkforward_fold_count": int(len(symbol_df)),
                "walkforward_avg_accuracy": float(symbol_df["fold_accuracy"].mean()),
                "walkforward_avg_macro_f1": float(symbol_df["fold_macro_f1"].mean()),
                "walkforward_avg_balanced_accuracy": float(symbol_df["fold_balanced_accuracy"].mean()),
                "walkforward_avg_long_precision": float(symbol_df["fold_long_precision"].mean()),
                "walkforward_avg_long_recall": float(symbol_df["fold_long_recall"].mean()),
                "walkforward_avg_long_f1": float(symbol_df["fold_long_f1"].mean()),
                "walkforward_avg_strategy_total_return": float(symbol_df["selected_strategy_total_return"].mean()),
                "walkforward_avg_buy_hold_return": float(symbol_df["selected_buy_hold_total_return"].mean()),
                "walkforward_avg_excess_return": float(symbol_df["selected_excess_return"].mean()),
                "walkforward_avg_sharpe": robust_average_sharpe,
                "walkforward_avg_max_drawdown": float(symbol_df["selected_max_drawdown"].mean()),
                "walkforward_selected_policy": selected_policy,
                "walkforward_deployment_active_rate": float(symbol_df["selected_deployment_active"].mean()),
                "walkforward_validation_active_rate": float(symbol_df["selected_validation_policy_active"].mean()),
                "futures_feature_completeness_score": symbol_df["futures_feature_completeness_score"].iloc[-1],
                "futures_completeness_label": symbol_df["futures_completeness_label"].iloc[-1],
                "futures_context_resilience_score": symbol_df["futures_context_resilience_score"].iloc[-1],
                "futures_context_resilience_label": symbol_df["futures_context_resilience_label"].iloc[-1],
                "futures_basis_reliance_score": symbol_df["futures_basis_reliance_score"].iloc[-1],
                "basis_feature_available": bool(symbol_df["basis_feature_available"].iloc[-1]),
                "walkforward_summary": (
                    f"{symbol} binary market+futures walk-forward used {len(symbol_df)} folds with "
                    f"accuracy {symbol_df['fold_accuracy'].mean() * 100:.1f}%, macro-F1 "
                    f"{symbol_df['fold_macro_f1'].mean() * 100:.1f}%, long-class F1 "
                    f"{symbol_df['fold_long_f1'].mean() * 100:.1f}%, and average policy excess return "
                    f"{symbol_df['selected_excess_return'].mean() * 100:.1f}%."
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def build_empty_binary_walkforward_summary(preferred_targets, timeframe, start_date, end_date):
    """Record when the window is too short for the deployment walk-forward test."""
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
                "model_name": "not_selected_no_reliable_folds",
                "window_start": start_date,
                "window_end": end_date,
                "walkforward_fold_count": 0,
                "walkforward_avg_accuracy": 0.0,
                "walkforward_avg_macro_f1": 0.0,
                "walkforward_avg_balanced_accuracy": 0.0,
                "walkforward_avg_long_precision": 0.0,
                "walkforward_avg_long_recall": 0.0,
                "walkforward_avg_long_f1": 0.0,
                "walkforward_avg_strategy_total_return": 0.0,
                "walkforward_avg_buy_hold_return": 0.0,
                "walkforward_avg_excess_return": 0.0,
                "walkforward_avg_sharpe": 0.0,
                "walkforward_avg_max_drawdown": 0.0,
                "walkforward_selected_policy": "not_selected_no_reliable_folds",
                "walkforward_deployment_active_rate": 0.0,
                "walkforward_validation_active_rate": 0.0,
                "futures_feature_completeness_score": latest_row.get("futures_feature_completeness_score"),
                "futures_completeness_label": latest_row.get("futures_completeness_label"),
                "futures_context_resilience_score": latest_row.get("futures_context_resilience_score"),
                "futures_context_resilience_label": latest_row.get("futures_context_resilience_label"),
                "futures_basis_reliance_score": latest_row.get("futures_basis_reliance_score"),
                "basis_feature_available": latest_row.get("basis_feature_available"),
                "walkforward_summary": (
                    f"{symbol} binary market+futures deployment walk-forward was skipped on {timeframe} "
                    "because the available futures-aligned window is too short for reliable rolling folds."
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_binary_walkforward_for_symbol(symbol, target_config, timeframe, start_date, end_date):
    """Run deployment-aware anchored walk-forward for one binary strategy family."""
    dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
    df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
    df = df[df["futures_data_available"] == True].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df = _build_binary_labels(df, target_config, timeframe)
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
    folds = make_binary_walkforward_folds(len(df), timeframe)
    detail_rows = []

    for fold in folds:
        train_df = df.iloc[:fold["train_end"]].reset_index(drop=True)
        test_df = df.iloc[fold["train_end"]:fold["test_end"]].reset_index(drop=True)
        fit_df, calibration_df, validation_df, _ = split_train_calibration_validation(train_df)
        fit_end = len(fit_df)
        validation_end = fit_end + len(calibration_df) + len(validation_df)
        train_X = X.iloc[:fold["train_end"]].reset_index(drop=True)

        if min(len(fit_df), len(calibration_df), len(validation_df), len(test_df)) == 0:
            continue

        best_validation_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
        selected_validation_metrics = None
        selected_deployment_metrics = None
        best_fold_row = None

        for candidate in build_binary_backend_candidate_frames(train_df, train_X, fit_end, validation_end):
            if min(len(candidate["fit_X"]), len(candidate["validation_X"]), len(candidate["test_X"])) == 0:
                continue

            try:
                model = candidate["trainer"](candidate["fit_X"], candidate["fit_y"])
            except ValueError:
                continue
            validation_decisions = build_binary_decision_frame(candidate["validation_df"], model, candidate["validation_X"])
            test_decisions = build_binary_decision_frame(candidate["test_df"], model, candidate["test_X"])

            fold_accuracy = accuracy_score(test_decisions["binary_label"], test_decisions["predicted_label"])
            fold_macro_f1 = f1_score(
                test_decisions["binary_label"],
                test_decisions["predicted_label"],
                average="macro",
                zero_division=0,
            )
            fold_balanced_accuracy = _safe_balanced_accuracy(
                test_decisions["binary_label"],
                test_decisions["predicted_label"],
            )
            fold_long_precision = precision_score(
                test_decisions["binary_label"],
                test_decisions["predicted_label"],
                pos_label="long",
                zero_division=0,
            )
            fold_long_recall = recall_score(
                test_decisions["binary_label"],
                test_decisions["predicted_label"],
                pos_label="long",
                zero_division=0,
            )
            fold_long_f1 = f1_score(
                test_decisions["binary_label"],
                test_decisions["predicted_label"],
                pos_label="long",
                zero_division=0,
            )

            for policy_name in BINARY_POLICY_VARIANTS:
                thresholds = choose_binary_thresholds(validation_decisions, timeframe, policy_name, symbol)
                test_backtest = apply_binary_policy(
                    test_decisions,
                    policy_name,
                    thresholds["enter_threshold"],
                    thresholds["exit_threshold"],
                    timeframe=timeframe,
                )
                test_metrics = build_backtest_metrics(test_backtest, timeframe)

                if should_replace_selected_binary_candidate(
                    best_validation_score,
                    thresholds["validation_score"],
                    selected_validation_metrics,
                    thresholds["metrics"],
                    selected_deployment_metrics,
                    test_metrics,
                    timeframe,
                ):
                    best_validation_score = thresholds["validation_score"]
                    selected_validation_metrics = thresholds["metrics"]
                    selected_deployment_metrics = test_metrics
                    best_fold_row = {
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
                        "selected_backend_model": candidate["backend_model_name"],
                        "selected_policy_name": policy_name,
                        "selected_enter_threshold": thresholds["enter_threshold"],
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
                        "fold_accuracy": float(fold_accuracy),
                        "fold_macro_f1": float(fold_macro_f1),
                        "fold_balanced_accuracy": float(fold_balanced_accuracy),
                        "fold_long_precision": float(fold_long_precision),
                        "fold_long_recall": float(fold_long_recall),
                        "fold_long_f1": float(fold_long_f1),
                    }

        if best_fold_row is not None:
            detail_rows.append(best_fold_row)

    return detail_rows


def evaluate_market_futures_binary_walkforward(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Run deployment-aware walk-forward evaluation for binary market + futures strategies."""
    detail_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    for symbol, target_config in preferred_targets.items():
        detail_rows.extend(
            evaluate_binary_walkforward_for_symbol(
                symbol,
                target_config,
                timeframe,
                start_date,
                end_date,
            )
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(get_market_futures_binary_walkforward_detail_path(timeframe, start_date, end_date), index=False)

    if detail_df.empty:
        summary_df = build_empty_binary_walkforward_summary(preferred_targets, timeframe, start_date, end_date)
    else:
        summary_df = summarize_binary_walkforward_detail(detail_df)
    summary_df.to_csv(get_market_futures_binary_walkforward_summary_path(timeframe, start_date, end_date), index=False)

    print("market + futures binary walk-forward detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {get_market_futures_binary_walkforward_detail_path(timeframe, start_date, end_date)}")
    print("market + futures binary walk-forward summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {get_market_futures_binary_walkforward_summary_path(timeframe, start_date, end_date)}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_binary_walkforward()
