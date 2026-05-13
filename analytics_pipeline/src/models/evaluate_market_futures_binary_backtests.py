"""Evaluate binary directional deployment policies for market + futures models."""

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_market_futures_binary_backtest_curve_path,
    get_market_futures_binary_backtest_summary_path,
    get_market_futures_binary_policy_variant_summary_path,
    get_market_futures_dataset_path,
)
from src.models.evaluate_market_futures_backtests import (
    ACTIVE_POLICY_SCORE_TOLERANCE,
    CALIBRATION_RATIO,
    LAG_WINDOW,
    MIN_ACTIVE_EXPOSURE,
    MIN_ACTIVE_TRADE_COUNT,
    PERIODS_PER_YEAR,
    TRANSACTION_COST_RATE,
    VALIDATION_RATIO,
    annualize_return,
    annualize_volatility,
    calculate_max_drawdown,
    get_timeframe_deployment_profile,
    is_active_policy,
    should_replace_selected_policy,
    train_lagged_random_forest_baseline,
)
from src.models.evaluate_market_futures_binary_directional import _build_binary_labels
from src.models.evaluate_market_futures_strategies import build_combined_feature_matrix, train_scaled_logistic_baseline
from src.models.market_futures_targets import get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE

BINARY_POLICY_VARIANTS = [
    "binary_confidence_gated_long_flat",
    "binary_regime_adaptive_long_flat",
    "binary_conviction_weighted_long_only",
]

BINARY_THRESHOLD_SPACES = {
    "BTCUSDT": {"enter_thresholds": [0.45, 0.50, 0.55, 0.60], "exit_thresholds": [0.45, 0.50, 0.55]},
    "ETHUSDT": {"enter_thresholds": [0.50, 0.55, 0.60, 0.65], "exit_thresholds": [0.45, 0.50, 0.55]},
    "SOLUSDT": {"enter_thresholds": [0.40, 0.45, 0.50, 0.55], "exit_thresholds": [0.40, 0.45, 0.50]},
}
DEPLOYMENT_SCORE_TOLERANCE = 0.015


def build_binary_lagged_feature_frames(X, y, window_size=LAG_WINDOW):
    """Flatten rolling windows into lagged tabular features with source-row tracking."""
    lagged_rows = []
    lagged_labels = []
    lagged_indices = []
    feature_names = list(X.columns)

    for idx in range(window_size - 1, len(X)):
        window = X.iloc[idx - window_size + 1: idx + 1].to_numpy()
        lagged_rows.append(window.reshape(-1))
        lagged_labels.append(y.iloc[idx])
        lagged_indices.append(idx)

    lagged_columns = []
    for lag in range(window_size):
        lag_suffix = f"t_minus_{window_size - lag - 1}"
        for feature_name in feature_names:
            lagged_columns.append(f"{feature_name}_{lag_suffix}")

    return (
        pd.DataFrame(lagged_rows, columns=lagged_columns),
        pd.Series(lagged_labels, name="binary_label"),
        pd.Series(lagged_indices, name="source_index"),
    )


def split_train_calibration_validation(df):
    """Split one dataset into fit, calibration, validation, and test segments."""
    train_end = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df = df.iloc[train_end:].reset_index(drop=True)

    train_length = len(train_df)
    calibration_start = int(train_length * (1.0 - CALIBRATION_RATIO - VALIDATION_RATIO))
    validation_start = int(train_length * (1.0 - VALIDATION_RATIO))

    fit_df = train_df.iloc[:calibration_start].reset_index(drop=True)
    calibration_df = train_df.iloc[calibration_start:validation_start].reset_index(drop=True)
    validation_df = train_df.iloc[validation_start:].reset_index(drop=True)
    return fit_df, calibration_df, validation_df, test_df


def build_binary_decision_frame(df, model, X):
    """Attach binary probabilities and next-period returns to one slice."""
    probabilities = model.predict_proba(X)
    class_labels = list(model.classes_)
    probability_frame = pd.DataFrame(probabilities, columns=class_labels, index=df.index)
    predicted_label = model.predict(X)

    output_df = df.copy()
    output_df["predicted_label"] = predicted_label
    output_df["long_probability"] = probability_frame.get("long", 0.0)
    output_df["flat_probability"] = probability_frame.get("flat", 0.0)
    output_df["prediction_confidence"] = probability_frame.max(axis=1)
    output_df["next_period_return"] = output_df["close"].shift(-1) / output_df["close"] - 1.0
    return output_df.dropna(subset=["next_period_return"]).reset_index(drop=True)


def build_binary_action_label(previous_position, target_position):
    if target_position > previous_position:
        return "enter_long" if previous_position == 0.0 else "increase_long"
    if target_position < previous_position:
        return "exit_to_cash" if target_position == 0.0 else "reduce_long"
    if target_position == 0.0:
        return "stay_cash"
    return "stay_long"


def apply_binary_policy(df, policy_name, enter_threshold, exit_threshold,
                        transaction_cost_rate=TRANSACTION_COST_RATE, timeframe="4h"):
    """Run one binary long/flat policy on directional model outputs."""
    backtest_df = df.copy().reset_index(drop=True)
    positions = []
    actions = []
    previous_position = 0.0
    intraday_mode = timeframe == "1h"
    daily_mode = timeframe == "1d"

    for _, row in backtest_df.iterrows():
        target_position = previous_position
        bullish_regime = (
            row.get("close", 0.0) > row.get("sma_20", 0.0) > row.get("sma_50", 0.0)
            and row.get("return_24h", 0.0) > 0.0
            and row.get("market_futures_alignment_score", 0.0) >= -0.25
        )
        bearish_regime = (
            row.get("close", 0.0) < row.get("sma_20", 0.0)
            or row.get("return_24h", 0.0) < 0.0
            or row.get("market_futures_alignment_score", 0.0) < -0.25
        )

        if policy_name == "binary_confidence_gated_long_flat":
            if row["predicted_label"] == "long" and row["long_probability"] >= enter_threshold:
                target_position = 1.0
            elif row["flat_probability"] >= exit_threshold:
                target_position = 0.0
            elif intraday_mode and previous_position > 0.0 and (
                bearish_regime or row["long_probability"] < max(enter_threshold - 0.10, 0.25)
            ):
                target_position = 0.0
            elif daily_mode and not bullish_regime:
                target_position = 0.0

        elif policy_name == "binary_regime_adaptive_long_flat":
            dynamic_enter = enter_threshold
            dynamic_exit = exit_threshold
            if bullish_regime:
                dynamic_enter = max(dynamic_enter - 0.05, 0.35)
            if bearish_regime:
                dynamic_enter = min(dynamic_enter + 0.10, 0.75)
                dynamic_exit = max(dynamic_exit - 0.05, 0.30)

            if row["predicted_label"] == "long" and row["long_probability"] >= dynamic_enter:
                target_position = 1.0
            elif row["flat_probability"] >= dynamic_exit or (previous_position > 0.0 and bearish_regime):
                target_position = 0.0
            elif intraday_mode and previous_position > 0.0 and row["long_probability"] < max(dynamic_enter - 0.12, 0.28):
                target_position = 0.0
            elif daily_mode and row["predicted_label"] == "long" and not bullish_regime:
                target_position = 0.0

        elif policy_name == "binary_conviction_weighted_long_only":
            conviction = max(row["long_probability"] - row["flat_probability"], 0.0)
            target_position = 0.0
            if row["predicted_label"] == "long" and row["long_probability"] >= enter_threshold:
                target_position = min(conviction / max(1.0 - enter_threshold, 1e-6), 1.0)
            elif row["flat_probability"] >= exit_threshold:
                target_position = 0.0
            if intraday_mode:
                target_position = min(target_position, 0.75)
                if previous_position > 0.0 and (bearish_regime or row["long_probability"] < max(enter_threshold - 0.10, 0.25)):
                    target_position = 0.0
            if daily_mode:
                target_position = min(target_position, 0.85)
                if not bullish_regime:
                    target_position = 0.0

        action = build_binary_action_label(previous_position, target_position)
        positions.append(target_position)
        actions.append(action)
        previous_position = target_position

    backtest_df["position"] = positions
    backtest_df["action"] = actions
    backtest_df["position_change"] = backtest_df["position"].diff().abs().fillna(backtest_df["position"])
    backtest_df["transaction_cost"] = backtest_df["position_change"] * transaction_cost_rate
    backtest_df["strategy_return"] = (
        backtest_df["position"] * backtest_df["next_period_return"] - backtest_df["transaction_cost"]
    )
    backtest_df["buy_hold_return"] = backtest_df["next_period_return"]
    backtest_df["strategy_equity_curve"] = (1.0 + backtest_df["strategy_return"]).cumprod()
    backtest_df["buy_hold_equity_curve"] = (1.0 + backtest_df["buy_hold_return"]).cumprod()
    return backtest_df


def build_backtest_metrics(backtest_df, timeframe):
    strategy_total_return = float(backtest_df["strategy_equity_curve"].iloc[-1] - 1.0)
    buy_hold_total_return = float(backtest_df["buy_hold_equity_curve"].iloc[-1] - 1.0)
    strategy_volatility = annualize_volatility(backtest_df["strategy_return"], timeframe)
    annualized_strategy_return = annualize_return(strategy_total_return, len(backtest_df), timeframe)
    sharpe_ratio = annualized_strategy_return / strategy_volatility if strategy_volatility > 0 else 0.0

    positive_period_mask = backtest_df["position"] > 0
    active_returns = backtest_df.loc[positive_period_mask, "strategy_return"]
    hit_rate = float((active_returns > 0).mean()) if not active_returns.empty else 0.0

    return {
        "strategy_total_return": strategy_total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "excess_return": strategy_total_return - buy_hold_total_return,
        "annualized_strategy_return": annualized_strategy_return,
        "annualized_strategy_volatility": strategy_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": calculate_max_drawdown(backtest_df["strategy_equity_curve"]),
        "exposure_ratio": float(backtest_df["position"].mean()),
        "trade_count": int((backtest_df["position_change"] > 0).sum()),
        "hit_rate": hit_rate,
    }


def build_policy_score(metrics, timeframe="4h"):
    profile = get_timeframe_deployment_profile(timeframe)
    inactivity_penalty = 0.0
    if metrics["trade_count"] == 0:
        inactivity_penalty += 0.03
    elif metrics["trade_count"] <= 2:
        inactivity_penalty += 0.015
    if metrics["exposure_ratio"] == 0.0:
        inactivity_penalty += 0.02
    elif metrics["exposure_ratio"] < profile["min_active_exposure"]:
        inactivity_penalty += 0.01

    inactivity_penalty *= profile["inactivity_penalty_scale"]
    adjusted_excess_return = metrics["excess_return"] - inactivity_penalty
    participation_bonus = (
        min(metrics["exposure_ratio"], 0.75) * 0.01
        + min(metrics["trade_count"], 8) * 0.001
    ) * profile["participation_bonus_scale"]
    return (
        adjusted_excess_return,
        metrics["strategy_total_return"] + participation_bonus,
        metrics["sharpe_ratio"],
        -abs(metrics["max_drawdown"]),
        metrics["hit_rate"],
    )


def choose_binary_thresholds(validation_df, timeframe, policy_name, symbol):
    best_config = None
    best_score = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
    threshold_space = dict(BINARY_THRESHOLD_SPACES.get(
        symbol,
        {"enter_thresholds": [0.45, 0.50, 0.55, 0.60], "exit_thresholds": [0.45, 0.50, 0.55]},
    ))
    profile = get_timeframe_deployment_profile(timeframe)
    threshold_space["enter_thresholds"] = [
        max(min(round(value + profile["buy_threshold_shift"], 2), 0.85), 0.25)
        for value in threshold_space["enter_thresholds"]
    ]
    threshold_space["exit_thresholds"] = [
        max(min(round(value + profile["exit_threshold_shift"], 2), 0.65), 0.15)
        for value in threshold_space["exit_thresholds"]
    ]

    for enter_threshold in threshold_space["enter_thresholds"]:
        for exit_threshold in threshold_space["exit_thresholds"]:
            backtest_df = apply_binary_policy(
                validation_df,
                policy_name,
                enter_threshold,
                exit_threshold,
                timeframe=timeframe,
            )
            metrics = build_backtest_metrics(backtest_df, timeframe)
            score = build_policy_score(metrics, timeframe=timeframe)
            if score > best_score:
                best_score = score
                best_config = {
                    "policy_name": policy_name,
                    "enter_threshold": enter_threshold,
                    "exit_threshold": exit_threshold,
                    "enter_threshold_grid": "|".join(f"{value:.2f}" for value in threshold_space["enter_thresholds"]),
                    "exit_threshold_grid": "|".join(f"{value:.2f}" for value in threshold_space["exit_thresholds"]),
                    "metrics": metrics,
                    "validation_score": score,
                }

    return best_config


def should_replace_selected_binary_candidate(
    best_validation_score,
    current_validation_score,
    selected_validation_metrics,
    current_validation_metrics,
    selected_deployment_metrics,
    current_deployment_metrics,
    timeframe,
):
    """Prefer live-capable binary candidates when validation scores are close."""
    if selected_validation_metrics is None or selected_deployment_metrics is None:
        return True

    selected_validation_active = is_active_policy(selected_validation_metrics, timeframe=timeframe)
    current_validation_active = is_active_policy(current_validation_metrics, timeframe=timeframe)
    selected_deployment_active = is_active_policy(selected_deployment_metrics, timeframe=timeframe)
    current_deployment_active = is_active_policy(current_deployment_metrics, timeframe=timeframe)

    validation_gap = best_validation_score[0] - current_validation_score[0]

    if (
        current_deployment_active
        and not selected_deployment_active
        and validation_gap <= DEPLOYMENT_SCORE_TOLERANCE
    ):
        return True

    if (
        current_deployment_active
        and not selected_deployment_active
        and not current_validation_active
        and not selected_validation_active
        and validation_gap <= 0.05
    ):
        return True

    if (
        current_deployment_active
        and selected_deployment_active
        and validation_gap <= DEPLOYMENT_SCORE_TOLERANCE
        and current_deployment_metrics["excess_return"] > selected_deployment_metrics["excess_return"] + 0.0025
    ):
        return True

    if (
        current_deployment_active
        and selected_deployment_active
        and validation_gap <= DEPLOYMENT_SCORE_TOLERANCE
        and current_deployment_metrics["trade_count"] > selected_deployment_metrics["trade_count"]
        and current_deployment_metrics["exposure_ratio"] >= max(selected_deployment_metrics["exposure_ratio"] - 0.02, 0.0)
    ):
        return True

    return should_replace_selected_policy(
        best_validation_score,
        current_validation_score,
        selected_validation_metrics,
        current_validation_metrics,
        timeframe=timeframe,
    )


def build_binary_backend_candidate_frames(df, X, fit_end, validation_end):
    y = df["binary_label"].reset_index(drop=True)
    candidates = []

    candidates.append(
        {
            "backend_model_name": "market_futures_binary_logistic",
            "fit_X": X.iloc[:fit_end].reset_index(drop=True),
            "fit_y": y.iloc[:fit_end].reset_index(drop=True),
            "validation_X": X.iloc[fit_end:validation_end].reset_index(drop=True),
            "validation_y": y.iloc[fit_end:validation_end].reset_index(drop=True),
            "validation_df": df.iloc[fit_end:validation_end].reset_index(drop=True),
            "test_X": X.iloc[validation_end:].reset_index(drop=True),
            "test_y": y.iloc[validation_end:].reset_index(drop=True),
            "test_df": df.iloc[validation_end:].reset_index(drop=True),
            "trainer": train_scaled_logistic_baseline,
        }
    )

    lagged_X, lagged_y, lagged_indices = build_binary_lagged_feature_frames(X, y, window_size=LAG_WINDOW)
    fit_mask = lagged_indices < fit_end
    validation_mask = (lagged_indices >= fit_end) & (lagged_indices < validation_end)
    test_mask = lagged_indices >= validation_end

    specs = [
        ("market_futures_binary_lagged_logistic", train_scaled_logistic_baseline),
        ("market_futures_binary_lagged_forest", train_lagged_random_forest_baseline),
    ]
    for backend_model_name, trainer in specs:
        if not fit_mask.any() or not validation_mask.any() or not test_mask.any():
            continue
        candidates.append(
            {
                "backend_model_name": backend_model_name,
                "fit_X": lagged_X.loc[fit_mask].reset_index(drop=True),
                "fit_y": lagged_y.loc[fit_mask].reset_index(drop=True),
                "validation_X": lagged_X.loc[validation_mask].reset_index(drop=True),
                "validation_y": lagged_y.loc[validation_mask].reset_index(drop=True),
                "validation_df": df.iloc[lagged_indices.loc[validation_mask]].reset_index(drop=True),
                "test_X": lagged_X.loc[test_mask].reset_index(drop=True),
                "test_y": lagged_y.loc[test_mask].reset_index(drop=True),
                "test_df": df.iloc[lagged_indices.loc[test_mask]].reset_index(drop=True),
                "trainer": trainer,
            }
        )

    return candidates


def build_binary_backtest_row(symbol, timeframe, start_date, end_date, backend_model_name, target_config,
                              policy_name, thresholds, latest_row, metrics, futures_context=None):
    validation_metrics = thresholds["metrics"]
    validation_policy_active = is_active_policy(validation_metrics, timeframe=timeframe)
    deployment_active = is_active_policy(metrics, timeframe=timeframe)
    futures_context = futures_context or {}
    futures_resilience_label = str(futures_context.get("futures_context_resilience_label", "unavailable") or "unavailable")
    futures_completeness_label = str(futures_context.get("futures_completeness_label", "unavailable") or "unavailable")
    basis_feature_available = bool(futures_context.get("basis_feature_available"))
    futures_support_text = f"Futures support is {futures_resilience_label} with {futures_completeness_label} coverage"
    if futures_completeness_label != "full" and not basis_feature_available:
        futures_support_text += " and basis currently missing"
    futures_support_text += "."
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_start": start_date,
        "window_end": end_date,
        "selected_backend_model": f"{backend_model_name}_{target_config['target_name']}",
        "selected_target_name": target_config["target_name"],
        "target_horizon_hours": target_config.get("effective_horizon_hours"),
        "target_exact_horizon_match": target_config.get("exact_horizon_match"),
        "target_resolution_note": target_config.get("horizon_resolution_note"),
        "policy_name": policy_name,
        "probability_mode": "raw",
        "enter_threshold": thresholds["enter_threshold"],
        "exit_threshold": thresholds["exit_threshold"],
        "enter_threshold_grid": thresholds["enter_threshold_grid"],
        "exit_threshold_grid": thresholds["exit_threshold_grid"],
        "latest_signal": latest_row["predicted_label"],
        "latest_action": latest_row["action"],
        "latest_position": float(latest_row["position"]),
        "latest_signal_confidence": float(latest_row["prediction_confidence"]),
        "validation_strategy_total_return": validation_metrics["strategy_total_return"],
        "validation_buy_hold_total_return": validation_metrics["buy_hold_total_return"],
        "validation_excess_return": validation_metrics["excess_return"],
        "validation_sharpe_ratio": validation_metrics["sharpe_ratio"],
        "validation_max_drawdown": validation_metrics["max_drawdown"],
        "validation_exposure_ratio": validation_metrics["exposure_ratio"],
        "validation_trade_count": validation_metrics["trade_count"],
        "validation_hit_rate": validation_metrics["hit_rate"],
        "validation_policy_active": validation_policy_active,
        "deployment_active": deployment_active,
        **futures_context,
        "strategy_total_return": metrics["strategy_total_return"],
        "buy_hold_total_return": metrics["buy_hold_total_return"],
        "excess_return": metrics["excess_return"],
        "annualized_strategy_return": metrics["annualized_strategy_return"],
        "annualized_strategy_volatility": metrics["annualized_strategy_volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "exposure_ratio": metrics["exposure_ratio"],
        "trade_count": metrics["trade_count"],
        "hit_rate": metrics["hit_rate"],
        "backtest_summary": (
            f"{symbol} uses {policy_name} on top of {backend_model_name}. "
            f"Binary deployment return is {metrics['strategy_total_return'] * 100:.1f}% versus buy-and-hold "
            f"{metrics['buy_hold_total_return'] * 100:.1f}%, with Sharpe {metrics['sharpe_ratio']:.2f}. "
            f"{futures_support_text}"
        ),
    }


def evaluate_binary_backtest_for_symbol(symbol, target_config, timeframe, start_date, end_date):
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
    fit_df, calibration_df, validation_df, test_df = split_train_calibration_validation(df)
    fit_end = len(fit_df)
    validation_end = fit_end + len(calibration_df) + len(validation_df)

    variant_rows = []
    selected_row = None
    selected_curve = None
    best_validation_score = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
    selected_validation_metrics = None
    selected_deployment_metrics = None

    for candidate in build_binary_backend_candidate_frames(df, X, fit_end, validation_end):
        if min(len(candidate["fit_X"]), len(candidate["validation_X"]), len(candidate["test_X"])) == 0:
            continue

        model = candidate["trainer"](candidate["fit_X"], candidate["fit_y"])
        validation_decisions = build_binary_decision_frame(candidate["validation_df"], model, candidate["validation_X"])
        test_decisions = build_binary_decision_frame(candidate["test_df"], model, candidate["test_X"])

        for policy_name in BINARY_POLICY_VARIANTS:
            thresholds = choose_binary_thresholds(validation_decisions, timeframe, policy_name, symbol)
            test_backtest = apply_binary_policy(
                test_decisions,
                policy_name,
                thresholds["enter_threshold"],
                thresholds["exit_threshold"],
                timeframe=timeframe,
            )
            metrics = build_backtest_metrics(test_backtest, timeframe)
            latest_row = test_backtest.iloc[-1]
            row = build_binary_backtest_row(
                symbol,
                timeframe,
                start_date,
                end_date,
                candidate["backend_model_name"],
                target_config,
                policy_name,
                thresholds,
                latest_row,
                metrics,
                futures_context=futures_context,
            )
            variant_rows.append(row)

            if should_replace_selected_binary_candidate(
                best_validation_score,
                thresholds["validation_score"],
                selected_validation_metrics,
                thresholds["metrics"],
                selected_deployment_metrics,
                metrics,
                timeframe,
            ):
                best_validation_score = thresholds["validation_score"]
                selected_validation_metrics = thresholds["metrics"]
                selected_deployment_metrics = metrics
                selected_row = row
                selected_curve = test_backtest

    curve_output = selected_curve[
        [
            "open_time",
            "close",
            "predicted_label",
            "long_probability",
            "flat_probability",
            "prediction_confidence",
            "position",
            "action",
            "strategy_return",
            "buy_hold_return",
            "strategy_equity_curve",
            "buy_hold_equity_curve",
        ]
    ].copy()
    curve_output.to_csv(get_market_futures_binary_backtest_curve_path(symbol, timeframe, start_date, end_date), index=False)
    return selected_row, variant_rows


def evaluate_market_futures_binary_backtests(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Evaluate deployment policies for the binary directional family."""
    selected_rows = []
    variant_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)

    for symbol, target_config in preferred_targets.items():
        selected_row, symbol_variants = evaluate_binary_backtest_for_symbol(
            symbol,
            target_config,
            timeframe,
            start_date,
            end_date,
        )
        selected_rows.append(selected_row)
        variant_rows.extend(symbol_variants)

    summary_df = pd.DataFrame(selected_rows)
    variant_df = pd.DataFrame(variant_rows)
    summary_df.to_csv(get_market_futures_binary_backtest_summary_path(timeframe, start_date, end_date), index=False)
    variant_df.to_csv(get_market_futures_binary_policy_variant_summary_path(timeframe, start_date, end_date), index=False)

    print("market + futures binary backtest summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {get_market_futures_binary_backtest_summary_path(timeframe, start_date, end_date)}")
    print("market + futures binary policy variant summary generated")
    print(f"rows saved: {len(variant_df)}")
    print(f"summary saved to: {get_market_futures_binary_policy_variant_summary_path(timeframe, start_date, end_date)}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_binary_backtests()
