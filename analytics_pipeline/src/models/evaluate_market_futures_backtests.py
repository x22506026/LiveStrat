"""Evaluate calibrated prescriptive policy variants from the preferred market + futures backend."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_market_futures_backtest_curve_path,
    get_market_futures_backtest_summary_path,
    get_market_futures_dataset_path,
    get_market_futures_policy_variant_summary_path,
)
from src.models.evaluate_market_futures_strategies import (
    build_combined_feature_matrix,
    train_scaled_logistic_baseline,
)
from src.models.market_futures_targets import build_target_labels, get_preferred_market_futures_targets


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
CALIBRATION_RATIO = 0.20
VALIDATION_RATIO = 0.20
LAG_WINDOW = 12
TEMPERATURE_GRID = np.arange(0.60, 2.65, 0.10)
TRANSACTION_COST_RATE = 0.001
POLICY_VARIANTS = [
    "confidence_gated_long_flat",
    "regime_adaptive_long_flat",
    "calibrated_confidence_gated_long_flat",
    "calibrated_regime_adaptive_long_flat",
    "conviction_weighted_long_only",
]
MIN_ACTIVE_TRADE_COUNT = 2
MIN_ACTIVE_EXPOSURE = 0.05
ACTIVE_POLICY_SCORE_TOLERANCE = 0.01
PERIODS_PER_YEAR = {
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}
TIMEFRAME_DEPLOYMENT_PROFILES = {
    "1h": {
        "buy_threshold_shift": 0.05,
        "exit_threshold_shift": -0.05,
        "min_active_trade_count": 4,
        "min_active_exposure": 0.08,
        "inactivity_penalty_scale": 1.30,
        "participation_bonus_scale": 1.15,
    },
    "4h": {
        "buy_threshold_shift": 0.0,
        "exit_threshold_shift": 0.0,
        "min_active_trade_count": 2,
        "min_active_exposure": 0.05,
        "inactivity_penalty_scale": 1.0,
        "participation_bonus_scale": 1.0,
    },
    "1d": {
        "buy_threshold_shift": 0.08,
        "exit_threshold_shift": 0.0,
        "min_active_trade_count": 1,
        "min_active_exposure": 0.03,
        "inactivity_penalty_scale": 0.80,
        "participation_bonus_scale": 0.85,
    },
}


def get_timeframe_deployment_profile(timeframe):
    """Return one deployment-tuning profile for a specific timeframe."""
    return TIMEFRAME_DEPLOYMENT_PROFILES.get(timeframe, TIMEFRAME_DEPLOYMENT_PROFILES["4h"])


def train_lagged_logistic_baseline(X_train, y_train):
    """Train the lagged logistic benchmark used in the preferred-model study."""
    return train_scaled_logistic_baseline(X_train, y_train)


def train_lagged_random_forest_baseline(X_train, y_train):
    """Train a nonlinear lagged tabular benchmark for the deployed backbone comparison."""
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
            )),
        ]
    )
    model.fit(X_train, y_train)
    return model


def build_lagged_feature_frames(X, y, window_size=LAG_WINDOW):
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
        pd.Series(lagged_labels, name="label"),
        pd.Series(lagged_indices, name="source_index"),
    )
SYMBOL_THRESHOLD_SPACES = {
    "BTCUSDT": {
        "buy_thresholds": [0.35, 0.40, 0.45, 0.50, 0.55],
        "exit_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45],
    },
    "ETHUSDT": {
        "buy_thresholds": [0.40, 0.45, 0.50, 0.55, 0.60],
        "exit_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45],
    },
    "SOLUSDT": {
        "buy_thresholds": [0.30, 0.35, 0.40, 0.45, 0.50],
        "exit_thresholds": [0.20, 0.25, 0.30, 0.35, 0.40],
    },
}


def stable_softmax(scores):
    """Convert raw scores into probabilities without numerical overflow."""
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def calibrate_multiclass_probabilities(scores, temperature):
    """Apply scalar temperature scaling to multiclass decision scores."""
    safe_temperature = max(float(temperature), 1e-6)
    return stable_softmax(scores / safe_temperature)


def fit_temperature_on_calibration(model, X_calibration, y_calibration):
    """Choose a temperature using the calibration split only."""
    if len(X_calibration) == 0 or len(y_calibration) == 0:
        return 1.0

    decision_scores = model.decision_function(X_calibration)
    if decision_scores.ndim == 1:
        decision_scores = np.column_stack([-decision_scores, decision_scores])

    class_labels = list(model.classes_)
    calibration_labels = set(pd.Series(y_calibration).dropna().unique().tolist())
    if len(class_labels) < 2 or len(calibration_labels) < 2:
        return 1.0
    if not calibration_labels.issubset(set(class_labels)):
        return 1.0

    best_temperature = 1.0
    best_loss = np.inf

    for temperature in TEMPERATURE_GRID:
        probabilities = calibrate_multiclass_probabilities(decision_scores, temperature)
        current_loss = log_loss(y_calibration, probabilities, labels=class_labels)
        if current_loss < best_loss:
            best_loss = current_loss
            best_temperature = float(temperature)

    return best_temperature


def supports_calibrated_probabilities(model):
    """Return whether the backend supports decision-function temperature scaling."""
    return hasattr(model, "decision_function")


def build_decision_frame(df, model, X, temperature=None):
    """Attach predictions and probabilities to the chronological dataset."""
    if temperature is None:
        probabilities = model.predict_proba(X)
    else:
        decision_scores = model.decision_function(X)
        if decision_scores.ndim == 1:
            decision_scores = np.column_stack([-decision_scores, decision_scores])
        probabilities = calibrate_multiclass_probabilities(decision_scores, temperature)

    class_labels = np.asarray(model.classes_)
    probability_frame = pd.DataFrame(probabilities, columns=class_labels, index=df.index)
    predicted_label = class_labels[np.argmax(probabilities, axis=1)]

    output_df = df.copy()
    output_df["predicted_label"] = predicted_label
    output_df["buy_probability"] = probability_frame.get("buy", 0.0)
    output_df["hold_probability"] = probability_frame.get("hold", 0.0)
    output_df["dont_buy_probability"] = probability_frame.get("dont_buy", 0.0)
    output_df["prediction_confidence"] = probability_frame.max(axis=1)
    output_df["next_period_return"] = output_df["close"].shift(-1) / output_df["close"] - 1.0
    return output_df.dropna(subset=["next_period_return"]).reset_index(drop=True)


def build_action_label(previous_position, target_position):
    """Convert position changes into a readable action label."""
    if target_position > previous_position:
        return "enter_long" if previous_position == 0.0 else "increase_long"
    if target_position < previous_position:
        return "exit_to_cash" if target_position == 0.0 else "reduce_long"
    if target_position == 0.0:
        return "stay_cash"
    return "stay_long"


def clamp_probability_threshold(value, lower=0.30, upper=0.80):
    """Keep dynamic policy thresholds in a sensible probability range."""
    return float(min(max(value, lower), upper))


def resolve_base_policy_name(policy_name):
    """Map calibrated policy variants to their underlying policy logic."""
    if policy_name.startswith("calibrated_"):
        return policy_name.replace("calibrated_", "", 1)
    return policy_name


def uses_calibrated_probabilities(policy_name):
    """Return whether a policy variant should consume calibrated probabilities."""
    return policy_name.startswith("calibrated_")


def get_threshold_space(symbol, target_name, policy_name, timeframe="4h"):
    """Return asset-aware and policy-aware threshold grids."""
    base_space = SYMBOL_THRESHOLD_SPACES.get(
        symbol,
        {
            "buy_thresholds": [0.40, 0.45, 0.50, 0.55, 0.60],
            "exit_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45],
        },
    )
    buy_thresholds = list(base_space["buy_thresholds"])
    exit_thresholds = list(base_space["exit_thresholds"])
    base_policy_name = resolve_base_policy_name(policy_name)

    if target_name.startswith("voladj"):
        buy_thresholds = [max(value - 0.05, 0.25) for value in buy_thresholds]
        exit_thresholds = [max(value - 0.05, 0.15) for value in exit_thresholds]

    if base_policy_name == "regime_adaptive_long_flat":
        buy_thresholds = [max(value - 0.05, 0.25) for value in buy_thresholds]
        exit_thresholds = [min(value + 0.05, 0.55) for value in exit_thresholds]

    if base_policy_name == "conviction_weighted_long_only":
        buy_thresholds = [max(value - 0.05, 0.25) for value in buy_thresholds]
        exit_thresholds = [min(value, 0.35) for value in exit_thresholds]

    timeframe_profile = get_timeframe_deployment_profile(timeframe)
    buy_shift = timeframe_profile["buy_threshold_shift"]
    exit_shift = timeframe_profile["exit_threshold_shift"]
    buy_thresholds = [clamp_probability_threshold(value + buy_shift, lower=0.25, upper=0.85) for value in buy_thresholds]
    exit_thresholds = [clamp_probability_threshold(value + exit_shift, lower=0.15, upper=0.65) for value in exit_thresholds]

    return {
        "buy_thresholds": sorted(set(round(value, 2) for value in buy_thresholds)),
        "exit_thresholds": sorted(set(round(value, 2) for value in exit_thresholds)),
    }


def apply_prescriptive_policy(df, policy_name, buy_threshold, exit_threshold,
                              transaction_cost_rate=TRANSACTION_COST_RATE, timeframe="4h"):
    """Run one long-only prescriptive policy on chronological model outputs."""
    base_policy_name = resolve_base_policy_name(policy_name)
    backtest_df = df.copy().reset_index(drop=True)
    target_positions = []
    actions = []
    previous_position = 0.0
    intraday_mode = timeframe == "1h"
    daily_mode = timeframe == "1d"

    for _, row in backtest_df.iterrows():
        predicted_label = row["predicted_label"]
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
        high_volatility = row.get("volatility_20", 0.0) >= row.get("volatility_50", row.get("volatility_20", 0.0))

        if base_policy_name == "confidence_gated_long_flat":
            if predicted_label == "buy" and row["buy_probability"] >= buy_threshold:
                target_position = 1.0
            elif predicted_label == "dont_buy" and row["dont_buy_probability"] >= exit_threshold:
                target_position = 0.0
            elif intraday_mode and previous_position > 0.0 and (
                bearish_regime or row["buy_probability"] < max(buy_threshold - 0.10, 0.25)
            ):
                target_position = 0.0
            elif daily_mode and not bullish_regime:
                target_position = 0.0

        elif base_policy_name == "regime_adaptive_long_flat":
            dynamic_buy_threshold = buy_threshold
            dynamic_exit_threshold = exit_threshold

            if bullish_regime and row.get("futures_activity_score", 0.0) >= -0.15:
                dynamic_buy_threshold -= 0.10
                dynamic_exit_threshold += 0.05

            if bearish_regime:
                dynamic_buy_threshold += 0.10
                dynamic_exit_threshold -= 0.05

            if high_volatility:
                dynamic_buy_threshold += 0.05

            dynamic_buy_threshold = clamp_probability_threshold(dynamic_buy_threshold)
            dynamic_exit_threshold = clamp_probability_threshold(dynamic_exit_threshold)

            if predicted_label == "buy" and row["buy_probability"] >= dynamic_buy_threshold:
                target_position = 1.0
            elif predicted_label == "dont_buy" and row["dont_buy_probability"] >= dynamic_exit_threshold:
                target_position = 0.0
            elif previous_position > 0.0 and bearish_regime and row["buy_probability"] < 0.45:
                target_position = 0.0
            elif intraday_mode and previous_position > 0.0 and row["buy_probability"] < max(dynamic_buy_threshold - 0.12, 0.28):
                target_position = 0.0
            elif daily_mode and predicted_label == "buy" and not bullish_regime:
                target_position = 0.0

        elif base_policy_name == "conviction_weighted_long_only":
            conviction_score = max(row["buy_probability"] - row["dont_buy_probability"], 0.0)
            target_position = 0.0
            if predicted_label == "buy" and row["buy_probability"] >= buy_threshold:
                target_position = min(conviction_score / max(1.0 - buy_threshold, 1e-6), 1.0)
            if predicted_label == "dont_buy" and row["dont_buy_probability"] >= exit_threshold:
                target_position = 0.0
            if intraday_mode:
                target_position = min(target_position, 0.75)
                if previous_position > 0.0 and (bearish_regime or row["buy_probability"] < max(buy_threshold - 0.10, 0.25)):
                    target_position = 0.0
            if daily_mode:
                target_position = min(target_position, 0.85)
                if not bullish_regime:
                    target_position = 0.0

        action = build_action_label(previous_position, target_position)
        target_positions.append(target_position)
        actions.append(action)
        previous_position = target_position

    backtest_df["position"] = target_positions
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


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from an equity curve series."""
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def annualize_return(total_return, periods, timeframe):
    """Convert compounded return to an annualized figure."""
    periods_per_year = PERIODS_PER_YEAR[timeframe]
    if periods <= 0:
        return 0.0
    return float((1.0 + total_return) ** (periods_per_year / periods) - 1.0)


def annualize_volatility(returns, timeframe):
    """Annualize realized volatility from periodic strategy returns."""
    if len(returns) <= 1:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR[timeframe]))


def build_backtest_metrics(backtest_df, timeframe):
    """Compute compact backtest metrics for app/report use."""
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
    """Rank policies with a preference for excess return, then risk-adjusted quality."""
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


def is_active_policy(metrics, timeframe="4h"):
    """Return whether a policy is materially participating rather than staying idle."""
    profile = get_timeframe_deployment_profile(timeframe)
    return (
        int(metrics.get("trade_count", 0)) >= profile["min_active_trade_count"]
        or float(metrics.get("exposure_ratio", 0.0)) >= profile["min_active_exposure"]
    )


def build_prediction_concentration_summary(decision_df):
    """Summarize whether a backend collapses into one dominant predicted class."""
    if decision_df is None or decision_df.empty:
        return {
            "predicted_buy_share": 0.0,
            "predicted_hold_share": 0.0,
            "predicted_dont_buy_share": 0.0,
            "mean_buy_probability": 0.0,
            "mean_hold_probability": 0.0,
            "mean_dont_buy_probability": 0.0,
            "dominant_predicted_label": "unavailable",
            "dominant_predicted_share": 0.0,
            "prediction_concentration_flag": "unavailable",
        }

    predicted_counts = decision_df["predicted_label"].value_counts(normalize=True)
    dominant_label = predicted_counts.index[0]
    dominant_share = float(predicted_counts.iloc[0])

    if dominant_share >= 0.9:
        concentration_flag = f"{dominant_label}_collapse"
    elif dominant_share >= 0.7:
        concentration_flag = f"{dominant_label}_dominant"
    else:
        concentration_flag = "distributed"

    return {
        "predicted_buy_share": float(predicted_counts.get("buy", 0.0)),
        "predicted_hold_share": float(predicted_counts.get("hold", 0.0)),
        "predicted_dont_buy_share": float(predicted_counts.get("dont_buy", 0.0)),
        "mean_buy_probability": float(decision_df["buy_probability"].mean()),
        "mean_hold_probability": float(decision_df["hold_probability"].mean()),
        "mean_dont_buy_probability": float(decision_df["dont_buy_probability"].mean()),
        "dominant_predicted_label": str(dominant_label),
        "dominant_predicted_share": dominant_share,
        "prediction_concentration_flag": concentration_flag,
    }


def should_replace_selected_policy(current_score, candidate_score, current_metrics, candidate_metrics, timeframe="4h"):
    """Prefer materially active candidates when validation scores are close."""
    if current_metrics is None:
        return True

    current_active = is_active_policy(current_metrics, timeframe=timeframe)
    candidate_active = is_active_policy(candidate_metrics, timeframe=timeframe)

    if candidate_active and not current_active:
        return candidate_score[0] >= current_score[0] - ACTIVE_POLICY_SCORE_TOLERANCE
    if current_active and not candidate_active:
        return False
    return candidate_score > current_score


def choose_thresholds(validation_df, timeframe, policy_name, symbol, target_name):
    """Select thresholds for one policy using only the validation slice."""
    best_config = None
    best_score = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
    threshold_space = get_threshold_space(symbol, target_name, policy_name, timeframe=timeframe)

    for buy_threshold in threshold_space["buy_thresholds"]:
        for exit_threshold in threshold_space["exit_thresholds"]:
            backtest_df = apply_prescriptive_policy(
                validation_df,
                policy_name,
                buy_threshold,
                exit_threshold,
                timeframe=timeframe,
            )
            metrics = build_backtest_metrics(backtest_df, timeframe)
            score = build_policy_score(metrics, timeframe=timeframe)
            if score > best_score:
                best_score = score
                best_config = {
                    "policy_name": policy_name,
                    "buy_threshold": buy_threshold,
                    "exit_threshold": exit_threshold,
                    "buy_threshold_grid": "|".join(f"{value:.2f}" for value in threshold_space["buy_thresholds"]),
                    "exit_threshold_grid": "|".join(f"{value:.2f}" for value in threshold_space["exit_thresholds"]),
                    "metrics": metrics,
                    "validation_score": score,
                }

    return best_config


def split_train_calibration_validation(df):
    """Split the train window into fit, calibration, validation, and test segments."""
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


def build_backend_candidate_frames(df, X, fit_end, calibration_end, validation_end):
    """Build candidate backend datasets for static and lagged models."""
    y = df["label"].reset_index(drop=True)

    candidates = []

    static_candidate = {
        "backend_model_name": "market_futures_logistic_preferred",
        "fit_X": X.iloc[:fit_end].reset_index(drop=True),
        "fit_y": y.iloc[:fit_end].reset_index(drop=True),
        "calibration_X": X.iloc[fit_end:calibration_end].reset_index(drop=True),
        "calibration_y": y.iloc[fit_end:calibration_end].reset_index(drop=True),
        "validation_X": X.iloc[calibration_end:validation_end].reset_index(drop=True),
        "validation_y": y.iloc[calibration_end:validation_end].reset_index(drop=True),
        "validation_df": df.iloc[calibration_end:validation_end].reset_index(drop=True),
        "test_X": X.iloc[validation_end:].reset_index(drop=True),
        "test_y": y.iloc[validation_end:].reset_index(drop=True),
        "test_df": df.iloc[validation_end:].reset_index(drop=True),
        "trainer": train_scaled_logistic_baseline,
    }
    candidates.append(static_candidate)

    lagged_X, lagged_y, lagged_indices = build_lagged_feature_frames(X, y, window_size=LAG_WINDOW)
    fit_mask = lagged_indices < fit_end
    calibration_mask = (lagged_indices >= fit_end) & (lagged_indices < calibration_end)
    validation_mask = (lagged_indices >= calibration_end) & (lagged_indices < validation_end)
    test_mask = lagged_indices >= validation_end

    lagged_specs = [
        ("market_futures_lagged_logistic_preferred", train_lagged_logistic_baseline),
        ("market_futures_lagged_forest_preferred", train_lagged_random_forest_baseline),
    ]

    for backend_model_name, trainer in lagged_specs:
        if not fit_mask.any() or not validation_mask.any() or not test_mask.any():
            continue
        candidates.append(
            {
                "backend_model_name": backend_model_name,
                "fit_X": lagged_X.loc[fit_mask].reset_index(drop=True),
                "fit_y": lagged_y.loc[fit_mask].reset_index(drop=True),
                "calibration_X": lagged_X.loc[calibration_mask].reset_index(drop=True),
                "calibration_y": lagged_y.loc[calibration_mask].reset_index(drop=True),
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


def build_backtest_row(symbol, timeframe, start_date, end_date, selected_backend_model, target_name, policy_name,
                       thresholds, latest_row, metrics, calibration_temperature, target_config,
                       decision_concentration=None, futures_context=None):
    """Build one machine-friendly row for a policy evaluation."""
    validation_metrics = thresholds["metrics"]
    validation_policy_active = is_active_policy(validation_metrics, timeframe=timeframe)
    deployment_active = is_active_policy(metrics, timeframe=timeframe)
    decision_concentration = decision_concentration or build_prediction_concentration_summary(None)
    futures_context = futures_context or {}
    futures_resilience_label = str(futures_context.get("futures_context_resilience_label", "unavailable") or "unavailable")
    futures_completeness_label = str(futures_context.get("futures_completeness_label", "unavailable") or "unavailable")
    basis_feature_available = bool(futures_context.get("basis_feature_available"))
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_start": start_date,
        "window_end": end_date,
        "selected_backend_model": f"{selected_backend_model}_{target_name}",
        "selected_target_name": target_name,
        "target_horizon_hours": target_config.get("effective_horizon_hours"),
        "target_exact_horizon_match": target_config.get("exact_horizon_match"),
        "target_resolution_note": target_config.get("horizon_resolution_note"),
        "policy_name": policy_name,
        "probability_mode": "temperature_scaled" if uses_calibrated_probabilities(policy_name) else "raw",
        "calibration_temperature": calibration_temperature,
        "buy_threshold": thresholds["buy_threshold"],
        "exit_threshold": thresholds["exit_threshold"],
        "buy_threshold_grid": thresholds["buy_threshold_grid"],
        "exit_threshold_grid": thresholds["exit_threshold_grid"],
        "latest_signal": latest_row["predicted_label"],
        "latest_action": latest_row["action"],
        "latest_position": float(latest_row["position"]),
        "latest_signal_confidence": float(latest_row["prediction_confidence"]),
        "validation_strategy_total_return": thresholds["metrics"]["strategy_total_return"],
        "validation_buy_hold_total_return": thresholds["metrics"]["buy_hold_total_return"],
        "validation_excess_return": thresholds["metrics"]["excess_return"],
        "validation_sharpe_ratio": thresholds["metrics"]["sharpe_ratio"],
        "validation_max_drawdown": thresholds["metrics"]["max_drawdown"],
        "validation_exposure_ratio": thresholds["metrics"]["exposure_ratio"],
        "validation_trade_count": thresholds["metrics"]["trade_count"],
        "validation_hit_rate": thresholds["metrics"]["hit_rate"],
        "validation_policy_active": validation_policy_active,
        "deployment_active": deployment_active,
        **futures_context,
        **decision_concentration,
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
            f"{symbol} uses {policy_name} on top of {selected_backend_model}. "
            f"Test return is {metrics['strategy_total_return'] * 100:.1f}% versus buy-and-hold "
            f"{metrics['buy_hold_total_return'] * 100:.1f}%, with Sharpe {metrics['sharpe_ratio']:.2f} "
            f"and max drawdown {metrics['max_drawdown'] * 100:.1f}%. Futures support is "
            f"{futures_resilience_label.replace('_', ' ')} with {futures_completeness_label.replace('_', ' ')} coverage"
            f"{'' if basis_feature_available else ' and basis currently missing'}. "
            f"Predictions are currently {decision_concentration['prediction_concentration_flag'].replace('_', ' ')}."
        ),
    }


def evaluate_backtest_for_symbol(symbol, target_config, timeframe, start_date, end_date):
    """Run calibrated and uncalibrated policy variants and keep the best one."""
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
    fit_df, calibration_df, validation_df, test_df = split_train_calibration_validation(df)

    fit_end = len(fit_df)
    calibration_end = fit_end + len(calibration_df)
    validation_end = calibration_end + len(validation_df)

    variant_rows = []
    selected_row = None
    selected_curve = None
    best_validation_score = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
    selected_validation_metrics = None

    for candidate in build_backend_candidate_frames(df, X, fit_end, calibration_end, validation_end):
        if min(len(candidate["fit_X"]), len(candidate["validation_X"]), len(candidate["test_X"])) == 0:
            continue

        base_model = candidate["trainer"](candidate["fit_X"], candidate["fit_y"])
        calibration_temperature = None
        if supports_calibrated_probabilities(base_model) and len(candidate["calibration_X"]) > 0:
            calibration_temperature = fit_temperature_on_calibration(
                base_model,
                candidate["calibration_X"],
                candidate["calibration_y"],
            )

        validation_decisions_raw = build_decision_frame(
            candidate["validation_df"],
            base_model,
            candidate["validation_X"],
        )
        test_decisions_raw = build_decision_frame(
            candidate["test_df"],
            base_model,
            candidate["test_X"],
        )

        validation_decisions_calibrated = None
        test_decisions_calibrated = None
        if calibration_temperature is not None:
            validation_decisions_calibrated = build_decision_frame(
                candidate["validation_df"],
                base_model,
                candidate["validation_X"],
                temperature=calibration_temperature,
            )
            test_decisions_calibrated = build_decision_frame(
                candidate["test_df"],
                base_model,
                candidate["test_X"],
                temperature=calibration_temperature,
            )

        for policy_name in POLICY_VARIANTS:
            if uses_calibrated_probabilities(policy_name) and calibration_temperature is None:
                continue

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
            metrics = build_backtest_metrics(test_backtest, timeframe)
            decision_concentration = build_prediction_concentration_summary(test_decisions)
            latest_row = test_backtest.iloc[-1]
            row = build_backtest_row(
                symbol,
                timeframe,
                start_date,
                end_date,
                candidate["backend_model_name"],
                target_config["target_name"],
                policy_name,
                thresholds,
                latest_row,
                metrics,
                calibration_temperature if calibration_temperature is not None else 1.0,
                target_config,
                decision_concentration=decision_concentration,
                futures_context=futures_context,
            )
            variant_rows.append(row)

            if should_replace_selected_policy(
                best_validation_score,
                thresholds["validation_score"],
                selected_validation_metrics,
                thresholds["metrics"],
                timeframe=timeframe,
            ):
                best_validation_score = thresholds["validation_score"]
                selected_validation_metrics = thresholds["metrics"]
                selected_row = row
                selected_curve = test_backtest

    curve_output = selected_curve[
        [
            "open_time",
            "close",
            "predicted_label",
            "buy_probability",
            "dont_buy_probability",
            "prediction_confidence",
            "position",
            "action",
            "strategy_return",
            "buy_hold_return",
            "strategy_equity_curve",
            "buy_hold_equity_curve",
        ]
    ].copy()
    curve_output.to_csv(
        get_market_futures_backtest_curve_path(symbol, timeframe, start_date, end_date),
        index=False,
    )

    return selected_row, variant_rows


def evaluate_market_futures_backtests(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Evaluate the prescriptive strategy layer for each preferred target."""
    selected_rows = []
    variant_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)
    for symbol, target_config in preferred_targets.items():
        selected_row, symbol_variant_rows = evaluate_backtest_for_symbol(
            symbol,
            target_config,
            timeframe,
            start_date,
            end_date,
        )
        selected_rows.append(selected_row)
        variant_rows.extend(symbol_variant_rows)

    summary_df = pd.DataFrame(selected_rows)
    summary_path = get_market_futures_backtest_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(summary_path, index=False)

    variant_summary_df = pd.DataFrame(variant_rows)
    variant_summary_path = get_market_futures_policy_variant_summary_path(
        timeframe,
        start_date,
        end_date,
    )
    variant_summary_df.to_csv(variant_summary_path, index=False)

    print("market + futures backtest summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    print("market + futures policy variant summary generated")
    print(f"rows saved: {len(variant_summary_df)}")
    print(f"summary saved to: {variant_summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_backtests()
