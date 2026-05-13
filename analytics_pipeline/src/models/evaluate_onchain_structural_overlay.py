"""Evaluate a daily on-chain structural overlay family for support, caution, and risk-off use."""

import pandas as pd

from src.config import ONCHAIN_FREQUENCY, TRAIN_RATIO, get_market_onchain_dataset_path, get_strategy_summary_path
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)


TIMEFRAME = ONCHAIN_FREQUENCY
SUPPORTED_ASSETS = ["BTC", "ETH"]
STRATEGY_GROUP = "onchain_structural_overlay"
DETAIL_GROUP = "onchain_structural_overlay_detail"


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_clip(series, lower=-2.0, upper=2.0, scale=2.0):
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower, upper) / scale


def build_daily_labels(df):
    """Create next-day labels from the market close."""
    df = df.copy()
    df["future_market_close"] = df["market_close"].shift(-1)
    df["future_return_1d"] = (df["future_market_close"] / df["market_close"]) - 1.0
    df["label"] = "hold"
    df.loc[df["future_return_1d"] >= 0.02, "label"] = "buy"
    df.loc[df["future_return_1d"] <= -0.02, "label"] = "dont_buy"
    return df.dropna(subset=["future_market_close", "future_return_1d"]).reset_index(drop=True)


def build_overlay_components(df):
    """Turn on-chain components into explicit overlay support and risk scores."""
    overlay_df = df.copy()

    network_activity_score = (
        0.6 * _normalized_clip(overlay_df["active_addresses_zscore_30d"])
        + 0.4 * _normalized_clip(overlay_df["transaction_count_zscore_30d"])
    ).clip(-1.5, 1.5)
    economic_activity_score = (
        _normalized_clip(overlay_df["economic_activity_zscore_30d"])
    ).clip(-1.5, 1.5)
    valuation_state_score = pd.Series(0.0, index=overlay_df.index)
    valuation_ratio = pd.to_numeric(overlay_df["valuation_ratio"], errors="coerce")
    valuation_state_score = valuation_state_score.mask(valuation_ratio >= 2.4, -1.25)
    valuation_state_score = valuation_state_score.mask((valuation_ratio >= 1.8) & (valuation_ratio < 2.4), -0.5)
    valuation_state_score = valuation_state_score.mask((valuation_ratio >= 1.2) & (valuation_ratio < 1.8), 0.25)
    valuation_state_score = valuation_state_score.mask((valuation_ratio > 0) & (valuation_ratio < 1.2), 0.75).fillna(0.0)
    exchange_flow_score = (
        -_normalized_clip(overlay_df["exchange_netflow_zscore_30d"])
    ).clip(-1.5, 1.5)

    valuation_tailwind = valuation_state_score.clip(lower=0.0).fillna(0.0)
    market_stress_flag = (
        (pd.to_numeric(overlay_df["market_close"], errors="coerce") < pd.to_numeric(overlay_df["market_sma_20"], errors="coerce")) &
        (pd.to_numeric(overlay_df["market_return_24h"], errors="coerce").fillna(0.0) < 0.0)
    ).astype(float)
    activity_deterioration = (
        (network_activity_score < -0.10) |
        (economic_activity_score < -0.10)
    ).astype(float)
    exchange_inflow_pressure = pd.to_numeric(
        overlay_df["exchange_netflow_zscore_30d"],
        errors="coerce",
    ).clip(lower=0.0).fillna(0.0).clip(upper=2.0) / 2.0
    component_count_source = (
        overlay_df["onchain_component_count"]
        if "onchain_component_count" in overlay_df.columns
        else pd.Series(0.0, index=overlay_df.index)
    )
    component_count = pd.to_numeric(
        component_count_source,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=4.0)
    overlay_confidence = (
        0.65 * (component_count / 4.0) +
        0.35 * pd.to_numeric(overlay_df["onchain_regime_score"], errors="coerce").abs().fillna(0.0).clip(upper=1.0)
    ).clip(lower=0.20, upper=1.0)

    exchange_risk = (
        (-exchange_flow_score).clip(lower=0.0).fillna(0.0)
    )
    exchange_relief = (
        exchange_flow_score.clip(lower=0.0).fillna(0.0)
    )
    valuation_risk = (
        (-valuation_state_score).clip(lower=0.0).fillna(0.0)
    )
    activity_support = network_activity_score.clip(lower=0.0).fillna(0.0)
    economic_support = economic_activity_score.clip(lower=0.0).fillna(0.0)
    activity_deterioration_score = (
        0.50 * (-network_activity_score).clip(lower=0.0).fillna(0.0).clip(upper=1.5) / 1.5
        + 0.50 * (-economic_activity_score).clip(lower=0.0).fillna(0.0).clip(upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    participation_breadth_score = (
        0.50 * activity_support.clip(upper=1.5) / 1.5
        + 0.50 * economic_support.clip(upper=1.5) / 1.5
    ).clip(lower=0.0, upper=1.0)
    divergence_risk = (
        market_stress_flag * (
            0.55 * exchange_inflow_pressure +
            0.45 * activity_deterioration
        )
    ).clip(lower=0.0, upper=1.0)
    fragility_score = (
        0.40 * exchange_risk.clip(upper=1.5) / 1.5
        + 0.30 * valuation_risk.clip(upper=1.5) / 1.5
        + 0.30 * activity_deterioration_score
    ).clip(lower=0.0, upper=1.0)

    raw_risk_score = (
        0.45 * exchange_risk.clip(upper=1.5) / 1.5
        + 0.35 * valuation_risk.clip(upper=1.5) / 1.5
        + 0.20 * exchange_inflow_pressure
    )
    raw_support_score = (
        0.30 * activity_support.clip(upper=1.5) / 1.5
        + 0.20 * economic_support.clip(upper=1.5) / 1.5
        + 0.20 * participation_breadth_score
        + 0.10 * exchange_relief.clip(upper=1.5) / 1.5
        + 0.20 * (
            pd.to_numeric(overlay_df["onchain_regime_score"], errors="coerce").clip(lower=0.0).fillna(0.0).clip(upper=1.5) / 1.5
        )
    )

    overlay_df["onchain_overlay_confidence"] = overlay_confidence
    overlay_df["onchain_divergence_risk_score"] = divergence_risk
    overlay_df["onchain_structural_fragility_score"] = fragility_score
    overlay_df["onchain_participation_breadth_score"] = participation_breadth_score
    overlay_df["onchain_overlay_risk_score"] = (
        raw_risk_score * (0.65 + 0.35 * overlay_confidence) +
        0.10 * divergence_risk +
        0.10 * fragility_score
    ).clip(lower=0.0, upper=1.0)
    overlay_df["onchain_overlay_support_score"] = (
        raw_support_score * (0.60 + 0.40 * overlay_confidence) +
        0.08 * (valuation_tailwind.clip(upper=1.0) / 1.0)
    ).clip(lower=0.0, upper=1.0)

    overlay_df["onchain_overlay_mode"] = "structural_mixed"
    overlay_df.loc[
        (overlay_df["onchain_overlay_risk_score"] >= 0.48) |
        (
            (overlay_df["onchain_divergence_risk_score"] >= 0.45) &
            (overlay_df["onchain_overlay_confidence"] >= 0.45)
        ),
        "onchain_overlay_mode"
    ] = "structural_risk_off"
    overlay_df.loc[
        (overlay_df["onchain_overlay_support_score"] >= 0.48) &
        (overlay_df["onchain_overlay_risk_score"] < 0.42),
        "onchain_overlay_mode"
    ] = "structural_support"
    overlay_df.loc[
        (overlay_df["onchain_overlay_mode"] == "structural_mixed") &
        (
            (overlay_df["onchain_overlay_risk_score"] >= 0.30) |
            (overlay_df["onchain_overlay_support_score"] >= 0.30)
        ),
        "onchain_overlay_mode"
    ] = "structural_watch"
    return overlay_df


def apply_onchain_structural_veto(df):
    """Use on-chain as a structural veto on market direction."""
    df = build_overlay_components(df)
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_close"] > df["market_sma_50"]) &
        (df["market_return_24h"] > 0) &
        (df["onchain_overlay_support_score"] >= 0.45) &
        (df["onchain_overlay_risk_score"] < 0.45)
    )
    dont_buy_condition = (
        (df["onchain_overlay_risk_score"] >= 0.45) |
        (
            (df["market_close"] < df["market_sma_50"]) &
            (df["market_return_24h"] < 0) &
            (df["onchain_overlay_risk_score"] >= 0.30)
        )
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def apply_onchain_structural_confirmation(df):
    """Require stronger market and on-chain agreement before acting."""
    df = build_overlay_components(df)
    df["strategy_signal"] = "hold"

    buy_condition = (
        (df["market_close"] > df["market_sma_20"]) &
        (df["market_sma_20"] > df["market_sma_50"]) &
        (df["market_price_sma_20_diff"] > 0) &
        (df["onchain_overlay_mode"] == "structural_support")
    )
    dont_buy_condition = (
        (df["market_close"] < df["market_sma_20"]) &
        (df["market_sma_20"] < df["market_sma_50"]) &
        (
            (df["onchain_overlay_mode"] == "structural_risk_off") |
            (df["onchain_regime_label"] == "weakening")
        )
    )

    df.loc[buy_condition, "strategy_signal"] = "buy"
    df.loc[dont_buy_condition, "strategy_signal"] = "dont_buy"
    return df


def evaluate_strategy(df, asset_symbol, model_name):
    """Evaluate one overlay strategy with the shared helpers."""
    _, _, _, y_true = make_time_based_split(df[["market_close"]], df["label"], TRAIN_RATIO)
    split_idx = int(len(df) * TRAIN_RATIO)
    y_pred = df["strategy_signal"].iloc[split_idx:]

    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, asset_symbol, TIMEFRAME)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)
    print_evaluation_summary(f"{asset_symbol} {model_name} evaluation", y_true, y_pred)
    return metrics_df, confusion_df


def _metric(metrics_df, name):
    return float(metrics_df.loc[metrics_df["metric"] == name, "value"].iloc[0])


def build_summary_row(asset_symbol, strategy_name, df, metrics_df):
    """Create one latest overlay summary row."""
    latest = df.iloc[-1]
    support_driver = str(latest.get("onchain_primary_support_driver", "none") or "none")
    risk_driver = str(latest.get("onchain_primary_risk_driver", "none") or "none")
    return {
        "asset_symbol": asset_symbol,
        "timeframe": TIMEFRAME,
        "strategy_name": strategy_name,
        "latest_window_end": latest["window_end_utc"],
        "latest_signal": latest["strategy_signal"],
        "latest_onchain_regime_label": latest["onchain_regime_label"],
        "latest_onchain_regime_score": _safe_float(latest["onchain_regime_score"]),
        "latest_overlay_mode": latest["onchain_overlay_mode"],
        "latest_overlay_risk_score": _safe_float(latest["onchain_overlay_risk_score"]),
        "latest_overlay_support_score": _safe_float(latest["onchain_overlay_support_score"]),
        "latest_overlay_confidence": _safe_float(latest["onchain_overlay_confidence"]),
        "latest_divergence_risk_score": _safe_float(latest["onchain_divergence_risk_score"]),
        "latest_onchain_primary_support_driver": support_driver,
        "latest_onchain_primary_risk_driver": risk_driver,
        "test_accuracy": _metric(metrics_df, "accuracy"),
        "test_macro_f1": _metric(metrics_df, "macro_f1"),
        "test_balanced_accuracy": _metric(metrics_df, "balanced_accuracy"),
        "overlay_summary": (
            f"{asset_symbol} {strategy_name} currently suggests {latest['strategy_signal']}. "
            f"On-chain regime is {latest['onchain_regime_label']} and overlay mode is "
            f"{str(latest['onchain_overlay_mode']).replace('_', ' ')} "
            f"(risk {float(latest['onchain_overlay_risk_score']):.2f}, support {float(latest['onchain_overlay_support_score']):.2f}, "
            f"confidence {float(latest['onchain_overlay_confidence']):.2f}). "
            f"Primary support driver is {support_driver.replace('_', ' ')}, primary risk driver is {risk_driver.replace('_', ' ')}."
        ),
    }


def evaluate_onchain_structural_overlay():
    """Run the structural overlay family on supported daily assets."""
    summary_rows = []
    detail_rows = []

    for asset_symbol in SUPPORTED_ASSETS:
        dataset_path = get_market_onchain_dataset_path(asset_symbol, TIMEFRAME)
        df = pd.read_csv(dataset_path)
        df = df[df["onchain_data_available"] == True].copy()
        df = build_daily_labels(df)

        veto_df = apply_onchain_structural_veto(df)
        veto_metrics, _ = evaluate_strategy(veto_df, asset_symbol, "onchain_structural_veto")
        summary_rows.append(build_summary_row(asset_symbol, "onchain_structural_veto", veto_df, veto_metrics))
        detail_rows.append(
            {
                "asset_symbol": asset_symbol,
                "strategy_name": "onchain_structural_veto",
                "component": "structural_overlay",
                "latest_overlay_mode": veto_df.iloc[-1]["onchain_overlay_mode"],
                "latest_overlay_risk_score": _safe_float(veto_df.iloc[-1]["onchain_overlay_risk_score"]),
                "latest_overlay_support_score": _safe_float(veto_df.iloc[-1]["onchain_overlay_support_score"]),
                "latest_overlay_confidence": _safe_float(veto_df.iloc[-1]["onchain_overlay_confidence"]),
                "latest_divergence_risk_score": _safe_float(veto_df.iloc[-1]["onchain_divergence_risk_score"]),
                "latest_onchain_primary_support_driver": str(veto_df.iloc[-1].get("onchain_primary_support_driver", "none") or "none"),
                "latest_onchain_primary_risk_driver": str(veto_df.iloc[-1].get("onchain_primary_risk_driver", "none") or "none"),
            }
        )

        confirmation_df = apply_onchain_structural_confirmation(df)
        confirmation_metrics, _ = evaluate_strategy(confirmation_df, asset_symbol, "onchain_structural_confirmation")
        summary_rows.append(build_summary_row(asset_symbol, "onchain_structural_confirmation", confirmation_df, confirmation_metrics))
        detail_rows.append(
            {
                "asset_symbol": asset_symbol,
                "strategy_name": "onchain_structural_confirmation",
                "component": "structural_overlay",
                "latest_overlay_mode": confirmation_df.iloc[-1]["onchain_overlay_mode"],
                "latest_overlay_risk_score": _safe_float(confirmation_df.iloc[-1]["onchain_overlay_risk_score"]),
                "latest_overlay_support_score": _safe_float(confirmation_df.iloc[-1]["onchain_overlay_support_score"]),
                "latest_overlay_confidence": _safe_float(confirmation_df.iloc[-1]["onchain_overlay_confidence"]),
                "latest_divergence_risk_score": _safe_float(confirmation_df.iloc[-1]["onchain_divergence_risk_score"]),
                "latest_onchain_primary_support_driver": str(confirmation_df.iloc[-1].get("onchain_primary_support_driver", "none") or "none"),
                "latest_onchain_primary_risk_driver": str(confirmation_df.iloc[-1].get("onchain_primary_risk_driver", "none") or "none"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(get_strategy_summary_path(STRATEGY_GROUP, TIMEFRAME), index=False)
    detail_df.to_csv(get_strategy_summary_path(DETAIL_GROUP, TIMEFRAME), index=False)

    print("on-chain structural overlay summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {get_strategy_summary_path(STRATEGY_GROUP, TIMEFRAME)}")
    print("on-chain structural overlay detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {get_strategy_summary_path(DETAIL_GROUP, TIMEFRAME)}")
    return summary_df


if __name__ == "__main__":
    evaluate_onchain_structural_overlay()
