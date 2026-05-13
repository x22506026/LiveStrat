"""Assess how trustworthy sentiment and on-chain context currently are per asset."""

from datetime import datetime, timezone

from pathlib import Path

import pandas as pd

from src.config import (
    ASSET_REGISTRY,
    ONCHAIN_FREQUENCY,
    PROCESSED_DIR,
    SENTIMENT_FREQUENCY,
    get_sentiment_summary_path,
    get_strategy_summary_path,
)


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_optional_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_latest_asset_csv(pattern):
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def _latest_row_or_empty(df):
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _derive_age_days(timestamp_value):
    if not timestamp_value:
        return None
    try:
        timestamp = pd.Timestamp(timestamp_value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        now = pd.Timestamp(datetime.now(timezone.utc))
        return max((now - timestamp).days, 0)
    except Exception:
        return None


def _score_gdelt_reliability(feature_row, summary_row):
    if not summary_row or not bool(summary_row.get("gdelt_sentiment_data_available", False)):
        return 0.0, "unavailable", "Asset-specific news sentiment is unavailable.", None, "unavailable"

    article_count = _safe_float(summary_row.get("gdelt_article_count"))
    domain_count = _safe_float(feature_row.get("gdelt_unique_domain_count"))
    sentiment_strength = abs(_safe_float(summary_row.get("gdelt_sentiment_mean")))
    model_used = str(summary_row.get("gdelt_sentiment_model_used", "unavailable") or "unavailable")
    coverage_quality = _safe_float(feature_row.get("gdelt_coverage_quality_score"))
    event_intensity = _safe_float(feature_row.get("gdelt_event_intensity_score"))
    dominant_theme = str(summary_row.get("gdelt_dominant_event_theme", "none") or "none")
    age_days = _derive_age_days(summary_row.get("latest_window_end"))

    score = 0.0
    score += min(article_count / 8.0, 1.0) * 0.25
    score += min(domain_count / 5.0, 1.0) * 0.15
    score += min(sentiment_strength / 0.30, 1.0) * 0.15
    score += coverage_quality * 0.30
    score += event_intensity * 0.05
    score += 0.10 if model_used == "finbert" else 0.0
    if age_days is not None:
        if age_days > 10:
            score *= 0.55
            freshness_label = "stale"
        elif age_days > 5:
            score *= 0.80
            freshness_label = "aging"
        else:
            freshness_label = "fresh"
    else:
        freshness_label = "unknown"

    if article_count < 2:
        label = "thin_confirmation"
        note = "Asset-specific news exists, but coverage is too thin to trust beyond weak confirmation."
    elif score >= 0.65:
        label = "usable_confirmation"
        note = "Asset-specific news has enough recent coverage to act as a meaningful confirmation layer."
    elif score >= 0.35:
        label = "conditional_confirmation"
        note = "Asset-specific news can be used as conditional confirmation, but coverage is still modest."
    else:
        label = "thin_confirmation"
        note = "Asset-specific news is present, but reliability is still too weak for strong influence."

    if freshness_label == "stale":
        note += " The latest asset-news snapshot is stale, so LiveStrat should discount it further."
    elif freshness_label == "aging":
        note += " The latest asset-news snapshot is aging, so it should stay in a supporting role."
    if dominant_theme != "none":
        note += f" Current dominant theme is {dominant_theme.replace('_', ' ')}."

    return score, label, note, age_days, freshness_label


def _score_broad_sentiment_reliability(sentiment_summary_row):
    if not sentiment_summary_row or not bool(sentiment_summary_row.get("sentiment_data_available", False)):
        return 0.0, "unavailable", "Broad market mood data is unavailable."

    change_1d = abs(_safe_float(sentiment_summary_row.get("sentiment_change_1d")))
    change_7d = abs(_safe_float(sentiment_summary_row.get("sentiment_change_7d")))
    sentiment_value = _safe_float(sentiment_summary_row.get("sentiment_value"))
    extremity = abs(sentiment_value - 50.0) / 50.0

    score = 0.25 + min(extremity, 1.0) * 0.15 + min(change_1d / 20.0, 1.0) * 0.10 + min(change_7d / 30.0, 1.0) * 0.10
    label = "broad_fallback"
    note = "Broad market mood is useful as a market-wide fallback context layer, not as asset-specific evidence."
    return score, label, note


def _score_onchain_reliability(overview_row, strategy_row):
    if not overview_row or not bool(overview_row.get("onchain_data_available", False)):
        return 0.0, "unavailable", "On-chain confirmation is unavailable for this asset."

    regime_score = abs(_safe_float(overview_row.get("latest_onchain_regime_score")))
    macro_f1 = _safe_float(strategy_row.get("test_macro_f1"))
    combined_view = str(overview_row.get("combined_view", "mixed") or "mixed")

    score = 0.35 + min(regime_score / 0.75, 1.0) * 0.30 + min(macro_f1 / 0.40, 1.0) * 0.20
    if combined_view in {"aligned_bullish", "aligned_bearish"}:
        score += 0.10

    if score >= 0.70:
        label = "usable_structural_confirmation"
        note = "On-chain context is strong enough to use as a meaningful daily structural confirmation layer."
    elif score >= 0.45:
        label = "conditional_structural_confirmation"
        note = "On-chain context is usable as slower structural confirmation, but not as a primary timing engine."
    else:
        label = "weak_structural_confirmation"
        note = "On-chain context exists, but reliability is still too weak for strong daily influence."

    return min(score, 1.0), label, note


def _derive_effective_sentiment_role(gdelt_label, broad_label):
    if gdelt_label == "usable_confirmation":
        return "asset_news_confirmation"
    if gdelt_label == "conditional_confirmation":
        return "asset_news_conditional_confirmation"
    if broad_label == "broad_fallback":
        return "broad_market_fallback"
    return "unavailable"


def _derive_combined_context_readiness(sentiment_score, onchain_score, gdelt_label, onchain_label):
    if onchain_label == "usable_structural_confirmation" or gdelt_label == "usable_confirmation":
        return "usable_context_confirmation"
    if (
        onchain_label == "conditional_structural_confirmation"
        or gdelt_label == "conditional_confirmation"
        or (onchain_score >= 0.45 and sentiment_score >= 0.20)
    ):
        return "conditional_context_confirmation"
    return "limited_context_confirmation"


def evaluate_context_reliability():
    """Generate one per-asset reliability table for sentiment and on-chain context."""
    broad_sentiment_df = _load_optional_csv(get_sentiment_summary_path(SENTIMENT_FREQUENCY))
    broad_sentiment_row = _latest_row_or_empty(broad_sentiment_df)
    onchain_overview_df = _load_optional_csv(get_strategy_summary_path("market_onchain", ONCHAIN_FREQUENCY).with_name("market_onchain_overview_1d.csv"))
    onchain_strategy_df = _load_optional_csv(get_strategy_summary_path("market_onchain", ONCHAIN_FREQUENCY))

    summary_rows = []
    detail_rows = []

    for asset_symbol, asset_config in ASSET_REGISTRY.items():
        if asset_config.get("tier") != "core":
            continue

        gdelt_summary_df = _load_latest_asset_csv(f"{asset_symbol}_gdelt_sentiment_summary_{SENTIMENT_FREQUENCY}_*.csv")
        gdelt_features_df = _load_latest_asset_csv(f"{asset_symbol}_gdelt_sentiment_features_{SENTIMENT_FREQUENCY}_*.csv")
        gdelt_summary_row = _latest_row_or_empty(gdelt_summary_df)
        gdelt_feature_row = _latest_row_or_empty(gdelt_features_df)
        onchain_overview_row = _latest_row_or_empty(onchain_overview_df[onchain_overview_df["asset_symbol"] == asset_symbol]) if not onchain_overview_df.empty else {}

        asset_strategy_df = onchain_strategy_df[onchain_strategy_df["asset_symbol"] == asset_symbol].copy() if not onchain_strategy_df.empty else pd.DataFrame()
        if asset_strategy_df.empty:
            onchain_strategy_row = {}
        else:
            onchain_strategy_row = (
                asset_strategy_df.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
                .iloc[0]
                .to_dict()
            )

        gdelt_score, gdelt_label, gdelt_note, gdelt_age_days, gdelt_freshness = _score_gdelt_reliability(gdelt_feature_row, gdelt_summary_row)
        broad_score, broad_label, broad_note = _score_broad_sentiment_reliability(broad_sentiment_row)
        onchain_score, onchain_label, onchain_note = _score_onchain_reliability(onchain_overview_row, onchain_strategy_row)
        effective_sentiment_role = _derive_effective_sentiment_role(gdelt_label, broad_label)
        if effective_sentiment_role.startswith("asset_news"):
            effective_sentiment_score = gdelt_score
        elif effective_sentiment_role == "broad_market_fallback":
            effective_sentiment_score = broad_score
        else:
            effective_sentiment_score = 0.0
        combined_readiness = _derive_combined_context_readiness(effective_sentiment_score, onchain_score, gdelt_label, onchain_label)

        detail_rows.extend(
            [
                {
                    "asset_symbol": asset_symbol,
                    "component": "asset_news_sentiment",
                    "reliability_score": gdelt_score,
                    "reliability_label": gdelt_label,
                    "detail_note": gdelt_note,
                },
                {
                    "asset_symbol": asset_symbol,
                    "component": "broad_market_sentiment",
                    "reliability_score": broad_score,
                    "reliability_label": broad_label,
                    "detail_note": broad_note,
                },
                {
                    "asset_symbol": asset_symbol,
                    "component": "onchain_context",
                    "reliability_score": onchain_score,
                    "reliability_label": onchain_label,
                    "detail_note": onchain_note,
                },
            ]
        )

        summary_rows.append(
            {
                "asset_symbol": asset_symbol,
                "market_symbol": asset_config["market_symbol"],
                "timeframe": "1d",
                "gdelt_article_count": int(_safe_float(gdelt_summary_row.get("gdelt_article_count", 0))),
                "gdelt_regime_label": gdelt_summary_row.get("gdelt_regime_label", "unavailable"),
                "gdelt_reliability_score": gdelt_score,
                "gdelt_reliability_label": gdelt_label,
                "gdelt_age_days": gdelt_age_days,
                "gdelt_freshness_status": gdelt_freshness,
                "gdelt_dominant_event_theme": gdelt_summary_row.get("gdelt_dominant_event_theme", "none"),
                "gdelt_risk_event_theme": gdelt_summary_row.get("gdelt_risk_event_theme", "none"),
                "gdelt_supportive_event_theme": gdelt_summary_row.get("gdelt_supportive_event_theme", "none"),
                "broad_sentiment_label": broad_sentiment_row.get("market_mood_label", "unavailable"),
                "broad_sentiment_reliability_score": broad_score,
                "broad_sentiment_reliability_label": broad_label,
                "effective_sentiment_role": effective_sentiment_role,
                "effective_sentiment_reliability_score": effective_sentiment_score,
                "onchain_regime_label": onchain_overview_row.get("latest_onchain_regime_label", "unavailable"),
                "onchain_reliability_score": onchain_score,
                "onchain_reliability_label": onchain_label,
                "combined_context_readiness": combined_readiness,
                "context_reliability_summary": (
                    f"{asset_symbol} context reliability is {combined_readiness.replace('_', ' ')}. "
                    f"Asset-news sentiment is {gdelt_label.replace('_', ' ')}, broad mood is {broad_label.replace('_', ' ')}, "
                    f"and on-chain is {onchain_label.replace('_', ' ')}."
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = get_strategy_summary_path("context_reliability", "1d")
    detail_path = get_strategy_summary_path("context_reliability_detail", "1d")
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("context reliability summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    print("context reliability detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {detail_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_context_reliability()
