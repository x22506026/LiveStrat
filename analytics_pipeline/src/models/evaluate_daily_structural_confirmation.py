"""Build a daily structural confirmation view from market, on-chain, and news context."""

from pathlib import Path

import pandas as pd

from src.config import (
    ONCHAIN_FREQUENCY,
    PROCESSED_DIR,
    get_market_onchain_overview_path,
    get_strategy_summary_path,
)


TIMEFRAME = ONCHAIN_FREQUENCY
STRATEGY_GROUP = "daily_structural_confirmation"
DETAIL_GROUP = "daily_structural_confirmation_detail"


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_market_onchain_overview():
    return pd.read_csv(get_market_onchain_overview_path(TIMEFRAME))


def _load_market_onchain_strategy_summary():
    return pd.read_csv(get_strategy_summary_path("market_onchain", TIMEFRAME))


def _load_onchain_overlay_summary():
    path = get_strategy_summary_path("onchain_structural_overlay", TIMEFRAME)
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_latest_gdelt_summary(asset_symbol):
    pattern = f"{asset_symbol}_gdelt_sentiment_summary_{TIMEFRAME}_*.csv"
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    if not candidates:
        return {}

    latest_path = candidates[-1]
    gdelt_df = pd.read_csv(latest_path)
    if gdelt_df.empty:
        return {}
    return gdelt_df.iloc[-1].to_dict()


def _load_context_reliability_map():
    path = get_strategy_summary_path("context_reliability", TIMEFRAME)
    if not Path(path).exists():
        return {}

    reliability_df = pd.read_csv(path)
    if reliability_df.empty:
        return {}
    return {
        str(row["asset_symbol"]): row.to_dict()
        for _, row in reliability_df.iterrows()
    }


def _classify_daily_posture(confirmation_score, onchain_available):
    if not onchain_available:
        return "market_only_research"
    if confirmation_score >= 0.30:
        return "structural_supportive"
    if confirmation_score <= -0.20:
        return "structural_caution"
    return "structural_mixed"


def _build_usage_note(posture_label):
    if posture_label == "structural_supportive":
        return "Use this as higher-timeframe confirmation for 4h and selective 1h strategies, not as a standalone execution trigger."
    if posture_label == "structural_caution":
        return "Use this as a veto or trust-reduction layer for faster strategies until structural conditions improve."
    if posture_label == "market_only_research":
        return "Use this only as daily market-first research because on-chain confirmation is unavailable."
    return "Use this as slow structural context and confirmation rather than a primary live trading decision."


def _adjust_usage_note_for_context(usage_note, context_row):
    context_row = context_row or {}
    combined_readiness = str(context_row.get("combined_context_readiness", "limited_context_confirmation") or "limited_context_confirmation")
    effective_sentiment_role = str(context_row.get("effective_sentiment_role", "unavailable") or "unavailable")

    if combined_readiness == "usable_context_confirmation":
        return usage_note + " Context quality is strong enough to support daily confirmation use."
    if combined_readiness == "conditional_context_confirmation":
        return usage_note + " Context quality is only conditional, so faster strategies should still defer to market and futures structure first."
    if effective_sentiment_role == "broad_market_fallback":
        return usage_note + " Asset-specific context is thin, so broad market mood should stay a light fallback only."
    return usage_note + " Context quality is limited, so treat this mainly as report-facing structural guidance."


def _score_sentiment(gdelt_label, article_count):
    article_count = _safe_float(article_count)
    if gdelt_label in {"bullish", "supportive"}:
        return 0.10 if article_count >= 3 else 0.05
    if gdelt_label in {"bearish", "weakening"}:
        return -0.10 if article_count >= 3 else -0.05
    return 0.0


def _score_strategy_signal(latest_signal, macro_f1):
    macro_f1 = _safe_float(macro_f1)
    confidence_weight = min(max(macro_f1, 0.0), 0.50)
    if latest_signal == "buy":
        return 0.20 + confidence_weight * 0.40
    if latest_signal == "dont_buy":
        return -(0.20 + confidence_weight * 0.40)
    return 0.0


def _score_market_onchain_alignment(combined_view, onchain_regime_score):
    onchain_regime_score = _safe_float(onchain_regime_score)
    if combined_view == "aligned_bullish":
        return 0.20 + min(max(onchain_regime_score, 0.0), 1.0) * 0.15
    if combined_view == "aligned_bearish":
        return -(0.20 + min(max(abs(onchain_regime_score), 0.0), 1.0) * 0.15)
    if combined_view == "market_only":
        return 0.0
    return onchain_regime_score * 0.15


def _score_structural_alignment(overview_row):
    alignment_score = _safe_float(overview_row.get("latest_market_onchain_alignment_score"))
    divergence_score = _safe_float(overview_row.get("latest_market_onchain_divergence_score"))
    confidence_score = _safe_float(overview_row.get("latest_onchain_confidence_score"))
    structural_label = str(overview_row.get("latest_market_onchain_structural_label", "mixed") or "mixed")

    base_score = alignment_score * 0.30
    if structural_label in {"distribution_risk", "divergence_watch"}:
        base_score -= divergence_score * 0.25
    elif structural_label in {"bullish_alignment", "bearish_alignment"}:
        base_score += confidence_score * 0.10
    return base_score


def _build_detail_rows(asset_symbol, best_strategy, overview_row, gdelt_row):
    latest_signal = str(best_strategy.get("latest_signal", "hold") or "hold")
    macro_f1 = _safe_float(best_strategy.get("test_macro_f1"))
    accuracy = _safe_float(best_strategy.get("test_accuracy"))
    combined_view = str(overview_row.get("combined_view", "mixed") or "mixed")
    onchain_regime_score = _safe_float(overview_row.get("latest_onchain_regime_score"))
    market_return_24h_pct = _safe_float(overview_row.get("latest_market_return_24h_pct"))
    gdelt_label = str(gdelt_row.get("gdelt_regime_label", "unavailable") or "unavailable")
    gdelt_articles = _safe_float(gdelt_row.get("gdelt_article_count"))

    detail_rows = [
        {
            "asset_symbol": asset_symbol,
            "timeframe": TIMEFRAME,
            "component": "daily_market_onchain_strategy",
            "component_signal": latest_signal,
            "component_score": _score_strategy_signal(latest_signal, macro_f1),
            "component_strength": macro_f1,
            "support_value": accuracy,
            "support_note": f"best daily specialist is {best_strategy.get('strategy_name', 'n/a')}",
        },
        {
            "asset_symbol": asset_symbol,
            "timeframe": TIMEFRAME,
            "component": "market_onchain_alignment",
            "component_signal": combined_view,
            "component_score": _score_market_onchain_alignment(combined_view, onchain_regime_score),
            "component_strength": onchain_regime_score,
            "support_value": market_return_24h_pct,
            "support_note": "combined market and on-chain daily alignment",
        },
        {
            "asset_symbol": asset_symbol,
            "timeframe": TIMEFRAME,
            "component": "structural_alignment_quality",
            "component_signal": str(overview_row.get("latest_market_onchain_structural_label", "mixed") or "mixed"),
            "component_score": _score_structural_alignment(overview_row),
            "component_strength": _safe_float(overview_row.get("latest_onchain_confidence_score")),
            "support_value": _safe_float(overview_row.get("latest_market_onchain_divergence_score")),
            "support_note": (
                "daily alignment, divergence, confidence quality, and driver mix: "
                f"{str(overview_row.get('latest_market_onchain_driver_summary', 'n/a') or 'n/a')}"
            ),
        },
        {
            "asset_symbol": asset_symbol,
            "timeframe": TIMEFRAME,
            "component": "asset_news_sentiment",
            "component_signal": gdelt_label,
            "component_score": _score_sentiment(gdelt_label, gdelt_articles),
            "component_strength": gdelt_articles,
            "support_value": _safe_float(gdelt_row.get("gdelt_sentiment_mean")),
            "support_note": "daily GDELT asset-news context",
        },
    ]
    return detail_rows


def _build_default_strategy_stub():
    return {
        "strategy_name": "market_onchain_specialist_unavailable",
        "latest_signal": "hold",
        "test_macro_f1": 0.0,
        "test_accuracy": 0.0,
    }


def _build_default_overlay_stub():
    return {
        "strategy_name": "onchain_structural_overlay_unavailable",
        "latest_signal": "hold",
        "latest_overlay_mode": "structural_mixed",
        "latest_overlay_risk_score": 0.0,
        "latest_overlay_support_score": 0.0,
        "test_macro_f1": 0.0,
        "test_accuracy": 0.0,
    }


def evaluate_daily_structural_confirmation():
    """Generate a daily structural confirmation family summary for report/app use."""
    overview_df = _load_market_onchain_overview()
    strategy_df = _load_market_onchain_strategy_summary()
    overlay_df = _load_onchain_overlay_summary()
    context_reliability_map = _load_context_reliability_map()

    summary_rows = []
    detail_rows = []

    for _, overview_row in overview_df.iterrows():
        asset_symbol = str(overview_row.get("asset_symbol", "") or "").strip()
        if not asset_symbol:
            continue

        asset_strategy_df = strategy_df[strategy_df["asset_symbol"] == asset_symbol].copy()
        if asset_strategy_df.empty:
            best_strategy = _build_default_strategy_stub()
        else:
            best_strategy = (
                asset_strategy_df.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
                .iloc[0]
                .to_dict()
            )
        asset_overlay_df = overlay_df[overlay_df["asset_symbol"] == asset_symbol].copy() if not overlay_df.empty else pd.DataFrame()
        if asset_overlay_df.empty:
            best_overlay = _build_default_overlay_stub()
        else:
            best_overlay = (
                asset_overlay_df.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
                .iloc[0]
                .to_dict()
            )
        gdelt_row = _load_latest_gdelt_summary(asset_symbol)
        context_row = context_reliability_map.get(asset_symbol, {})
        detail_component_rows = _build_detail_rows(asset_symbol, best_strategy, overview_row, gdelt_row)
        detail_component_rows.append(
            {
                "asset_symbol": asset_symbol,
                "timeframe": TIMEFRAME,
                "component": "onchain_structural_overlay",
                "component_signal": best_overlay.get("latest_overlay_mode", "structural_mixed"),
                "component_score": (
                    _safe_float(best_overlay.get("latest_overlay_support_score")) -
                    _safe_float(best_overlay.get("latest_overlay_risk_score"))
                ) * 0.20,
                "component_strength": max(
                    _safe_float(best_overlay.get("latest_overlay_support_score")),
                    _safe_float(best_overlay.get("latest_overlay_risk_score")),
                ),
                "support_value": _safe_float(best_overlay.get("test_macro_f1")),
                "support_note": f"best on-chain overlay is {best_overlay.get('strategy_name', 'n/a')}",
            }
        )
        detail_rows.extend(detail_component_rows)

        confirmation_score = sum(_safe_float(row["component_score"]) for row in detail_component_rows)
        onchain_available = bool(overview_row.get("onchain_data_available", False))
        posture_label = _classify_daily_posture(confirmation_score, onchain_available)
        gdelt_label = str(gdelt_row.get("gdelt_regime_label", "unavailable") or "unavailable")
        gdelt_articles = int(_safe_float(gdelt_row.get("gdelt_article_count", 0)))

        summary_rows.append(
            {
                "asset_symbol": asset_symbol,
                "market_symbol": overview_row.get("market_symbol"),
                "timeframe": TIMEFRAME,
                "latest_window_end": overview_row.get("latest_window_end"),
                "best_daily_specialist": best_strategy.get("strategy_name"),
                "best_daily_signal": best_strategy.get("latest_signal"),
                "daily_specialist_accuracy": _safe_float(best_strategy.get("test_accuracy")),
                "daily_specialist_macro_f1": _safe_float(best_strategy.get("test_macro_f1")),
                "best_onchain_overlay": best_overlay.get("strategy_name"),
                "best_onchain_overlay_mode": best_overlay.get("latest_overlay_mode"),
                "best_onchain_overlay_accuracy": _safe_float(best_overlay.get("test_accuracy")),
                "best_onchain_overlay_macro_f1": _safe_float(best_overlay.get("test_macro_f1")),
                "latest_onchain_overlay_risk_score": _safe_float(best_overlay.get("latest_overlay_risk_score")),
                "latest_onchain_overlay_support_score": _safe_float(best_overlay.get("latest_overlay_support_score")),
                "latest_onchain_overlay_confidence": _safe_float(best_overlay.get("latest_overlay_confidence")),
                "latest_market_return_24h_pct": _safe_float(overview_row.get("latest_market_return_24h_pct")),
                "latest_market_volatility_20": _safe_float(overview_row.get("latest_market_volatility_20")),
                "latest_onchain_regime_label": overview_row.get("latest_onchain_regime_label"),
                "latest_onchain_regime_score": _safe_float(overview_row.get("latest_onchain_regime_score")),
                "latest_onchain_confidence_score": _safe_float(overview_row.get("latest_onchain_confidence_score")),
                "latest_onchain_participation_breadth_score": _safe_float(overview_row.get("latest_onchain_participation_breadth_score")),
                "latest_onchain_structural_fragility_score": _safe_float(overview_row.get("latest_onchain_structural_fragility_score")),
                "latest_market_onchain_alignment_score": _safe_float(overview_row.get("latest_market_onchain_alignment_score")),
                "latest_market_onchain_divergence_score": _safe_float(overview_row.get("latest_market_onchain_divergence_score")),
                "latest_market_onchain_structural_label": overview_row.get("latest_market_onchain_structural_label"),
                "latest_onchain_primary_support_driver": overview_row.get("latest_onchain_primary_support_driver", "none"),
                "latest_onchain_primary_risk_driver": overview_row.get("latest_onchain_primary_risk_driver", "none"),
                "combined_daily_view": overview_row.get("combined_view"),
                "asset_news_regime_label": gdelt_label,
                "asset_news_article_count": gdelt_articles,
                "effective_sentiment_role": context_row.get("effective_sentiment_role", "unavailable"),
                "gdelt_reliability_label": context_row.get("gdelt_reliability_label", "unavailable"),
                "onchain_reliability_label": context_row.get("onchain_reliability_label", "unavailable"),
                "combined_context_readiness": context_row.get("combined_context_readiness", "limited_context_confirmation"),
                "onchain_data_available": onchain_available,
                "daily_confirmation_score": confirmation_score,
                "daily_posture_label": posture_label,
                "deployment_tier": "research_only",
                "recommended_use": _adjust_usage_note_for_context(_build_usage_note(posture_label), context_row),
                "daily_structural_summary": (
                    f"{asset_symbol} daily structural confirmation is {posture_label.replace('_', ' ')}. "
                    f"The strongest daily specialist is {best_strategy.get('strategy_name')} with signal "
                    f"{best_strategy.get('latest_signal')}, on-chain is "
                    f"{str(overview_row.get('latest_onchain_regime_label', 'unavailable')).replace('_', ' ')}, "
                    f"structural label is {str(overview_row.get('latest_market_onchain_structural_label', 'mixed')).replace('_', ' ')}, "
                    f"participation breadth is {_safe_float(overview_row.get('latest_onchain_participation_breadth_score')):.2f}, "
                    f"structural fragility is {_safe_float(overview_row.get('latest_onchain_structural_fragility_score')):.2f}, "
                    f"overlay mode is {str(best_overlay.get('latest_overlay_mode', 'structural_mixed')).replace('_', ' ')}, "
                    f"overlay confidence is {_safe_float(best_overlay.get('latest_overlay_confidence')):.2f}, "
                    f"asset-news sentiment is {gdelt_label.replace('_', ' ')}, and overall context readiness is "
                    f"{str(context_row.get('combined_context_readiness', 'limited_context_confirmation')).replace('_', ' ')}."
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = get_strategy_summary_path(STRATEGY_GROUP, TIMEFRAME)
    detail_path = get_strategy_summary_path(DETAIL_GROUP, TIMEFRAME)
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("daily structural confirmation summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    print("daily structural confirmation detail generated")
    print(f"rows saved: {len(detail_df)}")
    print(f"detail saved to: {detail_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_daily_structural_confirmation()
