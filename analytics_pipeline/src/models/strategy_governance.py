"""Governance helpers for strategy readiness, evaluation strength, and context-layer quality."""


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_layers(layers):
    return ", ".join(layer.replace("_", " ") for layer in layers) if layers else "no extra layers"


FAMILY_LABELS = {
    "promote_market_futures_backbone": "Market + futures backbone",
    "keep_market_only_as_benchmark_lead": "Market-only benchmark",
    "mixed_evidence_keep_both_visible": "Mixed evidence",
}

OPERATIONAL_FAMILY_LABELS = {
    "market_futures_backbone": "Market + futures live default",
    "intraday_binary_candidate": "Intraday binary candidate",
    "binary_directional_fallback": "Binary directional fallback",
    "conditional_market_futures": "Conditional market + futures",
    "market_only_benchmark": "Market-only live default",
    "daily_structural_research": "Daily structural research",
    "reduced_trust_benchmark": "Reduced-trust benchmark fallback",
}

CONTEXT_OVERLAY_LABELS = {
    "onchain_structural_confirmation_lead": "On-chain structural confirmation lead",
    "news_event_veto_lead": "News-event veto lead",
    "cautious_multimodal_overlay": "Cautious multimodal overlay",
    "market_futures_core_keep_context_secondary": "Core-first with secondary context",
}


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _label_text(value, fallback="unavailable"):
    text = str(value or fallback).strip()
    if not text:
        text = fallback
    return text.replace("_", " ")


def _derive_daily_confirmation_mode(summary):
    summary = summary or {}
    daily_posture_label = str(summary.get("daily_posture_label", "structural_mixed") or "structural_mixed")
    daily_structural_label = str(
        summary.get("latest_market_onchain_structural_label", summary.get("combined_daily_view", "mixed"))
        or "mixed"
    )
    daily_structural_confidence = _safe_float(
        summary.get("latest_onchain_confidence_score", summary.get("latest_onchain_overlay_confidence", 0))
    )
    onchain_support_driver = str(summary.get("latest_onchain_primary_support_driver", "none") or "none")
    onchain_risk_driver = str(summary.get("latest_onchain_primary_risk_driver", "none") or "none")
    onchain_participation_breadth = _safe_float(summary.get("latest_onchain_participation_breadth_score", 0))
    onchain_structural_fragility = _safe_float(summary.get("latest_onchain_structural_fragility_score", 0))

    if (
        daily_structural_label in {"distribution_risk", "divergence_watch"}
        or onchain_risk_driver in {"distribution_risk", "structural_fragility", "trend_divergence"}
        or onchain_structural_fragility >= 0.35
    ):
        return "cautionary"
    if (
        daily_posture_label == "structural_supportive"
        and daily_structural_confidence >= 0.55
        and onchain_participation_breadth >= 0.35
    ):
        return "supportive_broad"
    if (
        daily_posture_label == "structural_supportive"
        and daily_structural_confidence >= 0.45
        and onchain_support_driver in {
            "valuation_support",
            "exchange_relief",
            "trend_alignment",
            "network_tailwind",
            "participation_breadth",
        }
    ):
        return "supportive_narrow"
    return "mixed"


def _build_context_overlay_posture(summary):
    summary = summary or {}
    overlay_key = str(
        summary.get("context_overlay_lead", "market_futures_core_keep_context_secondary")
        or "market_futures_core_keep_context_secondary"
    )
    overlay_summary = str(
        summary.get("context_overlay_summary")
        or "No context-overlay comparison has been generated yet."
    )
    news_event_mode = str(summary.get("latest_news_event_mode", "quiet") or "quiet")
    news_event_risk = _safe_float(summary.get("latest_news_event_risk_score"))
    onchain_overlay_mode = str(summary.get("latest_onchain_overlay_mode", "structural_mixed") or "structural_mixed")
    onchain_risk = _safe_float(summary.get("daily_onchain_risk_score"))
    onchain_support = _safe_float(summary.get("daily_onchain_support_score"))
    news_theme = str(summary.get("latest_gdelt_dominant_event_theme", "none") or "none")
    onchain_support_driver = str(summary.get("latest_onchain_primary_support_driver", "none") or "none")
    onchain_risk_driver = str(summary.get("latest_onchain_primary_risk_driver", "none") or "none")

    role = "secondary_context"
    action = "keep_context_secondary"
    headline = "Market + futures should stay in front, with context acting as selective confirmation."

    if overlay_key == "news_event_veto_lead":
        role = "event_veto"
        action = "elevate_news_risk_checks"
        headline = (
            f"News context is currently the most meaningful overlay because event risk is {news_event_mode.replace('_', ' ')} "
            f"with score {news_event_risk:.2f}"
            + (f" and dominant theme {news_theme.replace('_', ' ')}." if news_theme != "none" else ".")
        )
    elif overlay_key == "onchain_structural_confirmation_lead":
        role = "structural_confirmation"
        action = "elevate_onchain_confirmation"
        headline = (
            f"On-chain context is currently the strongest overlay, with {onchain_overlay_mode.replace('_', ' ')} mode "
            f"(risk {onchain_risk:.2f}, support {onchain_support:.2f}). "
            f"Primary support driver is {_label_text(onchain_support_driver, 'none')}, "
            f"primary risk driver is {_label_text(onchain_risk_driver, 'none')}."
        )
    elif overlay_key == "cautious_multimodal_overlay":
        role = "cautious_overlay"
        action = "use_combined_context_cautiously"
        headline = (
            "Combined context is currently the strongest overlay route, but it should still be used cautiously as a "
            "trust modifier rather than a primary predictor."
        )

    return {
        "overlay_key": overlay_key,
        "overlay_label": CONTEXT_OVERLAY_LABELS.get(overlay_key, "Core-first with secondary context"),
        "overlay_role": role,
        "suggested_action": action,
        "headline": headline,
        "summary": overlay_summary,
    }


def _derive_futures_support_mode(summary):
    """Classify how trustworthy the current futures support is."""
    summary = summary or {}
    resilience_label = str(summary.get("futures_context_resilience_label", "unavailable") or "unavailable")
    completeness_label = str(summary.get("futures_completeness_label", "unavailable") or "unavailable")
    basis_feature_available = _safe_bool(summary.get("basis_feature_available"))
    resilience_score = _safe_float(summary.get("futures_context_resilience_score", 0))

    if resilience_label == "unavailable":
        return "unavailable"
    if resilience_label == "fragile" or resilience_score < 0.45:
        return "fragile"
    if completeness_label == "partial" or not basis_feature_available:
        return "robust_partial"
    if resilience_label == "robust" and completeness_label == "full":
        return "robust_full"
    return "partial"


def _derive_operational_family(summary, recommended_family):
    summary = summary or {}
    timeframe = str(summary.get("timeframe", "4h") or "4h").strip().lower()
    selected_backbone_family = str(summary.get("backbone_best_family", "three_class") or "three_class")
    deployment_active = _safe_bool(summary.get("backbone_deployment_active"))
    policy_excess = _safe_float(summary.get("backbone_excess_return"))
    binary_long_f1 = _safe_float(summary.get("backbone_walkforward_avg_long_f1"))
    structural_break_status = str(summary.get("structural_break_status", "stable") or "stable")
    trust_mode = str(summary.get("trust_mode", "normal") or "normal")
    futures_support_mode = _derive_futures_support_mode(summary)
    concentration_flag = str(
        summary.get("ternary_backbone_prediction_concentration_flag", "stable") or "stable"
    )

    fallback_trigger = "none"
    operational_family = "market_only_benchmark"
    operational_summary = "The market-only benchmark is the safest current default while evidence is mixed."

    if timeframe == "1d":
        fallback_trigger = "daily_research_mode"
        operational_family = "daily_structural_research"
        operational_summary = (
            "Daily outputs are useful as a longer-term check, not as the main live trading signal."
        )
    elif structural_break_status == "break" or trust_mode == "reduced_trust":
        fallback_trigger = "structural_break"
        operational_family = "reduced_trust_benchmark"
        operational_summary = (
            "A recent market shift is reducing trust in older patterns, so LiveStrat should use the market-only "
            "benchmark until fresh evidence improves."
        )
    elif futures_support_mode == "fragile":
        fallback_trigger = "fragile_futures_support"
        operational_family = "market_only_benchmark"
        operational_summary = (
            "Current futures data support is too weak to make market + futures the live default."
        )
    elif (
        timeframe == "1h"
        and selected_backbone_family == "binary_directional"
        and deployment_active
        and policy_excess > 0
        and binary_long_f1 > 0.20
    ):
        operational_family = "intraday_binary_candidate"
        operational_summary = (
            "Short-term conditions currently favor a simpler long-or-flat model over the three-class model."
        )
    elif selected_backbone_family == "three_class" and concentration_flag in {"hold_collapse", "dont_buy_collapse"}:
        fallback_trigger = concentration_flag
        if binary_long_f1 > 0.20:
            operational_family = "binary_directional_fallback"
            operational_summary = (
                "The three-class model is becoming one-sided, so the simpler long-or-flat model is cleaner right now."
            )
        else:
            operational_family = "market_only_benchmark"
            operational_summary = (
                "The three-class model is becoming one-sided, and the simpler model is not strong enough to replace "
                "the benchmark yet."
            )
    elif recommended_family == "promote_market_futures_backbone" and deployment_active and policy_excess > 0:
        operational_family = "market_futures_backbone"
        operational_summary = (
            "The market + futures backbone has both evaluation support and a live policy path that is still adding value."
        )
        if futures_support_mode == "robust_partial":
            fallback_trigger = "partial_futures_support"
            operational_family = "conditional_market_futures"
            operational_summary = (
                "The market + futures model is still the best live route, but futures coverage is only partial right now."
            )
    elif selected_backbone_family == "binary_directional" and deployment_active and policy_excess > 0 and binary_long_f1 > 0.20:
        operational_family = "conditional_market_futures"
        operational_summary = (
            "The simpler market + futures path is active and positive, but needs more windows before it becomes the "
            "main default."
        )
    elif recommended_family == "keep_market_only_as_benchmark_lead":
        operational_family = "market_only_benchmark"
        operational_summary = (
            "The market-only benchmark currently has cleaner evidence and should remain the live default."
        )
    elif selected_backbone_family == "binary_directional":
        operational_family = "market_only_benchmark"
        fallback_trigger = "binary_not_deployable" if fallback_trigger == "none" else fallback_trigger
        operational_summary = (
            "The simpler directional model is promising, but not strong enough to replace the benchmark yet."
        )
    elif recommended_family == "mixed_evidence_keep_both_visible":
        operational_family = "market_only_benchmark"
        operational_summary = (
            "Evidence remains mixed, so the benchmark should stay the live default while market + futures remains visible."
        )

    return {
        "timeframe": timeframe,
        "operational_family": operational_family,
        "operational_family_label": OPERATIONAL_FAMILY_LABELS.get(operational_family, "Market-only live default"),
        "fallback_trigger": fallback_trigger,
        "operational_summary": operational_summary,
        "selected_backbone_family": selected_backbone_family,
        "deployment_active": deployment_active,
        "policy_excess_return": policy_excess,
        "concentration_flag": concentration_flag,
        "structural_break_status": structural_break_status,
        "trust_mode": trust_mode,
        "futures_support_mode": futures_support_mode,
    }


def build_family_governance(summary, strategy_key="market_futures_backend"):
    """Summarize which family currently deserves lead status for this asset."""
    summary = summary or {}
    recommended_family = str(summary.get("recommended_family", "mixed_evidence_keep_both_visible") or "mixed_evidence_keep_both_visible")
    lead_family_label = FAMILY_LABELS.get(recommended_family, "Mixed evidence")
    operational_state = _derive_operational_family(summary, recommended_family)
    evidence_summary = str(
        summary.get("comparison_summary")
        or "No asset-aware family comparison summary has been generated yet."
    )
    futures_support_mode = operational_state.get("futures_support_mode", "unavailable")
    pipeline_freshness_label = str(summary.get("pipeline_freshness_label", "unknown") or "unknown")
    pipeline_window_age_days = _safe_float(summary.get("pipeline_window_age_days", 0))
    if futures_support_mode == "robust_partial":
        evidence_summary += " Futures context is robust overall, but current sublayer coverage is only partial."
    elif futures_support_mode == "fragile":
        evidence_summary += " Futures context is currently fragile, so benchmark caution should increase."
    if pipeline_freshness_label == "stale":
        evidence_summary += (
            f" The current evaluation window is stale ({pipeline_window_age_days:.0f} days old), so live confidence should stay softer until the pipeline refreshes."
        )
    elif pipeline_freshness_label == "recent":
        evidence_summary += " The current evaluation window is reasonably recent, but not fully current yet."

    if strategy_key == "market_futures_backend":
        if operational_state["operational_family"] in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}:
            alignment = "secondary_for_asset"
        elif operational_state["operational_family"] in {"intraday_binary_candidate", "binary_directional_fallback"}:
            alignment = "conditional_for_asset"
        elif recommended_family == "promote_market_futures_backbone":
            alignment = "lead_for_asset"
        elif recommended_family == "keep_market_only_as_benchmark_lead":
            alignment = "secondary_for_asset"
        else:
            alignment = "conditional_for_asset"
    elif strategy_key in {"scaled_model", "rule_based", "unscaled_model"}:
        if recommended_family == "keep_market_only_as_benchmark_lead":
            alignment = "lead_for_asset" if strategy_key == "scaled_model" else "supporting_for_asset"
        elif recommended_family == "promote_market_futures_backbone":
            alignment = "benchmark_only_for_asset"
        else:
            alignment = "conditional_for_asset"
    else:
        alignment = "conditional_for_asset"

    return {
        "recommended_family": recommended_family,
        "lead_family_label": lead_family_label,
        "strategy_alignment": alignment,
        "evidence_summary": evidence_summary,
        "operational_family": operational_state["operational_family"],
        "operational_family_label": operational_state["operational_family_label"],
        "operational_summary": operational_state["operational_summary"],
        "fallback_trigger": operational_state["fallback_trigger"],
        "selected_backbone_family": operational_state["selected_backbone_family"],
        "deployment_active": operational_state["deployment_active"],
        "policy_excess_return": operational_state["policy_excess_return"],
        "concentration_flag": operational_state["concentration_flag"],
        "structural_break_status": operational_state["structural_break_status"],
        "trust_mode": operational_state["trust_mode"],
        "futures_support_mode": futures_support_mode,
    }


def build_context_layer_assessment(summary, runtime_support=None):
    """Assess the actual strength and role of sentiment and on-chain layers."""
    summary = summary or {}
    runtime_support = runtime_support or {}

    sentiment_source = str(summary.get("latest_effective_sentiment_source", "unavailable") or "unavailable")
    sentiment_label = str(summary.get("latest_effective_sentiment_label", "unavailable") or "unavailable")
    gdelt_label = str(summary.get("latest_gdelt_regime_label", "unavailable") or "unavailable")
    gdelt_count = _safe_float(summary.get("latest_gdelt_article_count", 0))
    onchain_label = str(summary.get("latest_onchain_regime_label", "unavailable") or "unavailable")
    gdelt_reliability_label = str(summary.get("gdelt_reliability_label", "unavailable") or "unavailable")
    gdelt_reliability_score = _safe_float(summary.get("gdelt_reliability_score", 0))
    gdelt_freshness_status = str(summary.get("gdelt_freshness_status", summary.get("latest_gdelt_snapshot_status", "unavailable")) or "unavailable")
    gdelt_age_days = _safe_float(summary.get("gdelt_age_days", summary.get("latest_gdelt_snapshot_age_days", 0)))
    gdelt_dominant_theme = str(summary.get("latest_gdelt_dominant_event_theme", summary.get("gdelt_dominant_event_theme", "none")) or "none")
    gdelt_risk_theme = str(summary.get("latest_gdelt_risk_event_theme", summary.get("gdelt_risk_event_theme", "none")) or "none")
    gdelt_supportive_theme = str(summary.get("latest_gdelt_supportive_event_theme", summary.get("gdelt_supportive_event_theme", "none")) or "none")
    news_event_label = str(summary.get("latest_news_event_regime_label", "quiet") or "quiet")
    news_event_risk_score = _safe_float(summary.get("latest_news_event_risk_score", 0))
    broad_sentiment_reliability_label = str(
        summary.get("broad_sentiment_reliability_label", "unavailable") or "unavailable"
    )
    effective_sentiment_role = str(summary.get("effective_sentiment_role", "unavailable") or "unavailable")
    effective_sentiment_reliability_score = _safe_float(summary.get("effective_sentiment_reliability_score", 0))
    onchain_reliability_label = str(summary.get("onchain_reliability_label", "unavailable") or "unavailable")
    onchain_reliability_score = _safe_float(summary.get("onchain_reliability_score", 0))
    combined_context_readiness = str(summary.get("combined_context_readiness", "limited_context_confirmation") or "limited_context_confirmation")
    onchain_snapshot_status = str(summary.get("latest_onchain_snapshot_status", "unavailable") or "unavailable")
    onchain_snapshot_label = str(summary.get("latest_onchain_snapshot_label", "unavailable") or "unavailable")
    onchain_snapshot_age_days = _safe_float(summary.get("latest_onchain_snapshot_age_days", 0))
    daily_structural_label = str(
        summary.get("latest_market_onchain_structural_label", summary.get("combined_daily_view", "mixed"))
        or "mixed"
    )
    daily_structural_confidence = _safe_float(
        summary.get("latest_onchain_confidence_score", summary.get("latest_onchain_overlay_confidence", 0))
    )
    onchain_participation_breadth = _safe_float(summary.get("latest_onchain_participation_breadth_score", 0))
    onchain_structural_fragility = _safe_float(summary.get("latest_onchain_structural_fragility_score", 0))
    onchain_support_driver = str(summary.get("latest_onchain_primary_support_driver", "none") or "none")
    onchain_risk_driver = str(summary.get("latest_onchain_primary_risk_driver", "none") or "none")
    sentiment_timeframes = runtime_support.get("sentiment_timeframes", [])
    onchain_timeframes = runtime_support.get("onchain_timeframes", [])

    broad_sentiment = {
        "layer": "broad_sentiment",
        "role": "fallback_confirmation",
        "status": "available" if sentiment_source == "fear_greed_market_fallback" else "secondary",
        "readiness": "fallback_only" if broad_sentiment_reliability_label == "broad_fallback" else "supporting",
        "available_timeframes": sentiment_timeframes,
        "reliability_label": broad_sentiment_reliability_label,
        "reliability_score": effective_sentiment_reliability_score if effective_sentiment_role == "broad_market_fallback" else 0.0,
        "headline": (
            "Broad market sentiment is active as fallback confirmation."
            if sentiment_source == "fear_greed_market_fallback"
            else "Broad market sentiment is not the active effective layer for this asset."
        ),
    }

    if gdelt_label != "unavailable":
        asset_news = {
            "layer": "asset_news_sentiment",
            "role": "conditional_confirmation",
            "status": "available",
            "readiness": "conditional" if gdelt_reliability_label in {"thin_confirmation", "conditional_confirmation"} else "partial",
            "available_timeframes": sentiment_timeframes,
            "reliability_label": gdelt_reliability_label,
            "reliability_score": gdelt_reliability_score,
            "freshness_status": gdelt_freshness_status,
            "age_days": gdelt_age_days,
            "dominant_theme": gdelt_dominant_theme,
            "risk_theme": gdelt_risk_theme,
            "supportive_theme": gdelt_supportive_theme,
            "headline": (
                f"Asset-specific news sentiment is {gdelt_label.replace('_', ' ')} with {gdelt_count:.0f} recent articles, "
                f"current reliability is {gdelt_reliability_label.replace('_', ' ')}, and freshness is "
                f"{gdelt_freshness_status.replace('_', ' ')}. Event mode is "
                f"{news_event_label.replace('_', ' ')} with risk score {news_event_risk_score:.2f}. "
                f"Dominant theme is {_label_text(gdelt_dominant_theme, 'none')}, risk theme is "
                f"{_label_text(gdelt_risk_theme, 'none')}, and supportive theme is "
                f"{_label_text(gdelt_supportive_theme, 'none')}."
            ),
        }
    else:
        asset_news = {
            "layer": "asset_news_sentiment",
            "role": "conditional_confirmation",
            "status": "unavailable",
            "readiness": "limited",
            "available_timeframes": sentiment_timeframes,
            "reliability_label": gdelt_reliability_label,
            "reliability_score": gdelt_reliability_score,
            "freshness_status": gdelt_freshness_status,
            "age_days": gdelt_age_days,
            "dominant_theme": gdelt_dominant_theme,
            "risk_theme": gdelt_risk_theme,
            "supportive_theme": gdelt_supportive_theme,
            "headline": "Asset-specific news sentiment is unavailable, so this layer should not drive a decision.",
        }

    if onchain_label != "unavailable":
        onchain = {
            "layer": "onchain_daily",
            "role": "structural_confirmation",
            "status": "available",
            "readiness": "partial" if onchain_reliability_label == "usable_structural_confirmation" else "conditional",
            "available_timeframes": onchain_timeframes,
            "reliability_label": onchain_reliability_label,
            "reliability_score": onchain_reliability_score,
            "primary_support_driver": onchain_support_driver,
            "primary_risk_driver": onchain_risk_driver,
            "participation_breadth_score": onchain_participation_breadth,
            "structural_fragility_score": onchain_structural_fragility,
            "headline": (
                f"On-chain context is {onchain_label.replace('_', ' ')} with "
                f"{onchain_reliability_label.replace('_', ' ')} reliability. "
                f"Latest structural label is {daily_structural_label.replace('_', ' ')} "
                f"with confidence {daily_structural_confidence:.2f}. "
                f"Participation breadth is {onchain_participation_breadth:.2f} and fragility is "
                f"{onchain_structural_fragility:.2f}. "
                f"Primary support driver is {_label_text(onchain_support_driver, 'none')} and primary risk driver is "
                f"{_label_text(onchain_risk_driver, 'none')}, so it is best treated as a longer-term check."
            ),
        }
    elif onchain_snapshot_status == "stale":
        onchain = {
            "layer": "onchain_daily",
            "role": "structural_confirmation",
            "status": "stale",
            "readiness": "limited",
            "available_timeframes": onchain_timeframes,
            "reliability_label": onchain_reliability_label,
            "reliability_score": onchain_reliability_score,
            "primary_support_driver": onchain_support_driver,
            "primary_risk_driver": onchain_risk_driver,
            "participation_breadth_score": onchain_participation_breadth,
            "structural_fragility_score": onchain_structural_fragility,
            "headline": (
                f"The latest on-chain snapshot is stale ({onchain_snapshot_label.replace('_', ' ')}, {onchain_snapshot_age_days:.0f} days old)."
            ),
        }
    else:
        onchain = {
            "layer": "onchain_daily",
            "role": "structural_confirmation",
            "status": "unavailable",
            "readiness": "limited",
            "available_timeframes": onchain_timeframes,
            "reliability_label": onchain_reliability_label,
            "reliability_score": onchain_reliability_score,
            "primary_support_driver": onchain_support_driver,
            "primary_risk_driver": onchain_risk_driver,
            "participation_breadth_score": onchain_participation_breadth,
            "structural_fragility_score": onchain_structural_fragility,
            "headline": "On-chain context is unavailable, so LiveStrat should not rely on it for this asset right now.",
        }

    context_overlay_posture = _build_context_overlay_posture(summary)

    return {
        "combined_context_readiness": combined_context_readiness,
        "context_reliability_summary": summary.get("context_reliability_summary", ""),
        "context_overlay_posture": context_overlay_posture,
        "broad_sentiment": broad_sentiment,
        "asset_news_sentiment": asset_news,
        "onchain_daily": onchain,
    }


def _derive_evaluation_strength(accuracy, macro_f1, walkforward_folds, exact_timeframe_match):
    if accuracy <= 0 and macro_f1 <= 0:
        return (
            "limited",
            "Current evaluation is too thin to treat this as a fully trusted strategy result.",
        )

    if walkforward_folds >= 3 and exact_timeframe_match and accuracy >= 0.30:
        if macro_f1 >= 0.20:
            return (
                "moderate",
                "This strategy has a walk-forward style evaluation path and an exact timeframe match, which makes it the strongest current basis for user-facing decisions.",
            )
        return (
            "moderate",
            "This strategy has exact-timeframe walk-forward coverage, but class balance quality still needs improvement.",
        )

    if accuracy > 0:
        return (
            "provisional",
            "This strategy has some benchmark evidence, but either timeframe integrity or evaluation depth is still incomplete.",
        )

    return (
        "limited",
        "This strategy currently behaves more like a scaffold than a validated decision engine.",
    )


def build_strategy_governance(strategy_key, summary, requested_timeframe, resolved_timeframe, runtime_status=None, capabilities=None):
    """Assess whether a strategy should be treated as ready, partial, research, or experimental."""
    summary = summary or {}
    runtime_status = runtime_status or {}
    capabilities = capabilities or {}

    accuracy = _safe_float(summary.get("test_accuracy", summary.get("baseline_scaled_test_accuracy", 0)))
    macro_f1 = _safe_float(summary.get("test_macro_f1", 0))
    walkforward_folds = _safe_float(summary.get("walkforward_fold_count", 0))
    exact_timeframe_match = resolved_timeframe == requested_timeframe
    daily_posture_label = str(summary.get("daily_posture_label", "structural_mixed") or "structural_mixed")
    daily_structural_label = str(
        summary.get("latest_market_onchain_structural_label", summary.get("combined_daily_view", "mixed"))
        or "mixed"
    )
    daily_structural_confidence = _safe_float(
        summary.get("latest_onchain_confidence_score", summary.get("latest_onchain_overlay_confidence", 0))
    )
    daily_confirmation_mode = _derive_daily_confirmation_mode(summary)

    runtime_label = str(runtime_status.get("status", "unavailable") or "unavailable")
    runtime_role = str(runtime_status.get("role", "n/a") or "n/a")
    family_governance = build_family_governance(summary, strategy_key)
    futures_support_mode = family_governance.get("futures_support_mode", "unavailable")
    pipeline_freshness_label = str(summary.get("pipeline_freshness_label", "unknown") or "unknown")
    pipeline_refresh_status = str(summary.get("pipeline_refresh_status", "missing") or "missing")
    pipeline_window_age_days = _safe_float(summary.get("pipeline_window_age_days", 0))
    evaluation_strength, evaluation_reason = _derive_evaluation_strength(
        accuracy,
        macro_f1,
        walkforward_folds,
        exact_timeframe_match,
    )

    if runtime_label == "unavailable":
        readiness_label = "Unavailable"
        deployment_tier = "not_supported"
    elif runtime_label == "experimental":
        readiness_label = "Experimental"
        deployment_tier = "research_only"
    elif runtime_label == "research_only":
        readiness_label = "Research"
        deployment_tier = "research_only"
    elif not exact_timeframe_match:
        readiness_label = "Fallback"
        deployment_tier = "fallback_only"
    elif runtime_label == "conditional":
        readiness_label = "Conditional"
        deployment_tier = "conditional_support"
    elif strategy_key in {"rule_based", "scaled_model"}:
        readiness_label = "Benchmark Ready"
        deployment_tier = "benchmark"
    else:
        readiness_label = "Ready"
        deployment_tier = "production_candidate"

    if strategy_key == "market_futures_backend":
        if futures_support_mode == "robust_partial" and readiness_label == "Ready":
            readiness_label = "Conditional"
            deployment_tier = "conditional_support"
            evaluation_reason += " Futures support is robust overall, but current sublayer coverage is only partial."
        elif futures_support_mode == "fragile" and readiness_label in {"Ready", "Conditional"}:
            readiness_label = "Fallback"
            deployment_tier = "fallback_only"
            evaluation_reason += " Futures support is currently fragile, so benchmark-style fallback should stay available."

    risk_flags = []
    if not exact_timeframe_match:
        risk_flags.append("Requested timeframe falls back to another generated timeframe.")
    if runtime_label in {"experimental", "research_only", "conditional"}:
        risk_flags.append(f"Runtime status is {runtime_label.replace('_', ' ')}, so this strategy should not be treated as fully production-ready.")
    if macro_f1 <= 0.10 and accuracy > 0:
        risk_flags.append("Macro-F1 remains weak, so class balance quality still needs improvement.")
    if walkforward_folds < 3:
        risk_flags.append("Walk-forward coverage is still shallow.")

    if strategy_key == "market_futures_backend":
        if family_governance["operational_family"] in {"market_only_benchmark", "reduced_trust_benchmark"}:
            recommended_use = "Keep this visible for comparison, but let the market-only benchmark lead until model evidence improves."
        elif family_governance["operational_family"] == "daily_structural_research":
            recommended_use = "Use this as a longer-term research check, not as the primary live trading family."
        elif family_governance["operational_family"] == "intraday_binary_candidate":
            recommended_use = "Use the simpler directional model for short-term checks while keeping the benchmark available."
        elif family_governance["operational_family"] == "binary_directional_fallback":
            recommended_use = "Use the simpler long-or-flat model while the three-class model is unstable."
        else:
            recommended_use = "Primary candidate for user-facing strategy decisions while multi-timeframe coverage is being completed."
        if futures_support_mode == "robust_partial":
            recommended_use += " Futures data is usable, but coverage is partial, so confidence should stay measured."
        elif futures_support_mode == "fragile":
            recommended_use += " Current futures support is fragile, so this route should defer to the benchmark until coverage improves."
        if resolved_timeframe == "4h" and daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.50:
            recommended_use += " The daily check is supportive enough to increase trust slightly."
        elif resolved_timeframe == "4h" and daily_structural_label in {"distribution_risk", "divergence_watch"}:
            recommended_use += " The daily check is showing risk, so short-term signals should stay cautious."
        if resolved_timeframe == "4h" and daily_confirmation_mode == "supportive_narrow":
            recommended_use += " Higher-timeframe support is present but narrow."
        elif resolved_timeframe == "4h" and daily_confirmation_mode == "supportive_broad":
            recommended_use += " Higher-timeframe support is broad enough to make the 4h view more trustworthy."
    elif strategy_key == "rule_based":
        recommended_use = "Transparent benchmark for explanation and comparison, not the main deployed strategy."
    elif strategy_key == "scaled_model":
        if family_governance["recommended_family"] == "keep_market_only_as_benchmark_lead":
            recommended_use = "Current lead benchmark family for this asset and the cleanest market-only ML reference."
        else:
            recommended_use = "Clean market-only ML baseline for comparison and backup research."
    elif strategy_key == "market_onchain_specialist":
        recommended_use = "Use this as a daily confirmation or warning layer, not as the primary live trading family."
        if daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.50:
            recommended_use += " Current daily evidence is supportive enough to keep it visible."
        elif daily_structural_label in {"distribution_risk", "divergence_watch"}:
            recommended_use += " Current daily evidence is showing risk, so it is better used as caution."
        elif daily_confirmation_mode == "supportive_narrow":
            recommended_use += " Current support is narrow, so this layer should confirm rather than decide."
    else:
        recommended_use = "Use mainly for research comparison until stronger governance and evaluation support exist."

    context_dependency = "market_futures_core"
    if capabilities.get("effective_sentiment_available") or capabilities.get("onchain_available"):
        context_dependency = "market_futures_with_optional_context"
    if strategy_key == "unscaled_model":
        context_dependency = "market_only_baseline"

    if family_governance["strategy_alignment"] == "secondary_for_asset":
        risk_flags.append("Asset-aware checks currently prefer the market-only benchmark over this model.")
    if family_governance["strategy_alignment"] == "benchmark_only_for_asset":
        risk_flags.append("Asset-aware checks currently treat this strategy as a benchmark rather than the lead model.")
    if family_governance.get("fallback_trigger") in {"hold_collapse", "dont_buy_collapse"}:
        risk_flags.append("The three-class market+futures model is becoming one-sided, so backup rules are active.")
    if family_governance.get("structural_break_status") == "break":
        risk_flags.append("Recent market-shift checks are reducing trust in previously learned patterns.")
    if futures_support_mode == "robust_partial":
        risk_flags.append("Futures context is still robust overall, but current sublayer coverage is only partial, so futures-led confidence should stay measured.")
    if futures_support_mode == "fragile":
        risk_flags.append("Current futures support is fragile, so market-only fallback should be preferred until coverage improves.")
    if family_governance.get("operational_family") == "daily_structural_research":
        risk_flags.append("Daily outputs are still best treated as research context rather than the primary live trading family.")
    if resolved_timeframe == "4h" and daily_structural_label in {"distribution_risk", "divergence_watch"}:
        risk_flags.append("Daily evidence is showing risk, so the 4h view should stay conservative.")
    if strategy_key == "market_onchain_specialist" and daily_structural_confidence < 0.35:
        risk_flags.append("Daily confidence is still low, so on-chain should remain a support layer rather than a decisive signal.")
    if resolved_timeframe == "4h" and daily_confirmation_mode == "supportive_narrow":
        risk_flags.append("Daily on-chain support is present but narrow, so 4h strategies should avoid overclaiming broad structural backing.")
    if pipeline_refresh_status in {"missing", "failed", "completed_no_overview"}:
        risk_flags.append("Pipeline refresh coverage is incomplete for this timeframe, so user-facing confidence should stay conservative.")
    elif pipeline_freshness_label == "stale":
        risk_flags.append(
            f"The saved evaluation window is stale ({pipeline_window_age_days:.0f} days old), so this strategy should be framed more cautiously until a recent rerun is completed."
        )
    elif pipeline_freshness_label == "recent":
        risk_flags.append("The saved evaluation window is recent but not fully current, so wording should stay measured.")

    if pipeline_refresh_status in {"missing", "failed", "completed_no_overview"} and readiness_label not in {"Unavailable", "Experimental", "Research", "Fallback"}:
        readiness_label = "Fallback"
        deployment_tier = "fallback_only"
        evaluation_reason += " Pipeline refresh coverage is incomplete for this timeframe, so backup caution should stay active."
    elif pipeline_freshness_label == "stale" and readiness_label in {"Ready", "Benchmark Ready"}:
        readiness_label = "Conditional"
        deployment_tier = "conditional_support"
        evaluation_reason += f" The latest saved evaluation window is {pipeline_window_age_days:.0f} days old, so the signal should be treated with caution until a refresh is run."
    elif pipeline_freshness_label == "recent" and readiness_label == "Ready":
        readiness_label = "Conditional"
        deployment_tier = "conditional_support"
        evaluation_reason += " The latest saved evaluation window is recent but not fully current, so this should stay slightly measured."

    return {
        "readiness_label": readiness_label,
        "deployment_tier": deployment_tier,
        "runtime_label": runtime_label,
        "runtime_role": runtime_role,
        "timeframe_integrity": "exact" if exact_timeframe_match else "fallback",
        "evaluation_strength": evaluation_strength,
        "evaluation_reason": evaluation_reason,
        "recommended_use": recommended_use,
        "context_dependency": context_dependency,
        "daily_posture_label": daily_posture_label,
        "daily_structural_label": daily_structural_label,
        "daily_structural_confidence": daily_structural_confidence,
        "daily_confirmation_mode": daily_confirmation_mode,
        "futures_support_mode": futures_support_mode,
        "pipeline_freshness_label": pipeline_freshness_label,
        "pipeline_refresh_status": pipeline_refresh_status,
        "pipeline_window_age_days": pipeline_window_age_days,
        "risk_flags": risk_flags,
        "family_governance": family_governance,
    }


def build_multimodal_assessment(summary, runtime_support=None):
    """Assess whether sentiment and on-chain currently add meaningful value."""
    summary = summary or {}
    runtime_support = runtime_support or {}
    delta_macro_f1 = _safe_float(summary.get("delta_macro_f1_vs_market_futures", 0))
    best_variant = str(summary.get("ablation_best_variant", "n/a") or "n/a")
    selected_variant = str(summary.get("multimodal_selected_context_variant", "n/a") or "n/a")
    gdelt_reliability_label = str(summary.get("gdelt_reliability_label", "unavailable") or "unavailable")
    onchain_reliability_label = str(summary.get("onchain_reliability_label", "unavailable") or "unavailable")
    combined_context_readiness = str(summary.get("combined_context_readiness", "limited_context_confirmation") or "limited_context_confirmation")
    strategy_name = str(summary.get("strategy_name", "n/a") or "n/a")
    context_overlay_lead = str(
        summary.get("context_overlay_lead", "market_futures_core_keep_context_secondary")
        or "market_futures_core_keep_context_secondary"
    )
    gdelt_dominant_theme = str(summary.get("latest_gdelt_dominant_event_theme", summary.get("gdelt_dominant_event_theme", "none")) or "none")
    onchain_support_driver = str(summary.get("latest_onchain_primary_support_driver", "none") or "none")
    onchain_risk_driver = str(summary.get("latest_onchain_primary_risk_driver", "none") or "none")
    gdelt_available = gdelt_reliability_label != "unavailable"
    onchain_available = onchain_reliability_label != "unavailable"
    sentiment_timeframes = runtime_support.get("sentiment_timeframes", [])
    onchain_timeframes = runtime_support.get("onchain_timeframes", [])

    if delta_macro_f1 > 0.03 and combined_context_readiness == "usable_context_confirmation":
        uplift_label = "Improving"
        recommendation = "Context layers are helping enough to keep them as confirmation or warning inputs."
    elif delta_macro_f1 > 0.01 and combined_context_readiness == "conditional_context_confirmation":
        uplift_label = "Conditional"
        recommendation = "Context layers are adding some value, but they should remain gated by stronger market and futures evidence."
    elif delta_macro_f1 < -0.01:
        uplift_label = "Weakening"
        recommendation = "Current context layers are hurting or not clearly helping, so they should stay secondary."
    else:
        uplift_label = "Mixed"
        recommendation = "Context layers are informative, but they are not yet strong enough to replace the market+futures core."

    if strategy_name == "market_multimodal_news_event_veto":
        recommendation = (
            "Treat asset news as an event-risk warning layer: useful for caution, but not yet strong enough to "
            "become the primary prediction route."
        )
    elif context_overlay_lead == "cautious_multimodal_overlay":
        recommendation = (
            "The cautious multimodal route is currently the strongest context overlay, but it should still sit behind "
            "the market + futures core as a trust modifier."
        )
    elif context_overlay_lead == "onchain_structural_confirmation_lead":
        recommendation = (
            "On-chain is currently the clearest context route, so multimodal logic should use it as a longer-term "
            "confirmation check rather than broad sentiment."
        )
    elif context_overlay_lead == "news_event_veto_lead":
        recommendation = (
            "Context logic should treat news mainly as a warning and caution layer while keeping direction decisions tied "
            "to the market + futures backbone."
        )

    coverage_note = []
    if gdelt_available:
        coverage_note.append(f"asset news sentiment {gdelt_reliability_label.replace('_', ' ')}")
        if gdelt_dominant_theme != "none":
            coverage_note.append(f"news theme {_label_text(gdelt_dominant_theme, 'none')}")
    else:
        coverage_note.append("asset news sentiment limited")
    if onchain_available:
        coverage_note.append(f"on-chain {onchain_reliability_label.replace('_', ' ')}")
        coverage_note.append(
            f"on-chain drivers {_label_text(onchain_support_driver, 'none')} / {_label_text(onchain_risk_driver, 'none')}"
        )
    else:
        coverage_note.append("on-chain limited")

    return {
        "uplift_label": uplift_label,
        "delta_macro_f1": delta_macro_f1,
        "best_variant": best_variant,
        "selected_variant": selected_variant,
        "combined_context_readiness": combined_context_readiness,
        "context_overlay_lead": context_overlay_lead,
        "context_overlay_label": CONTEXT_OVERLAY_LABELS.get(
            context_overlay_lead,
            "Core-first with secondary context",
        ),
        "coverage_note": ", ".join(coverage_note),
        "recommendation": recommendation,
        "sentiment_timeframes": sentiment_timeframes,
        "onchain_timeframes": onchain_timeframes,
    }


def build_config_governance(config, market_summary=None):
    """Attach honest implementation guidance to a resolved strategy config."""
    config = config or {}
    market_summary = market_summary or {}
    required_layers = config.get("required_layers", [])
    optional_layers = config.get("optional_layers", [])
    unavailable_layers = config.get("unavailable_layers", [])
    scorecard = config.get("scorecard", {})
    asset_capabilities = config.get("asset_capabilities", {})
    if config.get("model_family") == "scaled_market_baseline":
        governance_strategy_key = "scaled_model"
    elif config.get("model_family") in {"daily_structural_confirmation_family", "market_onchain_specialist_daily"}:
        governance_strategy_key = "market_onchain_specialist"
    else:
        governance_strategy_key = "market_futures_backend"

    family_governance = build_family_governance(
        market_summary,
        governance_strategy_key,
    )
    futures_support_mode = family_governance.get("futures_support_mode", "unavailable")

    evaluation_strength, evaluation_reason = _derive_evaluation_strength(
        _safe_float(scorecard.get("reference_accuracy", 0)),
        _safe_float(scorecard.get("reference_macro_f1", 0)),
        4 if _safe_float(scorecard.get("reference_excess_return", 0)) or market_summary.get("walkforward_fold_count") else 0,
        config.get("resolved_timeframe") == config.get("requested_timeframe"),
    )

    if config.get("mode") == "preset" and config.get("strategy_id") == "recommended":
        deployment_tier = "production_candidate"
        readiness_label = "Ready" if not unavailable_layers else "Partial"
        preset_truth_status = "direct_backend_reference"
    elif config.get("mode") == "preset":
        deployment_tier = "structured_preset"
        readiness_label = "Partial" if unavailable_layers else "Mapped"
        preset_truth_status = "mapped_benchmark_reference"
    else:
        deployment_tier = "builder_blueprint"
        readiness_label = "Reduced" if unavailable_layers else "Research"
        preset_truth_status = "mapped_benchmark_reference"

    sentiment_state = "not_selected"
    if "sentiment" in required_layers or "sentiment" in optional_layers:
        if asset_capabilities.get("gdelt_reliability_label") == "usable_confirmation":
            sentiment_state = "asset_specific_confirmation"
        elif asset_capabilities.get("gdelt_reliability_label") in {"thin_confirmation", "conditional_confirmation"}:
            sentiment_state = "thin_asset_specific_confirmation"
        elif asset_capabilities.get("effective_sentiment_available"):
            sentiment_state = "fallback_confirmation"
        else:
            sentiment_state = "unavailable"

    onchain_state = "not_selected"
    if "onchain" in required_layers or "onchain" in optional_layers:
        if asset_capabilities.get("onchain_reliability_label") == "usable_structural_confirmation":
            onchain_state = "structural_confirmation"
        elif asset_capabilities.get("onchain_reliability_label") == "conditional_structural_confirmation":
            onchain_state = "conditional_structural_confirmation"
        else:
            onchain_state = "unavailable"

    combined_context_readiness = str(
        asset_capabilities.get("combined_context_readiness", "limited_context_confirmation") or "limited_context_confirmation"
    )
    daily_posture_label = str(market_summary.get("daily_posture_label", "structural_mixed") or "structural_mixed")
    daily_structural_label = str(
        market_summary.get("latest_market_onchain_structural_label", market_summary.get("combined_daily_view", "mixed"))
        or "mixed"
    )
    daily_structural_confidence = _safe_float(
        market_summary.get("latest_onchain_confidence_score", market_summary.get("latest_onchain_overlay_confidence", 0))
    )
    daily_confirmation_mode = _derive_daily_confirmation_mode(market_summary)

    if config.get("strategy_id") == "multimodal_balanced":
        if combined_context_readiness == "usable_context_confirmation":
            readiness_label = "Conditional"
            deployment_tier = "conditional_support"
            evaluation_reason = (
                "This context-aware preset has usable enough context support to stay visible, but it should still act as conditional confirmation rather than a primary engine."
            )
        elif combined_context_readiness == "conditional_context_confirmation":
            readiness_label = "Reduced"
            deployment_tier = "research_only"
            evaluation_reason = (
                "This context-aware preset has some support, but current sentiment/on-chain reliability is only conditional."
            )
        else:
            readiness_label = "Reduced"
            deployment_tier = "research_only"
            evaluation_reason = (
                "This context-aware preset should stay research-only because current sentiment/on-chain reliability is too limited."
            )
        if daily_structural_label in {"distribution_risk", "divergence_watch"}:
            evaluation_reason += " Daily structural alignment is showing divergence risk, so context should lean more toward caution than confirmation."
        elif daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.55:
            evaluation_reason += " Daily structural evidence is supportive enough to justify keeping this preset visible as conditional confirmation."
        if daily_confirmation_mode == "supportive_narrow":
            evaluation_reason += " That support is currently narrower and more valuation-led than broad participation-led."

    if config.get("strategy_id") == "daily_structural_confirmation":
        if combined_context_readiness == "usable_context_confirmation":
            readiness_label = "Research"
            deployment_tier = "research_only"
            evaluation_reason = (
                "Daily structural confirmation is the strongest current slower context route, but it should still remain a research and veto layer rather than a live execution default."
            )
        elif combined_context_readiness == "conditional_context_confirmation":
            readiness_label = "Reduced"
            deployment_tier = "research_only"
            evaluation_reason = (
                "Daily structural confirmation exists, but the slower context layers are only conditionally trustworthy right now."
            )
        else:
            readiness_label = "Reduced"
            deployment_tier = "research_only"
            evaluation_reason = (
                "Daily structural confirmation is currently limited by weak context support and should stay dissertation-facing rather than decision-leading."
            )

        if daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.50:
            evaluation_reason += " The latest daily structural posture is supportive with enough confidence to justify keeping it visible as a meaningful confirmation layer."
        elif daily_structural_label in {"distribution_risk", "divergence_watch"}:
            evaluation_reason += " The latest daily structural alignment is showing divergence or distribution risk, so this layer should lean toward caution and veto."
        elif daily_confirmation_mode == "supportive_narrow":
            evaluation_reason += " The latest support is narrower and should be framed as targeted structural help rather than broad regime strength."

    if config.get("strategy_id") == "conservative_trend" and daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.50:
        evaluation_reason += " Daily structural confirmation is currently supportive, which fits the slower confirmation logic behind this preset."
    elif config.get("strategy_id") == "conservative_trend" and daily_structural_label in {"distribution_risk", "divergence_watch"}:
        evaluation_reason += " Daily structural alignment is currently cautionary, so this preset should stay stricter about confirmation."
    elif config.get("strategy_id") == "conservative_trend" and daily_confirmation_mode == "supportive_narrow":
        evaluation_reason += " The daily support is narrower than broad participation, which suits a stricter confirmation preset better than a more aggressive one."

    if config.get("strategy_id") == "futures_crowd_reversal":
        if daily_structural_label in {"distribution_risk", "divergence_watch"}:
            evaluation_reason += " Daily structural divergence increases the relevance of this reversal-style preset as a research comparison."
        elif daily_confirmation_mode == "supportive_narrow":
            evaluation_reason += " Daily support exists, but it is narrower and valuation-led, so reversal logic should remain secondary rather than central."
        elif daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.55:
            evaluation_reason += " Daily structure is broadly supportive, so reversal logic should stay secondary rather than central."

    if config.get("strategy_id") == "recommended":
        if futures_support_mode == "robust_partial" and readiness_label in {"Ready", "Partial"}:
            readiness_label = "Conditional"
            deployment_tier = "conditional_support"
            evaluation_reason += " Futures support is robust overall, but current sublayer coverage is only partial, so the default preset should stay a little more measured."
        elif futures_support_mode == "fragile":
            readiness_label = "Reduced"
            deployment_tier = "fallback_only"
            evaluation_reason += " Futures support is currently fragile, so the default preset should lean on fallback logic rather than full futures-led confidence."
        if daily_confirmation_mode == "supportive_narrow":
            evaluation_reason += " The latest higher-timeframe support is narrower than broad participation, so the default preset should present this as measured confirmation rather than full structural backing."
        elif daily_confirmation_mode == "supportive_broad":
            evaluation_reason += " The latest higher-timeframe support is broad enough to strengthen trust in the default preset."

    if config.get("strategy_id") in {"conservative_trend", "momentum_breakout", "futures_crowd_reversal"}:
        if futures_support_mode == "robust_partial" and readiness_label in {"Mapped", "Partial"}:
            readiness_label = "Conditional"
            deployment_tier = "conditional_support"
            evaluation_reason += " Futures support is still broadly usable, but current sublayer coverage is only partial, so this preset should be framed more cautiously."
        elif futures_support_mode == "fragile":
            readiness_label = "Reduced"
            deployment_tier = "research_only"
            evaluation_reason += " Futures support is currently fragile, so this futures-sensitive preset should stay more comparative than decision-leading."

    if config.get("strategy_id") == "daily_structural_confirmation" and daily_posture_label == "structural_supportive" and daily_structural_confidence < 0.55:
        readiness_label = "Reduced"
        evaluation_reason += " Even though posture is supportive, confidence is still only middling, so the layer should stay cautious."

    return {
        "readiness_label": readiness_label,
        "deployment_tier": deployment_tier,
        "evaluation_strength": evaluation_strength,
        "evaluation_reason": evaluation_reason,
        "lead_family_label": family_governance["lead_family_label"],
        "family_recommendation": family_governance["recommended_family"],
        "family_alignment": family_governance["strategy_alignment"],
        "family_evidence_summary": family_governance["evidence_summary"],
        "sentiment_state": sentiment_state,
        "onchain_state": onchain_state,
        "combined_context_readiness": combined_context_readiness,
        "daily_posture_label": daily_posture_label,
        "daily_structural_label": daily_structural_label,
        "daily_structural_confidence": daily_structural_confidence,
        "daily_confirmation_mode": daily_confirmation_mode,
        "futures_support_mode": futures_support_mode,
        "preset_truth_status": preset_truth_status,
        "required_layer_summary": _format_layers(required_layers),
        "optional_layer_summary": _format_layers(optional_layers),
        "unavailable_layer_summary": _format_layers(unavailable_layers),
    }
