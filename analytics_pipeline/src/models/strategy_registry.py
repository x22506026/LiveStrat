"""User-facing strategy registry and backend blueprint helpers for LiveStrat."""

import pandas as pd

from .strategy_governance import build_config_governance, build_family_governance


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value is pd.NA or str(value).strip() == "":
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def _derive_futures_support_note(market_summary):
    market_summary = market_summary or {}
    resilience_label = str(market_summary.get("futures_context_resilience_label", "unavailable") or "unavailable")
    completeness_label = str(market_summary.get("futures_completeness_label", "unavailable") or "unavailable")
    basis_feature_available = str(market_summary.get("basis_feature_available", "")).strip().lower() == "true"

    if resilience_label == "robust" and completeness_label == "full":
        return " Futures structure is currently broad and fully covered."
    if resilience_label == "robust" and (completeness_label == "partial" or not basis_feature_available):
        return " Futures structure is still broadly usable, but current sublayer coverage is partial."
    if resilience_label == "fragile":
        return " Futures structure is currently fragile, so futures-led conviction should stay reduced."
    return ""


SUPPORTED_STRATEGY_TIMEFRAMES = ("1h", "4h", "1d")
FINAL_TIMEFRAME_STRATEGY_ARCHITECTURE = {
    "1h": {
        "label": "Intraday execution",
        "summary": "Use benchmark-first and binary-aware strategies for noisier, faster reaction windows.",
    },
    "4h": {
        "label": "Core swing execution",
        "summary": "Use the strongest current market-plus-futures backbone and comparison benchmarks.",
    },
    "1d": {
        "label": "Structural confirmation",
        "summary": "Use daily and context strategies as extra checks, not as the main trading signal.",
    },
}

TIMEFRAME_POLICY_MAP = {
    "1h": {
        "label": "Intraday",
        "target_family_hint": "fixed_h24",
        "policy_bias": "faster_reaction",
        "execution_risk": "higher_noise",
        "readiness_bias": "conditional",
        "preferred_families": ["market_only", "binary_directional"],
        "fallback_order": ["market_only_benchmark", "binary_directional_fallback", "conditional_market_futures"],
    },
    "4h": {
        "label": "Core swing",
        "target_family_hint": "fixed_h24",
        "policy_bias": "balanced_confirmation",
        "execution_risk": "moderate",
        "readiness_bias": "strongest_current_support",
        "preferred_families": ["market_futures", "binary_directional", "market_only"],
        "fallback_order": ["market_futures_backbone", "binary_directional_fallback", "market_only_benchmark"],
    },
    "1d": {
        "label": "Structural daily",
        "target_family_hint": "fixed_h72",
        "policy_bias": "slower_confirmation",
        "execution_risk": "lower_frequency",
        "readiness_bias": "research_only",
        "preferred_families": ["market_only", "context_confirmation"],
        "fallback_order": ["market_only_benchmark", "reduced_trust_benchmark"],
    },
}


def _normalize_timeframe(value, default="4h"):
    value = str(value or default).strip().lower()
    return value if value in SUPPORTED_STRATEGY_TIMEFRAMES else default


def _normalize_timeframe_selection(timeframe_selection, default="4h"):
    if isinstance(timeframe_selection, str):
        timeframe_values = [timeframe_selection]
    elif isinstance(timeframe_selection, (list, tuple, set)):
        timeframe_values = list(timeframe_selection)
    else:
        timeframe_values = [default]

    normalized = []
    for timeframe in timeframe_values:
        normalized_timeframe = _normalize_timeframe(timeframe, default=default)
        if normalized_timeframe not in normalized:
            normalized.append(normalized_timeframe)

    return normalized or [_normalize_timeframe(default)]


def _resolve_custom_timeframes(selection, default="4h"):
    timeframe_scope = str(selection.get("timeframe_scope", "") or "").strip().lower()
    if timeframe_scope == "1h_4h_stack":
        return ["1h", "4h"]
    if timeframe_scope == "4h_1d_stack":
        return ["4h", "1d"]
    if timeframe_scope in SUPPORTED_STRATEGY_TIMEFRAMES:
        return [timeframe_scope]
    return _normalize_timeframe_selection(selection.get("timeframes", [default]), default=default)


def build_timeframe_strategy_policy(requested_timeframe="4h", resolved_timeframe=None, selected_timeframes=None):
    requested = _normalize_timeframe(requested_timeframe)
    resolved = _normalize_timeframe(resolved_timeframe or requested)
    selected = _normalize_timeframe_selection(selected_timeframes or [requested], default=requested)
    exact_match = requested == resolved
    profile = TIMEFRAME_POLICY_MAP.get(resolved, TIMEFRAME_POLICY_MAP["4h"])

    if resolved == "1d":
        deployment_posture = "research_only"
        fallback_policy = "daily_outputs_should_not_override_faster_live_backbones"
    elif resolved == "1h":
        deployment_posture = "conditional_live_candidate"
        fallback_policy = "fallback_to_market_only_when_binary_or_backbone_deployment_is_weak"
    else:
        deployment_posture = "primary_live_candidate"
        fallback_policy = "fallback_to_binary_or_market_only_if_backbone_collapses"

    return {
        "requested_timeframe": requested,
        "resolved_timeframe": resolved,
        "selected_timeframes": selected,
        "exact_timeframe_match": exact_match,
        "primary_timeframe": resolved,
        "timeframe_label": profile["label"],
        "target_family_hint": profile["target_family_hint"],
        "policy_bias": profile["policy_bias"],
        "execution_risk": profile["execution_risk"],
        "readiness_bias": profile["readiness_bias"],
        "preferred_families": profile["preferred_families"],
        "fallback_order": profile["fallback_order"],
        "deployment_posture": deployment_posture,
        "fallback_policy": fallback_policy,
        "timeframe_scope": "multi_timeframe" if len(selected) > 1 else "single_timeframe",
    }


def build_asset_capability_state(asset, market_summary=None):
    """Resolve what data layers are actually available for one asset right now."""
    market_summary = market_summary or {}

    latest_gdelt_regime_label = str(market_summary.get("latest_gdelt_regime_label", "unavailable") or "unavailable")
    latest_effective_sentiment_label = str(
        market_summary.get("latest_effective_sentiment_label", "unavailable") or "unavailable"
    )
    latest_effective_sentiment_source = str(
        market_summary.get("latest_effective_sentiment_source", "unavailable") or "unavailable"
    )
    latest_onchain_regime_label = str(market_summary.get("latest_onchain_regime_label", "unavailable") or "unavailable")
    latest_onchain_snapshot_label = str(
        market_summary.get("latest_onchain_snapshot_label", "unavailable") or "unavailable"
    )
    latest_onchain_snapshot_status = str(
        market_summary.get("latest_onchain_snapshot_status", "unavailable") or "unavailable"
    )
    latest_onchain_snapshot_age_days = market_summary.get("latest_onchain_snapshot_age_days")
    gdelt_reliability_label = str(market_summary.get("gdelt_reliability_label", "unavailable") or "unavailable")
    broad_sentiment_reliability_label = str(
        market_summary.get("broad_sentiment_reliability_label", "unavailable") or "unavailable"
    )
    effective_sentiment_role = str(market_summary.get("effective_sentiment_role", "unavailable") or "unavailable")
    combined_context_readiness = str(
        market_summary.get("combined_context_readiness", "limited_context_confirmation") or "limited_context_confirmation"
    )
    onchain_reliability_label = str(
        market_summary.get("onchain_reliability_label", "unavailable") or "unavailable"
    )
    futures_completeness_label = str(
        market_summary.get("futures_completeness_label", "unavailable") or "unavailable"
    )
    futures_context_resilience_label = str(
        market_summary.get("futures_context_resilience_label", "unavailable") or "unavailable"
    )
    futures_feature_completeness_score = _safe_float(market_summary.get("futures_feature_completeness_score", 0))
    futures_context_resilience_score = _safe_float(market_summary.get("futures_context_resilience_score", 0))
    futures_basis_reliance_score = _safe_float(market_summary.get("futures_basis_reliance_score", 0))
    basis_feature_available = _safe_bool(market_summary.get("basis_feature_available"))
    basis_feature_mode = str(market_summary.get("basis_feature_mode", "unavailable") or "unavailable")
    basis_proxy_active = _safe_bool(market_summary.get("basis_proxy_active"))
    effective_basis_feature_available = _safe_bool(market_summary.get("effective_basis_feature_available"))
    defi_context_available = _safe_bool(market_summary.get("defi_context_available"))
    defi_snapshot_status = str(market_summary.get("defi_snapshot_status", "unavailable") or "unavailable")
    defi_regime_label = str(market_summary.get("defi_regime_label", "unavailable") or "unavailable")
    defi_chain_name = str(market_summary.get("defi_chain_name", "unavailable") or "unavailable")
    defi_tvl_usd = _safe_float(market_summary.get("latest_defi_tvl_usd", 0))
    defi_tvl_change_pct_30d = _safe_float(market_summary.get("defi_tvl_change_pct_30d", 0))
    defi_snapshot_age_days = market_summary.get("defi_snapshot_age_days")
    selected_primary_model = str(
        market_summary.get("selected_primary_model", market_summary.get("selected_backend_model", "")) or ""
    )

    market_available = True
    futures_available = (
        market_summary.get("current_pipeline_mode") == "market_futures_backend"
        or "futures" in selected_primary_model
        or "market_futures" in selected_primary_model
        or futures_context_resilience_label in {"robust", "partial"}
    )
    gdelt_asset_news_available = latest_gdelt_regime_label != "unavailable"
    effective_sentiment_available = latest_effective_sentiment_label != "unavailable"
    snapshot_has_onchain = (
        latest_onchain_snapshot_label != "unavailable"
        and latest_onchain_snapshot_status in {"available", "stale"}
    )
    onchain_available = (
        latest_onchain_regime_label != "unavailable"
        or snapshot_has_onchain
    )
    defi_available = defi_context_available or defi_snapshot_status in {"fresh", "aging", "stale"}

    sentiment_mode = "unavailable"
    if gdelt_asset_news_available:
        sentiment_mode = "asset_specific_news"
    elif effective_sentiment_available and latest_effective_sentiment_source == "fear_greed_market_fallback":
        sentiment_mode = "market_wide_fallback"

    combined_multimodal_ready = market_available and futures_available and (
        combined_context_readiness in {"usable_context_confirmation", "conditional_context_confirmation"}
        or defi_available
    )

    return {
        "asset": asset,
        "market_available": market_available,
        "futures_available": futures_available,
        "gdelt_asset_news_available": gdelt_asset_news_available,
        "effective_sentiment_available": effective_sentiment_available,
        "effective_sentiment_source": latest_effective_sentiment_source,
        "effective_sentiment_label": latest_effective_sentiment_label,
        "sentiment_mode": sentiment_mode,
        "effective_sentiment_role": effective_sentiment_role,
        "gdelt_reliability_label": gdelt_reliability_label,
        "broad_sentiment_reliability_label": broad_sentiment_reliability_label,
        "futures_completeness_label": futures_completeness_label,
        "futures_context_resilience_label": futures_context_resilience_label,
        "futures_feature_completeness_score": futures_feature_completeness_score,
        "futures_context_resilience_score": futures_context_resilience_score,
        "futures_basis_reliance_score": futures_basis_reliance_score,
        "basis_feature_available": basis_feature_available,
        "basis_feature_mode": basis_feature_mode,
        "basis_proxy_active": basis_proxy_active,
        "effective_basis_feature_available": effective_basis_feature_available,
        "onchain_available": onchain_available,
        "onchain_regime_label": latest_onchain_regime_label,
        "onchain_snapshot_label": latest_onchain_snapshot_label,
        "onchain_snapshot_status": latest_onchain_snapshot_status,
        "onchain_snapshot_age_days": latest_onchain_snapshot_age_days,
        "onchain_fresh_available": latest_onchain_snapshot_status == "available",
        "onchain_reliability_label": onchain_reliability_label,
        "defi_available": defi_available,
        "defi_chain_name": defi_chain_name,
        "defi_tvl_usd": defi_tvl_usd,
        "defi_tvl_change_pct_30d": defi_tvl_change_pct_30d,
        "defi_snapshot_status": defi_snapshot_status,
        "defi_regime_label": defi_regime_label,
        "defi_snapshot_age_days": defi_snapshot_age_days,
        "combined_context_readiness": combined_context_readiness,
        "combined_multimodal_ready": combined_multimodal_ready,
    }


def build_capability_notes(capabilities):
    """Create plain-English notes explaining what is and is not available."""
    notes = []
    if capabilities["futures_available"]:
        futures_resilience = capabilities.get("futures_context_resilience_label", "unavailable").replace("_", " ")
        futures_completeness = capabilities.get("futures_completeness_label", "unavailable").replace("_", " ")
        if capabilities.get("basis_feature_available"):
            notes.append(
                f"Futures structure is available with {futures_resilience} resilience and {futures_completeness} sublayer coverage."
            )
        elif capabilities.get("basis_proxy_active"):
            basis_mode = str(capabilities.get("basis_feature_mode", "proxy")).replace("_", " ")
            notes.append(
                f"Futures structure is available with {futures_resilience} resilience and {futures_completeness} sublayer coverage; basis is currently using a {basis_mode} fallback."
            )
        else:
            notes.append(
                f"Futures structure is available with {futures_resilience} resilience and {futures_completeness} sublayer coverage, but basis is currently missing so futures context is less complete."
            )
    else:
        notes.append("Futures structure is currently unavailable for this asset/timeframe.")

    if capabilities["gdelt_asset_news_available"]:
        notes.append(
            f"Asset-specific news sentiment is available, but current reliability is "
            f"{capabilities.get('gdelt_reliability_label', 'unavailable').replace('_', ' ')}."
        )
    elif capabilities["effective_sentiment_available"]:
        notes.append("Asset-specific news sentiment is unavailable, so sentiment falls back to market-wide mood context.")
    else:
        notes.append("Sentiment context is currently unavailable for this asset.")

    if capabilities.get("onchain_snapshot_status") == "available":
        notes.append(
            f"On-chain context is available with "
            f"{capabilities.get('onchain_reliability_label', 'unavailable').replace('_', ' ')} reliability."
        )
    elif capabilities.get("onchain_snapshot_status") == "stale":
        notes.append(
            f"The latest available on-chain snapshot is stale ({capabilities.get('onchain_snapshot_label', 'unknown')}, {capabilities.get('onchain_snapshot_age_days', 'n/a')} days old)."
        )
    else:
        notes.append("On-chain context is currently unavailable.")

    if capabilities.get("defi_available"):
        tvl = capabilities.get("defi_tvl_usd", 0)
        tvl_note = f", TVL ${tvl:,.0f}" if tvl else ""
        notes.append(
            f"DeFi ecosystem context is available ({capabilities.get('defi_regime_label', 'unavailable').replace('_', ' ')}, "
            f"{capabilities.get('defi_snapshot_status', 'unknown')} snapshot{tvl_note})."
        )
    else:
        notes.append("DeFi ecosystem context is currently unavailable.")

    if capabilities["combined_multimodal_ready"]:
        notes.append(
            f"Combined multimodal strategies can use extra context, with overall readiness at "
            f"{capabilities.get('combined_context_readiness', 'limited_context_confirmation').replace('_', ' ')}."
        )
    else:
        notes.append(
            "Combined multimodal strategies should be treated cautiously because context reliability is still limited."
        )
    return notes


def adapt_layers_to_capabilities(required_layers, optional_layers, capabilities):
    """Filter and annotate requested layers against the asset's current capability state."""
    available_map = {
        "market": capabilities["market_available"],
        "futures": capabilities["futures_available"],
        "sentiment": capabilities["effective_sentiment_available"],
        "onchain": capabilities["onchain_available"],
        "defi": capabilities.get("defi_available", False),
    }
    requested_layers = list(dict.fromkeys((required_layers or []) + (optional_layers or [])))
    unavailable_layers = [layer for layer in requested_layers if not available_map.get(layer, False)]
    resolved_required_layers = [layer for layer in required_layers if available_map.get(layer, False)]
    resolved_optional_layers = [layer for layer in optional_layers if available_map.get(layer, False)]

    if "market" not in resolved_required_layers:
        resolved_required_layers.insert(0, "market")
    return resolved_required_layers, resolved_optional_layers, unavailable_layers


def _build_preset_strategies():
    return [
        {
            "id": "recommended",
            "name": "Balanced Default",
            "strategy_family": "core_market_futures",
            "supported_timeframes": ["1h", "4h"],
            "preferred_timeframes": ["4h", "1h"],
            "timeframe_role": "adaptive_default",
            "deployment_role": "live_candidate",
            "evaluation_basis": "direct_backend_reference",
            "distinct_engine_status": "direct_current_backend",
            "tagline": "LiveStrat's strongest current all-round strategy choice per asset.",
            "core_engine": "Market + Futures",
            "risk_profile": "Balanced",
            "best_for": "Users who want the strongest current default without tuning filters.",
            "explanation": (
                "Uses the asset's currently selected backend model, calibrated policy, and latest "
                "walk-forward-tested configuration."
            ),
            "confirmation_layers": ["Futures structure"],
            "optional_layers": ["Sentiment", "On-chain"],
            "backend_blueprint": {
                "target_family": "asset_selected_preferred_target",
                "model_family": "selected_market_futures_backend",
                "policy_family": "selected_policy_family",
                "probability_mode": "selected_probability_mode",
                "confirmation_mode": "selected_backend_confirmation",
                "required_layers": ["market", "futures"],
                "optional_layers": ["sentiment", "onchain"],
            },
        },
        {
            "id": "conservative_trend",
            "name": "Trend Confirmation",
            "strategy_family": "structured_market_futures_preset",
            "supported_timeframes": ["4h", "1d"],
            "preferred_timeframes": ["4h", "1d"],
            "timeframe_role": "swing_and_structural",
            "deployment_role": "benchmark_plus_confirmation",
            "evaluation_basis": "mapped_benchmark_reference",
            "distinct_engine_status": "mapped_to_shared_evaluated_family",
            "tagline": "Only enters when directional trend and confirmation are both strong.",
            "core_engine": "Market trend",
            "risk_profile": "Low",
            "best_for": "Users who prefer fewer entries and more time in cash.",
            "explanation": (
                "Designed around trend-following logic, stronger entry thresholds, and confirmation "
                "filters such as volatility, volume, and futures alignment."
            ),
            "confirmation_layers": ["Volume", "Volatility", "Futures alignment"],
            "optional_layers": ["Sentiment confirmation", "On-chain confirmation"],
            "backend_blueprint": {
                "target_family": "fixed_h24",
                "model_family": "logistic_market_futures",
                "policy_family": "regime_adaptive_long_flat",
                "probability_mode": "temperature_scaled",
                "confirmation_mode": "strict_consensus",
                "required_layers": ["market"],
                "optional_layers": ["futures", "sentiment", "onchain"],
            },
        },
        {
            "id": "momentum_breakout",
            "name": "Momentum Breakout",
            "strategy_family": "structured_market_futures_preset",
            "supported_timeframes": ["1h"],
            "preferred_timeframes": ["1h"],
            "timeframe_role": "intraday_specialist",
            "deployment_role": "conditional_intraday_candidate",
            "evaluation_basis": "mapped_benchmark_reference",
            "distinct_engine_status": "mapped_to_shared_evaluated_family",
            "tagline": "Favors continuation when price and activity accelerate together.",
            "core_engine": "Momentum",
            "risk_profile": "Medium",
            "best_for": "Users who want faster entries when trend and activity both strengthen.",
            "explanation": (
                "Built around short-horizon returns, participation strength, and breakout-style "
                "continuation confirmation."
            ),
            "confirmation_layers": ["Volume", "Trade activity", "Taker flow"],
            "optional_layers": ["Sentiment tailwind", "On-chain support"],
            "backend_blueprint": {
                "target_family": "fixed_h24",
                "model_family": "logistic_market_futures",
                "policy_family": "confidence_gated_long_flat",
                "probability_mode": "raw",
                "confirmation_mode": "weighted_score",
                "required_layers": ["market"],
                "optional_layers": ["futures", "sentiment", "onchain"],
            },
        },
        {
            "id": "futures_crowd_reversal",
            "name": "Crowd Reversal",
            "strategy_family": "futures_structure_research_preset",
            "supported_timeframes": ["4h"],
            "preferred_timeframes": ["4h"],
            "timeframe_role": "4h_specialist",
            "deployment_role": "research_comparison",
            "evaluation_basis": "mapped_benchmark_reference",
            "distinct_engine_status": "mapped_to_shared_evaluated_family",
            "tagline": "Looks for crowded positioning that could unwind.",
            "core_engine": "Futures structure",
            "risk_profile": "High",
            "best_for": "Users who want contrarian, higher-risk setups.",
            "explanation": (
                "Focuses on funding, long/short ratios, open interest pressure, and positioning "
                "imbalances that may justify caution or reversal."
            ),
            "confirmation_layers": ["Funding", "Long/short ratio", "Open interest"],
            "optional_layers": ["Sentiment extreme", "On-chain divergence"],
            "backend_blueprint": {
                "target_family": "voladj_h24",
                "model_family": "logistic_market_futures",
                "policy_family": "conviction_weighted_long_only",
                "probability_mode": "temperature_scaled",
                "confirmation_mode": "double_confirmation",
                "required_layers": ["market", "futures"],
                "optional_layers": ["sentiment", "onchain"],
            },
        },
        {
            "id": "multimodal_balanced",
            "name": "Context-Aware Balanced",
            "strategy_family": "experimental_context_confirmation",
            "supported_timeframes": ["4h"],
            "preferred_timeframes": ["4h"],
            "timeframe_role": "4h_context_research",
            "deployment_role": "experimental_confirmation",
            "evaluation_basis": "mapped_benchmark_reference",
            "distinct_engine_status": "mapped_to_shared_evaluated_family",
            "tagline": "Combines market structure with optional sentiment and on-chain confirmation layers.",
            "core_engine": "Market + Futures",
            "risk_profile": "Balanced",
            "best_for": "Users who want multiple data sources without fully custom tuning.",
            "explanation": (
                "Starts from the market+futures backbone, then optionally requires additional "
                "confirmation from sentiment and on-chain layers before acting."
            ),
            "confirmation_layers": ["Futures structure"],
            "optional_layers": ["Sentiment", "On-chain"],
            "backend_blueprint": {
                "target_family": "preferred_or_multimodal_target",
                "model_family": "logistic_market_futures_plus_confirmations",
                "policy_family": "regime_adaptive_long_flat",
                "probability_mode": "temperature_scaled",
                "confirmation_mode": "double_confirmation",
                "required_layers": ["market", "futures"],
                "optional_layers": ["sentiment", "onchain"],
            },
        },
        {
            "id": "daily_structural_confirmation",
            "name": "Structural Confirmation",
            "strategy_family": "daily_structural_specialist",
            "supported_timeframes": ["4h"],
            "preferred_timeframes": ["4h"],
            "timeframe_role": "4h_structural_confirmation",
            "deployment_role": "confirmation_candidate",
            "evaluation_basis": "daily_structural_specialist_reference",
            "distinct_engine_status": "specialist_daily_context_family",
            "tagline": "Uses daily evidence as an extra check for the core 4h strategy.",
            "core_engine": "Structural context",
            "risk_profile": "Low",
            "best_for": "Users who want a higher-timeframe confirmation layer for swing strategy decisions.",
            "explanation": (
                "Uses daily outputs and context data to check whether the 4h strategy looks supported or risky."
            ),
            "confirmation_layers": ["Daily market alignment", "Structural regime"],
            "optional_layers": ["Sentiment context", "On-chain context"],
            "backend_blueprint": {
                "target_family": "daily_structural_confirmation",
                "model_family": "market_onchain_specialist_daily",
                "policy_family": "structural_confirmation_only",
                "probability_mode": "research_scorecard",
                "confirmation_mode": "daily_structural_confirmation",
                "required_layers": ["market"],
                "optional_layers": ["onchain", "sentiment"],
            },
        },
    ]


def _build_custom_builder():
    return {
        "title": "Build Your Own Strategy",
        "intro": (
            "Custom strategies should stay structured rather than fully freeform. Users choose a core "
            "signal style, then add confirmation layers, risk settings, and decision rules."
        ),
        "sections": [
            {
                "id": "core_signal",
                "label": "Core Signal",
                "description": "Choose the main analytical style that drives entries and exits.",
                "options": [
                    {
                        "id": "trend_following",
                        "label": "Trend Following",
                        "description": "Uses moving-average structure, direction, and continuation logic.",
                    },
                    {
                        "id": "momentum",
                        "label": "Momentum",
                        "description": "Uses return acceleration, participation, and breakout-style behavior.",
                    },
                    {
                        "id": "reversal",
                        "label": "Reversal",
                        "description": "Looks for stretched moves, crowding, and possible mean reversion.",
                    },
                    {
                        "id": "futures_structure",
                        "label": "Futures Structure",
                        "description": "Focuses on funding, open interest, taker flow, and positioning imbalance.",
                    },
                ],
            },
            {
                "id": "timeframe_scope",
                "label": "Timeframe Scope",
                "description": "Choose whether the strategy should act on one timeframe or coordinate across multiple timeframes.",
                "options": [
                    {
                        "id": "1h",
                        "label": "1 hour",
                        "description": "Faster, noisier, and more suitable for binary or benchmark-style execution.",
                    },
                    {
                        "id": "4h",
                        "label": "4 hours",
                        "description": "Strongest current LiveStrat core timeframe and the safest default for most strategies.",
                    },
                    {
                        "id": "1d",
                        "label": "1 day",
                        "description": "Best treated as extra research context rather than the main live signal.",
                    },
                    {
                        "id": "1h_4h_stack",
                        "label": "1h + 4h stack",
                        "description": "Use 4h as the structural anchor and 1h as the execution or confirmation layer.",
                    },
                    {
                        "id": "4h_1d_stack",
                        "label": "4h + 1d stack",
                        "description": "Use daily context as an extra check on a 4h strategy.",
                    },
                ],
            },
            {
                "id": "data_sources",
                "label": "Data Sources",
                "description": "Choose which layers are allowed to influence the strategy.",
                "options": [
                    {
                        "id": "market",
                        "label": "Market",
                        "description": "Required core layer using Binance spot OHLCV and activity features.",
                    },
                    {
                        "id": "futures",
                        "label": "Futures",
                        "description": "Optional structure layer using funding, open interest, and crowding data.",
                    },
                    {
                        "id": "sentiment",
                        "label": "Sentiment",
                        "description": "Optional layer that can support or warn against market signals.",
                    },
                    {
                        "id": "onchain",
                        "label": "On-chain",
                        "description": "Optional layer that can support or warn against market signals.",
                    },
                ],
            },
            {
                "id": "confirmation_filters",
                "label": "Confirmation Filters",
                "description": "Choose which filters must agree before the strategy acts.",
                "options": [
                    {
                        "id": "volume_filter",
                        "label": "Volume",
                        "description": "Only act when participation supports the move.",
                    },
                    {
                        "id": "volatility_filter",
                        "label": "Volatility",
                        "description": "Avoid low-quality signals in unstable conditions.",
                    },
                    {
                        "id": "funding_filter",
                        "label": "Funding",
                        "description": "Use futures funding pressure as confirmation or caution.",
                    },
                    {
                        "id": "open_interest_filter",
                        "label": "Open Interest",
                        "description": "Use leverage build-up or unwind behavior as confirmation.",
                    },
                    {
                        "id": "sentiment_filter",
                        "label": "Sentiment",
                        "description": "Require market mood to support or block the signal.",
                    },
                    {
                        "id": "onchain_filter",
                        "label": "On-chain",
                        "description": "Require network or valuation context to support or block the signal.",
                    },
                ],
            },
            {
                "id": "decision_rules",
                "label": "Decision Rules",
                "description": "Define how many confirmations are needed before an action is taken.",
                "options": [
                    {
                        "id": "single_confirmation",
                        "label": "Require 1 confirmation",
                        "description": "Flexible and more active, but less selective.",
                    },
                    {
                        "id": "double_confirmation",
                        "label": "Require 2 confirmations",
                        "description": "Balanced mode for clearer support before acting.",
                    },
                    {
                        "id": "strict_consensus",
                        "label": "Strict consensus",
                        "description": "Only act when nearly all selected layers agree.",
                    },
                    {
                        "id": "weighted_score",
                        "label": "Weighted score",
                        "description": "Combine selected layers into a weighted decision score.",
                    },
                ],
            },
            {
                "id": "risk_profile",
                "label": "Risk Profile",
                "description": "Translate user preferences into threshold strictness and exposure style.",
                "options": [
                    {
                        "id": "low_risk",
                        "label": "Conservative",
                        "description": "Higher confirmation thresholds, lower exposure, more time in cash.",
                    },
                    {
                        "id": "balanced_risk",
                        "label": "Balanced",
                        "description": "Moderate confirmation thresholds and moderate exposure.",
                    },
                    {
                        "id": "high_risk",
                        "label": "Aggressive",
                        "description": "Lower entry thresholds, faster responses, and more market exposure.",
                    },
                ],
            },
        ],
        "backend_mapping_notes": [
            "Preset strategies should map to tested policy and model families rather than raw model names in the UI.",
            "Sentiment and on-chain should first act as optional confirmation filters, not mandatory inputs for every strategy.",
            "A future multimodal strategy can require sentiment or on-chain agreement before a market-led signal becomes actionable.",
        ],
        "mapping_schema": {
            "core_signal_map": {
                "trend_following": {
                    "target_family": "fixed_h24",
                    "model_family": "logistic_market_core",
                    "policy_family": "regime_adaptive_long_flat",
                    "feature_focus": ["trend", "moving_averages", "volatility"],
                },
                "momentum": {
                    "target_family": "fixed_h24",
                    "model_family": "logistic_market_core",
                    "policy_family": "confidence_gated_long_flat",
                    "feature_focus": ["returns", "volume", "taker_flow"],
                },
                "reversal": {
                    "target_family": "voladj_h24",
                    "model_family": "logistic_market_futures",
                    "policy_family": "conviction_weighted_long_only",
                    "feature_focus": ["mean_reversion", "crowding", "volatility"],
                },
                "futures_structure": {
                    "target_family": "voladj_h24",
                    "model_family": "logistic_market_futures",
                    "policy_family": "regime_adaptive_long_flat",
                    "feature_focus": ["funding", "open_interest", "long_short_ratio"],
                },
            },
            "decision_rule_map": {
                "single_confirmation": {
                    "confirmation_mode": "single_confirmation",
                    "minimum_confirmations": 1,
                },
                "double_confirmation": {
                    "confirmation_mode": "double_confirmation",
                    "minimum_confirmations": 2,
                },
                "strict_consensus": {
                    "confirmation_mode": "strict_consensus",
                    "minimum_confirmations": 3,
                },
                "weighted_score": {
                    "confirmation_mode": "weighted_score",
                    "minimum_confirmations": 0,
                },
            },
            "risk_profile_map": {
                "low_risk": {
                    "risk_profile": "Conservative",
                    "probability_mode": "temperature_scaled",
                    "threshold_profile": "high_confirmation",
                    "exposure_style": "lower_exposure",
                },
                "balanced_risk": {
                    "risk_profile": "Balanced",
                    "probability_mode": "temperature_scaled",
                    "threshold_profile": "balanced_confirmation",
                    "exposure_style": "moderate_exposure",
                },
                "high_risk": {
                    "risk_profile": "Aggressive",
                    "probability_mode": "raw",
                    "threshold_profile": "lower_confirmation",
                    "exposure_style": "higher_exposure",
                },
            },
            "data_source_map": {
                "market": {
                    "layer_role": "required_core",
                    "backend_input": "binance_spot_market_features",
                },
                "futures": {
                    "layer_role": "structural_confirmation",
                    "backend_input": "binance_futures_structure_features",
                },
                "sentiment": {
                    "layer_role": "optional_confirmation_veto",
                    "backend_input": "sentiment_context_features",
                },
                "onchain": {
                    "layer_role": "optional_confirmation_veto",
                    "backend_input": "onchain_context_features",
                },
            },
            "confirmation_filter_map": {
                "volume_filter": {
                    "filter_family": "market_volume_confirmation",
                    "source_layer": "market",
                },
                "volatility_filter": {
                    "filter_family": "volatility_regime_filter",
                    "source_layer": "market",
                },
                "funding_filter": {
                    "filter_family": "funding_confirmation",
                    "source_layer": "futures",
                },
                "open_interest_filter": {
                    "filter_family": "open_interest_confirmation",
                    "source_layer": "futures",
                },
                "sentiment_filter": {
                    "filter_family": "sentiment_veto_or_confirmation",
                    "source_layer": "sentiment",
                },
                "onchain_filter": {
                    "filter_family": "onchain_veto_or_confirmation",
                    "source_layer": "onchain",
                },
            },
        },
    }


def get_default_custom_selection():
    """Return a stable default custom builder selection."""
    return {
        "core_signal": "trend_following",
        "timeframes": ["4h"],
        "data_sources": ["market"],
        "confirmation_filters": [],
        "decision_rules": "double_confirmation",
        "risk_profile": "balanced_risk",
    }


def get_strategy_registry():
    """Return the preset strategy catalog and custom builder schema."""
    presets = _build_preset_strategies()
    default_selection = get_default_custom_selection()
    return {
        "preset_strategies": presets,
        "timeframe_catalogs": {
            timeframe: get_timeframe_strategy_catalog(timeframe)
            for timeframe in SUPPORTED_STRATEGY_TIMEFRAMES
        },
        "timeframe_architecture": FINAL_TIMEFRAME_STRATEGY_ARCHITECTURE,
        "custom_builder": _build_custom_builder(),
        "default_custom_selection": default_selection,
        "preset_blueprints": {
            preset["id"]: preset["backend_blueprint"]
            for preset in presets
            if "backend_blueprint" in preset
        },
        "default_custom_blueprint": build_custom_strategy_blueprint(default_selection),
    }


def get_preset_strategy(preset_id):
    """Return one preset strategy definition by id."""
    for preset in _build_preset_strategies():
        if preset["id"] == preset_id:
            return preset
    return None


def get_timeframe_strategy_catalog(timeframe):
    """Return only the strategies that are intentionally offered for one timeframe."""
    timeframe = _normalize_timeframe(timeframe)
    rows = []
    for preset in _build_preset_strategies():
        supported = timeframe in preset.get("supported_timeframes", [])
        rows.append(
            {
                "id": preset["id"],
                "name": preset["name"],
                "supported": supported,
                "preferred": timeframe in preset.get("preferred_timeframes", []),
                "timeframe_role": preset.get("timeframe_role", "general"),
                "deployment_role": preset.get("deployment_role", "research_comparison"),
                "core_engine": preset.get("core_engine", "Market"),
                "risk_profile": preset.get("risk_profile", "Balanced"),
                "distinct_engine_status": preset.get("distinct_engine_status", "mapped_to_shared_evaluated_family"),
            }
        )
    return {
        "timeframe": timeframe,
        "label": FINAL_TIMEFRAME_STRATEGY_ARCHITECTURE.get(timeframe, {}).get("label", timeframe),
        "summary": FINAL_TIMEFRAME_STRATEGY_ARCHITECTURE.get(timeframe, {}).get("summary", ""),
        "strategies": [row for row in rows if row["supported"]],
    }


def resolve_preset_timeframe(preset, requested_timeframe):
    """Resolve one preset against its intentionally supported timeframes."""
    requested = _normalize_timeframe(requested_timeframe)
    supported = list(preset.get("supported_timeframes", ["4h"]))
    preferred = list(preset.get("preferred_timeframes", supported[:1] or ["4h"]))
    if requested in supported:
        return {
            "requested_timeframe": requested,
            "resolved_timeframe": requested,
            "requested_timeframe_supported": True,
            "timeframe_fit_label": "direct_support",
        }

    fallback_timeframe = preferred[0] if preferred else supported[0]
    return {
        "requested_timeframe": requested,
        "resolved_timeframe": fallback_timeframe,
        "requested_timeframe_supported": False,
        "timeframe_fit_label": "catalog_fallback",
    }


def build_custom_strategy_blueprint(selection):
    """Translate user builder choices into a backend strategy blueprint."""
    schema = _build_custom_builder()["mapping_schema"]
    default_selection = get_default_custom_selection()

    merged_selection = {
        "core_signal": selection.get("core_signal", default_selection["core_signal"]),
        "timeframes": _resolve_custom_timeframes(selection, default=default_selection["timeframes"][0]),
        "data_sources": selection.get("data_sources", default_selection["data_sources"]),
        "confirmation_filters": selection.get(
            "confirmation_filters",
            default_selection["confirmation_filters"],
        ),
        "decision_rules": selection.get("decision_rules", default_selection["decision_rules"]),
        "risk_profile": selection.get("risk_profile", default_selection["risk_profile"]),
    }

    core_mapping = schema["core_signal_map"][merged_selection["core_signal"]]
    decision_mapping = schema["decision_rule_map"][merged_selection["decision_rules"]]
    risk_mapping = schema["risk_profile_map"][merged_selection["risk_profile"]]
    timeframe_policy = build_timeframe_strategy_policy(
        requested_timeframe=merged_selection["timeframes"][0],
        resolved_timeframe=merged_selection["timeframes"][0],
        selected_timeframes=merged_selection["timeframes"],
    )
    data_source_mappings = [
        schema["data_source_map"][source_id]
        for source_id in merged_selection["data_sources"]
        if source_id in schema["data_source_map"]
    ]
    confirmation_mappings = [
        schema["confirmation_filter_map"][filter_id]
        for filter_id in merged_selection["confirmation_filters"]
        if filter_id in schema["confirmation_filter_map"]
    ]

    return {
        "core_signal": merged_selection["core_signal"],
        "timeframes": merged_selection["timeframes"],
        "timeframe_policy": timeframe_policy,
        "required_layers": [source_id for source_id in merged_selection["data_sources"] if source_id == "market"],
        "optional_layers": [source_id for source_id in merged_selection["data_sources"] if source_id != "market"],
        "target_family": timeframe_policy["target_family_hint"] if timeframe_policy["timeframe_scope"] == "multi_timeframe" else core_mapping["target_family"],
        "model_family": core_mapping["model_family"],
        "policy_family": (
            "multi_timeframe_confirmation_stack"
            if timeframe_policy["timeframe_scope"] == "multi_timeframe"
            else core_mapping["policy_family"]
        ),
        "probability_mode": risk_mapping["probability_mode"],
        "confirmation_mode": decision_mapping["confirmation_mode"],
        "minimum_confirmations": decision_mapping["minimum_confirmations"],
        "threshold_profile": risk_mapping["threshold_profile"],
        "exposure_style": risk_mapping["exposure_style"],
        "feature_focus": core_mapping["feature_focus"],
        "data_source_roles": data_source_mappings,
        "confirmation_filters": confirmation_mappings,
        "sentiment_role": (
            "optional_confirmation_veto"
            if any(source["backend_input"] == "sentiment_context_features" for source in data_source_mappings)
            or any(filter_config["source_layer"] == "sentiment" for filter_config in confirmation_mappings)
            else "not_selected"
        ),
        "onchain_role": (
            "optional_confirmation_veto"
            if any(source["backend_input"] == "onchain_context_features" for source in data_source_mappings)
            or any(filter_config["source_layer"] == "onchain" for filter_config in confirmation_mappings)
            else "not_selected"
        ),
    }


def resolve_preset_strategy_config(preset_id, asset, market_summary=None, requested_timeframe="4h", resolved_timeframe=None):
    """Resolve a preset into a concrete backend strategy config."""
    preset = get_preset_strategy(preset_id) or get_preset_strategy("recommended")
    market_summary = market_summary or {}
    capabilities = build_asset_capability_state(asset, market_summary)
    backend_blueprint = dict(preset.get("backend_blueprint", {}))
    family_governance = build_family_governance(market_summary, "market_futures_backend")
    operational_family = family_governance.get("operational_family", "market_only_benchmark")
    timeframe_resolution = resolve_preset_timeframe(preset, requested_timeframe)
    effective_resolved_timeframe = (
        timeframe_resolution["resolved_timeframe"]
        if not timeframe_resolution["requested_timeframe_supported"]
        else (
            requested_timeframe
            if _normalize_timeframe(requested_timeframe) == "1d"
            else (resolved_timeframe or timeframe_resolution["resolved_timeframe"])
        )
    )
    timeframe_policy = build_timeframe_strategy_policy(
        requested_timeframe=requested_timeframe,
        resolved_timeframe=effective_resolved_timeframe,
        selected_timeframes=[effective_resolved_timeframe],
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

    if preset["id"] == "recommended":
        if operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}:
            backend_blueprint.update(
                {
                    "target_family": timeframe_policy["target_family_hint"] if timeframe_policy["resolved_timeframe"] != "1d" else "future_return_bucket",
                    "model_family": market_summary.get("benchmark_best_model_name", "scaled_market_baseline"),
                    "policy_family": "classification_only",
                    "probability_mode": "benchmark_classification",
                    "confirmation_mode": "market_only_benchmark",
                    "required_layers": ["market"],
                    "optional_layers": [],
                }
            )
        elif operational_family == "intraday_binary_candidate":
            backend_blueprint.update(
                {
                    "target_family": market_summary.get("binary_backbone_target_name", timeframe_policy["target_family_hint"]),
                    "model_family": market_summary.get("binary_backbone_deployed_model_name", market_summary.get("backbone_best_model_name", "market_futures_binary_directional")),
                    "policy_family": market_summary.get("binary_backbone_policy_name", "binary_confidence_gated_long_flat"),
                    "probability_mode": market_summary.get("binary_backbone_probability_mode", "raw"),
                    "confirmation_mode": "intraday_binary_candidate",
                    "required_layers": ["market", "futures"],
                    "optional_layers": [],
                }
            )
        elif operational_family == "binary_directional_fallback":
            backend_blueprint.update(
                {
                    "target_family": market_summary.get("binary_backbone_target_name", market_summary.get("backbone_target_name", "asset_selected_preferred_target")),
                    "model_family": market_summary.get("binary_backbone_deployed_model_name", market_summary.get("backbone_best_model_name", "market_futures_binary_directional")),
                    "policy_family": market_summary.get("binary_backbone_policy_name", "binary_confidence_gated_long_flat"),
                    "probability_mode": market_summary.get("binary_backbone_probability_mode", "raw"),
                    "confirmation_mode": "binary_directional_fallback",
                    "required_layers": ["market", "futures"],
                    "optional_layers": ["sentiment", "onchain"],
                }
            )
        else:
            backend_blueprint.update(
                {
                    "target_family": market_summary.get("selected_target_name", timeframe_policy["target_family_hint"]),
                    "model_family": market_summary.get("selected_primary_model", market_summary.get("selected_backend_model", "selected_market_futures_backend")),
                    "policy_family": market_summary.get("policy_name", "selected_policy_family"),
                    "probability_mode": market_summary.get("probability_mode", "selected_probability_mode"),
                }
            )
    elif preset["id"] == "daily_structural_confirmation":
        backend_blueprint.update(
            {
                "target_family": "daily_structural_confirmation",
                "model_family": "daily_structural_confirmation_family",
                "policy_family": "structural_confirmation_only",
                "probability_mode": "research_scorecard",
                "confirmation_mode": "daily_structural_confirmation",
                "required_layers": ["market"],
                "optional_layers": ["onchain", "sentiment"],
            }
        )
    elif preset["id"] == "conservative_trend" and daily_structural_label in {"distribution_risk", "divergence_watch"}:
        backend_blueprint.update(
            {
                "confirmation_mode": "strict_consensus",
                "policy_family": "regime_adaptive_long_flat",
            }
        )
    elif preset["id"] == "conservative_trend" and daily_confirmation_mode == "supportive_narrow":
        backend_blueprint.update(
            {
                "confirmation_mode": "strict_consensus",
            }
        )
    elif preset["id"] == "multimodal_balanced" and daily_posture_label == "structural_supportive" and daily_structural_confidence >= 0.55:
        backend_blueprint.update(
            {
                "confirmation_mode": "double_confirmation",
                "optional_layers": ["sentiment", "onchain"],
            }
        )
    elif preset["id"] == "multimodal_balanced" and daily_structural_label in {"distribution_risk", "divergence_watch"}:
        backend_blueprint.update(
            {
                "confirmation_mode": "strict_consensus",
                "optional_layers": ["onchain", "sentiment"],
            }
        )
    elif preset["id"] == "futures_crowd_reversal" and daily_confirmation_mode == "supportive_narrow":
        backend_blueprint.update(
            {
                "confirmation_mode": "reversal_research_secondary",
            }
        )
    elif preset["id"] == "futures_crowd_reversal" and daily_confirmation_mode == "cautionary":
        backend_blueprint.update(
            {
                "confirmation_mode": "reversal_with_structural_caution",
            }
        )

    required_layers, optional_layers, unavailable_layers = adapt_layers_to_capabilities(
        backend_blueprint.get("required_layers", ["market"]),
        backend_blueprint.get("optional_layers", []),
        capabilities,
    )

    config = {
        "mode": "preset",
        "strategy_id": preset["id"],
        "strategy_name": preset["name"],
        "strategy_family": preset.get("strategy_family", "structured_preset"),
        "asset": asset,
        "display_summary": f"{preset['name']} for {asset}",
        "timeframe_policy": timeframe_policy,
        "supported_timeframes": preset.get("supported_timeframes", ["1h", "4h"]),
        "preferred_timeframes": preset.get("preferred_timeframes", ["4h"]),
        "requested_timeframe_supported": timeframe_resolution["requested_timeframe_supported"],
        "timeframe_fit_label": timeframe_resolution["timeframe_fit_label"],
        "core_engine": (
            "Market"
            if preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}
            else (
                "Binary market + futures"
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else (
                    "Daily structural context"
                    if preset["id"] == "daily_structural_confirmation"
                    else preset.get("core_engine", "Market")
                )
            )
        ),
        "risk_profile": preset.get("risk_profile", "Balanced"),
        "best_for": preset.get("best_for", ""),
        "daily_confirmation_fit": (
            "supportive"
            if daily_confirmation_mode in {"supportive_broad", "supportive_narrow"}
            else (
                "cautious"
                if daily_confirmation_mode == "cautionary" or daily_structural_confidence < 0.35
                else "mixed"
            )
        ),
        "daily_confirmation_mode": daily_confirmation_mode,
        "distinct_engine_status": preset.get("distinct_engine_status", "mapped_to_shared_evaluated_family"),
        "preset_evaluation_basis": preset.get("evaluation_basis", "mapped_benchmark_reference"),
        "daily_posture_label": daily_posture_label,
        "daily_structural_label": daily_structural_label,
        "daily_structural_confidence": daily_structural_confidence,
        "required_layers": required_layers,
        "optional_layers": optional_layers,
        "unavailable_layers": unavailable_layers,
        "asset_capabilities": capabilities,
        "capability_notes": build_capability_notes(capabilities),
        "target_family": backend_blueprint.get("target_family", timeframe_policy["target_family_hint"]),
        "model_family": backend_blueprint.get("model_family", "logistic_market_core"),
        "policy_family": backend_blueprint.get("policy_family", "confidence_gated_long_flat"),
        "probability_mode": backend_blueprint.get("probability_mode", "temperature_scaled"),
        "confirmation_mode": backend_blueprint.get("confirmation_mode", "double_confirmation"),
        "latest_signal": (
            market_summary.get("benchmark_latest_prediction")
            if preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}
            else (
                market_summary.get("daily_posture_label", market_summary.get("combined_daily_view", "structural_mixed"))
                if preset["id"] == "daily_structural_confirmation"
                else (
                market_summary.get("binary_backbone_latest_signal", market_summary.get("backbone_best_latest_signal", market_summary.get("latest_signal")))
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else market_summary.get("selected_primary_signal", market_summary.get("latest_signal"))
                )
            )
        ),
        "latest_signal_confidence": (
            market_summary.get("benchmark_prediction_confidence", market_summary.get("selected_primary_confidence", market_summary.get("latest_signal_confidence")))
            if preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}
            else (
                market_summary.get("daily_confirmation_score", 0.0)
                if preset["id"] == "daily_structural_confirmation"
                else (
                market_summary.get("binary_backbone_latest_signal_confidence", market_summary.get("backbone_best_latest_confidence", market_summary.get("latest_signal_confidence")))
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else market_summary.get(
                    "selected_primary_confidence",
                    market_summary.get("latest_signal_confidence"),
                )
                )
            )
        ),
        "test_accuracy": (
            market_summary.get("benchmark_best_accuracy")
            if preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}
            else (
                market_summary.get("daily_specialist_accuracy", 0.0)
                if preset["id"] == "daily_structural_confirmation"
                else (
                market_summary.get("binary_backbone_best_accuracy", market_summary.get("backbone_best_accuracy"))
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else market_summary.get("test_accuracy")
                )
            )
        ),
        "test_macro_f1": (
            market_summary.get("benchmark_best_macro_f1")
            if preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"}
            else (
                market_summary.get("daily_specialist_macro_f1", 0.0)
                if preset["id"] == "daily_structural_confirmation"
                else (
                market_summary.get("binary_backbone_best_macro_f1", market_summary.get("backbone_best_macro_f1"))
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else market_summary.get("test_macro_f1")
                )
            )
        ),
        "walkforward_avg_excess_return": (
            0.0
            if preset["id"] in {"daily_structural_confirmation"} or (preset["id"] == "recommended" and operational_family in {"market_only_benchmark", "reduced_trust_benchmark", "daily_structural_research"})
            else (
                market_summary.get("binary_backbone_excess_return", 0.0)
                if preset["id"] == "recommended" and operational_family in {"binary_directional_fallback", "intraday_binary_candidate"}
                else market_summary.get("walkforward_avg_excess_return")
            )
        ),
        "evaluation_basis_note": (
            "This preset adapts to the asset's current lead family. For this asset, the market-only benchmark currently has the cleaner evidence base."
            if preset["id"] == "recommended" and operational_family == "market_only_benchmark"
            else (
                "This preset is temporarily falling back to the market-only benchmark because structural-break diagnostics are reducing trust in the learned backbone relationships."
                if preset["id"] == "recommended" and operational_family == "reduced_trust_benchmark"
                else (
                    "This preset is using the intraday binary route because that family is currently the cleaner 1h execution candidate for this asset/timeframe."
                    if preset["id"] == "recommended" and operational_family == "intraday_binary_candidate"
                    else (
                "This preset is using the daily path as an extra check rather than the main signal."
                        if preset["id"] == "recommended" and operational_family == "daily_structural_research"
                        else (
                            "This preset is using the binary directional fallback because the three-class backbone is collapsing into a one-sided posture for this asset/timeframe."
                            if preset["id"] == "recommended" and operational_family == "binary_directional_fallback"
                            else (
                                "This preset uses daily evidence to support or question the faster strategy, rather than acting alone."
                                if preset["id"] == "daily_structural_confirmation"
                                else (
                                    "This preset keeps context layers in a supporting role and should only be trusted when sentiment and on-chain reliability are strong enough to justify extra confirmation."
                                    if preset["id"] == "multimodal_balanced"
                                    else (
                                        "This preset remains useful as a reversal-style research comparison, but daily support is currently narrower than broad structural participation."
                                        if preset["id"] == "futures_crowd_reversal" and daily_confirmation_mode == "supportive_narrow"
                                        else (
                                        "This preset uses the asset's currently selected backend engine and latest saved evaluation outputs."
                                        if preset["id"] == "recommended"
                                        else "This preset is a structured product-facing configuration mapped onto an evaluated backend family rather than a separately trained named engine."
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "timeframe_note": (
            f"{timeframe_policy['timeframe_label']} mode is active on {timeframe_policy['resolved_timeframe']}. "
            f"Policy style is {timeframe_policy['policy_bias'].replace('_', ' ')}, and backup policy is {timeframe_policy['fallback_policy'].replace('_', ' ')}."
            + (
                f" Requested timeframe {requested_timeframe} is not part of this preset's intentional catalog, so LiveStrat resolves it through {timeframe_policy['resolved_timeframe']}."
                if not timeframe_resolution["requested_timeframe_supported"]
                else ""
            )
            + (
                f" Daily state is {daily_posture_label.replace('_', ' ')} "
                f"with {daily_structural_label.replace('_', ' ')} and confidence {daily_structural_confidence:.2f}."
                if timeframe_policy["resolved_timeframe"] in {"4h", "1d"}
                else ""
            )
            + (
                " This preset currently fits the daily evidence."
                if timeframe_policy["resolved_timeframe"] in {"4h", "1d"} and daily_confirmation_mode == "supportive_broad"
                else (
                    " This preset currently has narrower, valuation-led daily support rather than broad participation support."
                    if timeframe_policy["resolved_timeframe"] in {"4h", "1d"} and daily_confirmation_mode == "supportive_narrow"
                    else (
                    " This preset should stay conservative because the daily evidence is cautionary."
                    if timeframe_policy["resolved_timeframe"] in {"4h", "1d"} and daily_confirmation_mode == "cautionary"
                    else ""
                    )
                )
            )
            + _derive_futures_support_note(market_summary)
        ),
        "family_governance": family_governance,
    }
    config["scorecard"] = build_strategy_scorecard(config, market_summary)
    config["governance"] = build_config_governance(config, market_summary)
    config["implementation_status"] = config["governance"]["readiness_label"].lower().replace(" ", "_")
    return config


def resolve_custom_strategy_config(selection, asset, market_summary=None, requested_timeframe="4h", resolved_timeframe=None):
    """Resolve a custom builder selection into a backend strategy config."""
    market_summary = market_summary or {}
    blueprint = build_custom_strategy_blueprint(selection)
    capabilities = build_asset_capability_state(asset, market_summary)
    merged_selection = {
        "core_signal": selection.get("core_signal", get_default_custom_selection()["core_signal"]),
        "timeframes": _resolve_custom_timeframes(selection, default=requested_timeframe),
        "data_sources": selection.get("data_sources", get_default_custom_selection()["data_sources"]),
        "confirmation_filters": selection.get(
            "confirmation_filters",
            get_default_custom_selection()["confirmation_filters"],
        ),
        "decision_rules": selection.get("decision_rules", get_default_custom_selection()["decision_rules"]),
        "risk_profile": selection.get("risk_profile", get_default_custom_selection()["risk_profile"]),
    }
    timeframe_policy = build_timeframe_strategy_policy(
        requested_timeframe=requested_timeframe,
        resolved_timeframe=resolved_timeframe or requested_timeframe,
        selected_timeframes=merged_selection["timeframes"],
    )

    required_layers, optional_layers, unavailable_layers = adapt_layers_to_capabilities(
        blueprint["required_layers"],
        blueprint["optional_layers"],
        capabilities,
    )
    sentiment_role = blueprint["sentiment_role"]
    if sentiment_role != "not_selected":
        if capabilities["gdelt_asset_news_available"]:
            sentiment_role = "asset_specific_confirmation"
        elif capabilities["effective_sentiment_available"]:
            sentiment_role = "market_wide_fallback_confirmation"
        else:
            sentiment_role = "unavailable"

    onchain_role = blueprint["onchain_role"]
    if onchain_role != "not_selected" and not capabilities["onchain_available"]:
        onchain_role = "unavailable"

    config = {
        "mode": "custom",
        "strategy_id": f"custom_{asset.lower()}_{blueprint['core_signal']}",
        "strategy_name": "Custom Strategy",
        "strategy_family": "custom_builder_blueprint",
        "asset": asset,
        "display_summary": f"Custom strategy for {asset}",
        "selection": merged_selection,
        "timeframe_policy": timeframe_policy,
        "required_layers": required_layers,
        "optional_layers": optional_layers,
        "unavailable_layers": unavailable_layers,
        "asset_capabilities": capabilities,
        "capability_notes": build_capability_notes(capabilities),
        "target_family": timeframe_policy["target_family_hint"] if timeframe_policy["timeframe_scope"] == "multi_timeframe" else blueprint["target_family"],
        "model_family": blueprint["model_family"],
        "policy_family": blueprint["policy_family"],
        "probability_mode": blueprint["probability_mode"],
        "confirmation_mode": blueprint["confirmation_mode"],
        "minimum_confirmations": blueprint["minimum_confirmations"],
        "threshold_profile": blueprint["threshold_profile"],
        "exposure_style": blueprint["exposure_style"],
        "feature_focus": blueprint["feature_focus"],
        "confirmation_filters": blueprint["confirmation_filters"],
        "sentiment_role": sentiment_role,
        "onchain_role": onchain_role,
        "benchmark_reference_model": market_summary.get("selected_primary_model", market_summary.get("selected_backend_model")),
        "benchmark_reference_accuracy": market_summary.get("test_accuracy"),
        "benchmark_reference_macro_f1": market_summary.get("test_macro_f1"),
        "distinct_engine_status": "mapped_to_shared_evaluated_family",
        "preset_evaluation_basis": "mapped_benchmark_reference",
        "evaluation_basis_note": (
            "This custom strategy is currently a resolved backend blueprint benchmarked against the closest evaluated family, not a separately trained custom engine."
        ),
        "timeframe_note": (
            f"Custom strategy is configured for {', '.join(merged_selection['timeframes'])}. "
            f"Primary resolved timeframe is {timeframe_policy['resolved_timeframe']}, with backup policy {timeframe_policy['fallback_policy'].replace('_', ' ')}."
            + _derive_futures_support_note(market_summary)
        ),
    }
    config["scorecard"] = build_strategy_scorecard(config, market_summary)
    config["governance"] = build_config_governance(config, market_summary)
    config["implementation_status"] = config["governance"]["readiness_label"].lower().replace(" ", "_")
    return config


def _count_selected_modalities(config):
    """Count total selected data layers across required and optional sets."""
    required_layers = config.get("required_layers", [])
    optional_layers = config.get("optional_layers", [])
    return len(set(required_layers + optional_layers))


def _resolve_benchmark_metrics(config, market_summary):
    """Choose the most honest available benchmark reference for one strategy config."""
    market_summary = market_summary or {}
    model_family = str(config.get("model_family", ""))
    required_layers = config.get("required_layers", [])
    optional_layers = config.get("optional_layers", [])
    uses_multimodal_context = any(layer in ["sentiment", "onchain"] for layer in required_layers + optional_layers)

    if (
        "selected_market_futures_backend" in model_family
        or "market_futures" in model_family
        or "futures" in required_layers
        or "futures" in optional_layers
        or uses_multimodal_context
    ):
        return {
            "benchmark_source": "market_futures_backend",
            "reference_accuracy": _safe_float(market_summary.get("backbone_best_accuracy", market_summary.get("test_accuracy", 0.0))),
            "reference_macro_f1": _safe_float(market_summary.get("backbone_best_macro_f1", market_summary.get("test_macro_f1", 0.0))),
            "reference_balanced_accuracy": _safe_float(market_summary.get("backbone_best_balanced_accuracy", market_summary.get("test_balanced_accuracy", 0.0))),
            "reference_excess_return": _safe_float(market_summary.get("backbone_walkforward_avg_excess_return", market_summary.get("walkforward_avg_excess_return", 0.0))),
            "reference_model": market_summary.get(
                "backbone_deployed_model_name",
                market_summary.get("selected_primary_model", market_summary.get("selected_backend_model", "market_futures_backend")),
            ),
            "reference_summary": market_summary.get("primary_summary", market_summary.get("backend_summary", "")),
            "reference_metric_coverage": "full",
        }

    return {
        "benchmark_source": "market_only_scaled_baseline",
        "reference_accuracy": _safe_float(market_summary.get("benchmark_best_accuracy", market_summary.get("baseline_scaled_test_accuracy", 0.0))),
        "reference_macro_f1": _safe_float(market_summary.get("benchmark_best_macro_f1", 0.0)),
        "reference_balanced_accuracy": _safe_float(market_summary.get("benchmark_best_balanced_accuracy", 0.0)),
        "reference_excess_return": 0.0,
        "reference_model": market_summary.get("benchmark_best_model_name", "scaled_market_baseline"),
        "reference_summary": market_summary.get("analysis_summary", ""),
        "reference_metric_coverage": "full" if market_summary.get("benchmark_best_macro_f1") not in (None, "") else "accuracy_only",
    }


def build_strategy_scorecard(config, market_summary=None):
    """Build an honest scorecard for one resolved strategy configuration."""
    market_summary = market_summary or {}
    benchmark = _resolve_benchmark_metrics(config, market_summary)
    modality_count = _count_selected_modalities(config)
    confirmation_filters = config.get("confirmation_filters", [])
    confirmation_count = len(confirmation_filters) if isinstance(confirmation_filters, list) else 0
    confirmation_mode = str(config.get("confirmation_mode", "double_confirmation"))
    model_family = str(config.get("model_family", ""))
    probability_mode = str(config.get("probability_mode", "temperature_scaled"))

    interpretability_score = 5
    if "lstm" in model_family or "sequence" in model_family:
        interpretability_score = 2
    elif "forest" in model_family:
        interpretability_score = 3
    elif "logistic" in model_family:
        interpretability_score = 5

    academic_depth_score = min(5, 2 + max(modality_count - 1, 0) + min(confirmation_count, 2))
    implementation_complexity_score = min(
        5,
        1 + max(modality_count - 1, 0) + min(confirmation_count, 2) + (1 if confirmation_mode == "strict_consensus" else 0),
    )
    live_readiness_score = max(
        1,
        min(
            5,
            4
            - max(modality_count - 2, 0)
            - (1 if confirmation_mode == "strict_consensus" else 0)
            + (1 if probability_mode == "temperature_scaled" else 0),
        ),
    )

    evaluation_mode = "direct_backend_reference" if config.get("mode") == "preset" and config.get("strategy_id") == "recommended" else "mapped_benchmark_reference"
    unavailable_layers = config.get("unavailable_layers", [])
    reference_metric_coverage = benchmark.get("reference_metric_coverage", "full")
    capability_penalty_note = (
        f" Current unavailable layers: {', '.join(unavailable_layers)}."
        if unavailable_layers else
        ""
    )
    metric_sentence = (
        f"Reference accuracy is {benchmark['reference_accuracy'] * 100:.1f}%, "
        f"macro-F1 is {benchmark['reference_macro_f1'] * 100:.1f}%, and walk-forward excess return is "
        f"{benchmark['reference_excess_return'] * 100:.1f}%."
        if reference_metric_coverage == "full"
        else (
            f"Reference accuracy is {benchmark['reference_accuracy'] * 100:.1f}%. "
            "Macro-F1, balanced accuracy, and walk-forward return are not exposed for this mapped baseline in the current summary row."
        )
    )
    scorecard_summary = (
        f"{config.get('strategy_name', 'Strategy')} for {config.get('asset', 'n/a')} is currently tied to the "
        f"{benchmark['benchmark_source']} benchmark. {metric_sentence} "
        f"This configuration scores {academic_depth_score}/5 for academic depth, {interpretability_score}/5 for interpretability, "
        f"and {implementation_complexity_score}/5 for implementation complexity.{capability_penalty_note}"
    )

    return {
        "evaluation_mode": evaluation_mode,
        "benchmark_source": benchmark["benchmark_source"],
        "reference_model": benchmark["reference_model"],
        "reference_accuracy": benchmark["reference_accuracy"],
        "reference_macro_f1": benchmark["reference_macro_f1"],
        "reference_balanced_accuracy": benchmark["reference_balanced_accuracy"],
        "reference_excess_return": benchmark["reference_excess_return"],
        "reference_metric_coverage": reference_metric_coverage,
        "interpretability_score": interpretability_score,
        "academic_depth_score": academic_depth_score,
        "implementation_complexity_score": implementation_complexity_score,
        "live_readiness_score": live_readiness_score,
        "selected_modality_count": modality_count,
        "selected_confirmation_count": confirmation_count,
        "scorecard_summary": scorecard_summary,
        "reference_summary": benchmark["reference_summary"],
    }
