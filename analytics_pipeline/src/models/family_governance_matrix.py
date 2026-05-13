"""Build an asset-timeframe family-governance matrix from comparison outputs."""

from pathlib import Path

import pandas as pd


SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")
RECOMMENDATION_LABELS = {
    "promote_market_futures_backbone": "Market + futures backbone",
    "keep_market_only_as_benchmark_lead": "Market-only benchmark",
    "mixed_evidence_keep_both_visible": "Mixed evidence",
}


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _scan_comparison_files(processed_dir):
    processed_dir = Path(processed_dir)
    return sorted(processed_dir.glob("strategy_backbone_comparison_*.csv"))


def _label_strength(row):
    recommendation = str(row.get("recommended_family", "") or "")
    macro_uplift = _safe_float(row.get("walkforward_macro_f1_uplift"))
    excess_return = _safe_float(row.get("deployed_policy_excess_return"))

    if recommendation == "promote_market_futures_backbone":
        if macro_uplift > 0.05 or excess_return > 0.01:
            return "strong"
        return "moderate"
    if recommendation == "keep_market_only_as_benchmark_lead":
        if macro_uplift < -0.08:
            return "strong"
        return "moderate"
    return "mixed"


def _build_entry(row):
    recommendation = str(row.get("recommended_family", "mixed_evidence_keep_both_visible") or "mixed_evidence_keep_both_visible")
    backbone_family = str(row.get("backbone_best_family", "three_class") or "three_class")
    deployment_active = str(row.get("backbone_deployment_active", "")).strip().lower() == "true"
    concentration_flag = str(
        row.get("ternary_backbone_prediction_concentration_flag", "stable") or "stable"
    )
    return {
        "asset": row["symbol"],
        "timeframe": row["timeframe"],
        "lead_family": recommendation,
        "lead_family_label": RECOMMENDATION_LABELS.get(recommendation, "Mixed evidence"),
        "governance_strength": _label_strength(row),
        "backbone_family_type": backbone_family,
        "benchmark_model": row.get("benchmark_best_model_name"),
        "backbone_model": row.get("backbone_best_model_name"),
        "deployed_backbone_model": row.get("backbone_deployed_model_name"),
        "backbone_deployment_active": deployment_active,
        "prediction_concentration_flag": concentration_flag,
        "futures_completeness_label": row.get("backbone_futures_completeness_label"),
        "futures_context_resilience_label": row.get("backbone_futures_context_resilience_label"),
        "basis_feature_available": row.get("backbone_basis_feature_available"),
        "classification_macro_f1_uplift": _safe_float(row.get("classification_macro_f1_uplift")),
        "walkforward_macro_f1_uplift": _safe_float(row.get("walkforward_macro_f1_uplift")),
        "deployed_policy_excess_return": _safe_float(row.get("deployed_policy_excess_return")),
        "comparison_window_start": row.get("window_start"),
        "comparison_window_end": row.get("window_end"),
        "evidence_summary": row.get("comparison_summary") or "No comparison summary available.",
    }


def build_family_governance_matrix(processed_dir):
    """Build one matrix describing lead strategy family by asset and timeframe."""
    files = _scan_comparison_files(processed_dir)
    assets = {}
    timeframes = {
        timeframe: {
            "timeframe": timeframe,
            "assets": {},
            "lead_market_futures_assets": [],
            "lead_market_only_assets": [],
            "mixed_assets": [],
        }
        for timeframe in SUPPORTED_TIMEFRAMES
    }

    for file_path in files:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            entry = _build_entry(row)
            asset = entry["asset"]
            timeframe = entry["timeframe"]

            assets.setdefault(asset, {"asset": asset, "timeframes": {}})
            assets[asset]["timeframes"][timeframe] = entry
            timeframes.setdefault(
                timeframe,
                {
                    "timeframe": timeframe,
                    "assets": {},
                    "lead_market_futures_assets": [],
                    "lead_market_only_assets": [],
                    "mixed_assets": [],
                },
            )
            timeframes[timeframe]["assets"][asset] = entry

    for timeframe, timeframe_entry in timeframes.items():
        lead_market_futures_assets = []
        lead_market_only_assets = []
        mixed_assets = []

        for asset, asset_entry in timeframe_entry["assets"].items():
            if asset_entry["lead_family"] == "promote_market_futures_backbone":
                lead_market_futures_assets.append(asset)
            elif asset_entry["lead_family"] == "keep_market_only_as_benchmark_lead":
                lead_market_only_assets.append(asset)
            else:
                mixed_assets.append(asset)

        timeframe_entry["lead_market_futures_assets"] = sorted(lead_market_futures_assets)
        timeframe_entry["lead_market_only_assets"] = sorted(lead_market_only_assets)
        timeframe_entry["mixed_assets"] = sorted(mixed_assets)
        timeframe_entry["summary"] = (
            f"{timeframe} currently favors market + futures for {len(lead_market_futures_assets)} asset(s), "
            f"market-only for {len(lead_market_only_assets)} asset(s), and remains mixed for {len(mixed_assets)} asset(s)."
        )

    recommended_actions = []
    for timeframe in SUPPORTED_TIMEFRAMES:
        timeframe_entry = timeframes.get(timeframe, {})
        for asset in timeframe_entry.get("lead_market_only_assets", []):
            asset_entry = timeframe_entry["assets"][asset]
            futures_resilience = str(asset_entry.get("futures_context_resilience_label", "") or "")
            futures_completeness = str(asset_entry.get("futures_completeness_label", "") or "")
            basis_available = str(asset_entry.get("basis_feature_available", "")).strip().lower() == "true"
            if futures_resilience == "robust" and (futures_completeness == "partial" or not basis_available):
                recommended_actions.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "priority": "high",
                        "action": "improve_futures_coverage",
                        "reason": asset_entry["evidence_summary"],
                    }
                )
                continue
            recommended_actions.append(
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "priority": "high",
                    "action": "improve_market_futures_backbone",
                    "reason": asset_entry["evidence_summary"],
                }
            )
        for asset in timeframe_entry.get("mixed_assets", []):
            asset_entry = timeframe_entry["assets"][asset]
            futures_resilience = str(asset_entry.get("futures_context_resilience_label", "") or "")
            futures_completeness = str(asset_entry.get("futures_completeness_label", "") or "")
            basis_available = str(asset_entry.get("basis_feature_available", "")).strip().lower() == "true"
            if futures_resilience == "robust" and (futures_completeness == "partial" or not basis_available):
                recommended_actions.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "priority": "medium",
                        "action": "improve_futures_coverage",
                        "reason": asset_entry["evidence_summary"],
                    }
                )
                continue
            if asset_entry.get("prediction_concentration_flag") in {"hold_collapse", "dont_buy_collapse"}:
                recommended_actions.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "priority": "high",
                        "action": "enable_binary_fallback_or_reduce_trust",
                        "reason": asset_entry["evidence_summary"],
                    }
                )
                continue
            if asset_entry.get("backbone_family_type") == "binary_directional" and (
                not asset_entry.get("backbone_deployment_active")
                or asset_entry.get("deployed_policy_excess_return", 0.0) <= 0.0
            ):
                recommended_actions.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "priority": "high",
                        "action": "improve_binary_deployment_policy",
                        "reason": asset_entry["evidence_summary"],
                    }
                )
                continue
            recommended_actions.append(
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "priority": "medium",
                    "action": "collect_more_evidence",
                    "reason": timeframe_entry["assets"][asset]["evidence_summary"],
                }
            )

    return {
        "assets": assets,
        "timeframes": timeframes,
        "recommended_actions": recommended_actions,
    }
