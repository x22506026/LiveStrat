"""Scan processed evaluation files and summarize real strategy coverage by asset and timeframe."""

import csv
from collections import defaultdict
from pathlib import Path


SUPPORTED_FAMILIES = (
    "market_trend_benchmark",
    "market_futures",
    "multimodal",
    "onchain_specialist",
    "cross_asset_relative_strength",
    "market_baseline_scaled",
    "market_baseline_unscaled",
    "rule_based",
)


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _family_from_model_name(model_name):
    name = str(model_name or "")
    if "market_multimodal" in name:
        return "multimodal"
    if "market_onchain" in name:
        return "onchain_specialist"
    if "baseline_scaled" in name:
        return "market_baseline_scaled"
    if "baseline_unscaled" in name:
        return "market_baseline_unscaled"
    if "rule_based" in name:
        return "rule_based"
    if "market_trend" in name:
        return "market_trend_benchmark"
    if "market_futures" in name:
        return "market_futures"
    return "other"


def _normalize_symbol(symbol):
    value = str(symbol or "").strip().upper()
    return value if value.endswith("USDT") else f"{value}USDT"


def _initialize_family_summary():
    return {
        "available_timeframes": set(),
        "models_tested": 0,
        "best_model_name": None,
        "best_accuracy": 0.0,
        "best_macro_f1": 0.0,
        "best_balanced_accuracy": 0.0,
    }


def _update_family_summary(family_summary, timeframe, model_name, accuracy=0.0, macro_f1=0.0, balanced_accuracy=0.0):
    family_summary["available_timeframes"].add(timeframe)
    family_summary["models_tested"] += 1

    current_best_tuple = (
        family_summary["best_macro_f1"],
        family_summary["best_accuracy"],
        family_summary["best_balanced_accuracy"],
    )
    candidate_tuple = (macro_f1, accuracy, balanced_accuracy)

    if candidate_tuple > current_best_tuple:
        family_summary["best_model_name"] = model_name
        family_summary["best_accuracy"] = accuracy
        family_summary["best_macro_f1"] = macro_f1
        family_summary["best_balanced_accuracy"] = balanced_accuracy


def _add_cross_asset_relative_strength_coverage(processed_dir, assets):
    for summary_path in processed_dir.glob("cross_asset_relative_strength_summary_*.csv"):
        with open(summary_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                asset = _normalize_symbol(row.get("symbol"))
                timeframe = row.get("timeframe")
                hit_rate = _safe_float(row.get("top_pick_hit_rate"))
                _update_family_summary(
                    assets[asset]["cross_asset_relative_strength"],
                    timeframe,
                    "cross_sectional_relative_strength_ranker",
                    accuracy=hit_rate,
                    macro_f1=hit_rate,
                    balanced_accuracy=hit_rate,
                )


def build_evaluation_coverage_snapshot(processed_dir):
    """Return real evaluated coverage from generated metrics files."""
    processed_dir = Path(processed_dir)
    metrics_by_model = defaultdict(lambda: defaultdict(dict))

    for metrics_path in processed_dir.glob("*evaluation_metrics*.csv"):
        with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip().split(",")
            if header != ["model_name", "symbol", "timeframe", "metric", "value"]:
                continue
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) != 5:
                    continue
                model_name, symbol, timeframe, metric, value = parts
                asset_key = _normalize_symbol(symbol)
                metrics_by_model[(asset_key, timeframe, model_name)][metric] = _safe_float(value)

    assets = defaultdict(lambda: {family: _initialize_family_summary() for family in SUPPORTED_FAMILIES})

    for (asset, timeframe, model_name), metrics in metrics_by_model.items():
        family = _family_from_model_name(model_name)
        if family not in SUPPORTED_FAMILIES:
            continue

        _update_family_summary(
            assets[asset][family],
            timeframe,
            model_name,
            accuracy=_safe_float(metrics.get("accuracy", 0)),
            macro_f1=_safe_float(metrics.get("macro_f1", 0)),
            balanced_accuracy=_safe_float(metrics.get("balanced_accuracy", 0)),
        )

    _add_cross_asset_relative_strength_coverage(processed_dir, assets)

    serializable_assets = {}
    for asset, family_map in assets.items():
        serializable_assets[asset] = {}
        for family, summary in family_map.items():
            serializable_assets[asset][family] = {
                **summary,
                "available_timeframes": sorted(summary["available_timeframes"]),
            }

    return {
        "assets": serializable_assets,
        "families": list(SUPPORTED_FAMILIES),
    }
