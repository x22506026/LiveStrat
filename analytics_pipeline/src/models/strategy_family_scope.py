"""Describe which strategy families can fairly include each asset."""

from pathlib import Path

from src.config import ASSET_REGISTRY
from src.models.evaluation_coverage import build_evaluation_coverage_snapshot
from src.models.strategy_family_evidence import build_strategy_family_evidence_snapshot


CORE_ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
EXPANSION_ASSETS = ("BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
TIMEFRAMES = ("1h", "4h", "1d")
DECISION_TIMEFRAMES = ("1h", "4h")

FAMILY_DEFINITIONS = {
    "market_trend_benchmark": {
        "label": "Market trend benchmark",
        "required_layers": ["spot_market"],
        "optional_layers": [],
        "fair_comparison_scope": "all_market_assets",
        "defensible_use": "Baseline ML comparison across every asset with the same market feature recipe.",
    },
    "cross_asset_relative_strength": {
        "label": "Cross-asset relative strength",
        "required_layers": ["spot_market"],
        "optional_layers": [],
        "fair_comparison_scope": "all_market_assets_same_window",
        "defensible_use": "Portfolio-style ranking across assets on the same timeframe and date window.",
    },
    "market_futures_core": {
        "label": "Market + futures core",
        "required_layers": ["spot_market", "futures_structure"],
        "optional_layers": [],
        "fair_comparison_scope": "seven_asset_futures_with_timeframe_notes",
        "defensible_use": "Primary decision family where market and futures structure are available and evaluated per asset.",
    },
    "market_futures_binary": {
        "label": "Binary futures fallback",
        "required_layers": ["spot_market", "futures_structure"],
        "optional_layers": [],
        "fair_comparison_scope": "seven_asset_futures_with_timeframe_notes",
        "defensible_use": "Fallback/specialist long-flat family for noisy directional decisions.",
    },
    "context_confirmation": {
        "label": "Context confirmation",
        "required_layers": ["spot_market", "futures_structure"],
        "optional_layers": ["broad_sentiment", "asset_news", "onchain_daily"],
        "fair_comparison_scope": "context_validated_core_assets",
        "defensible_use": "Ablation layer that tests whether context improves the market/futures backbone.",
    },
    "multimodal_context": {
        "label": "Multimodal context model",
        "required_layers": ["spot_market", "futures_structure"],
        "optional_layers": ["broad_sentiment", "asset_news", "onchain_daily"],
        "fair_comparison_scope": "context_validated_core_assets",
        "defensible_use": "Research model that combines market, futures, sentiment/news, and on-chain context.",
    },
    "daily_structural_confirmation": {
        "label": "Daily structural confirmation",
        "required_layers": ["spot_market", "onchain_daily"],
        "optional_layers": ["broad_sentiment", "asset_news"],
        "fair_comparison_scope": "onchain_supported_core_assets",
        "defensible_use": "Daily confirmation overlay, not a primary intraday execution engine.",
    },
    "structural_break_governance": {
        "label": "Structural-break governance",
        "required_layers": ["spot_market"],
        "optional_layers": ["futures_structure", "asset_news", "onchain_daily"],
        "fair_comparison_scope": "assets_with_generated_diagnostics",
        "defensible_use": "Risk/governance layer for deciding when signals should be trusted less.",
    },
}


def _asset_code(symbol):
    for code, config in ASSET_REGISTRY.items():
        if config.get("market_symbol") == symbol:
            return code
    return symbol.replace("USDT", "")


def _asset_tier(symbol):
    if symbol in CORE_ASSETS:
        return "full_stack_core"
    if symbol in EXPANSION_ASSETS:
        return "market_first_expansion"
    return "unclassified"


def _family_rows(evidence_rows, family):
    return [row for row in evidence_rows if row.get("family") == family]


def _coverage_assets(coverage, family):
    assets = []
    for asset, family_map in coverage.get("assets", {}).items():
        section = family_map.get(family, {})
        if section.get("available_timeframes"):
            assets.append(asset)
    return sorted(assets)


def _defensibility_label(family, supported_assets):
    supported_set = set(supported_assets)
    core_set = set(CORE_ASSETS)
    expansion_set = set(EXPANSION_ASSETS)

    if family in {"market_trend_benchmark", "cross_asset_relative_strength"}:
        if core_set.union(expansion_set).issubset(supported_set):
            return "strong_for_7_asset_market_comparison"
        if len(supported_assets) >= 3:
            return "usable_market_comparison"
        return "insufficient_universe"

    if family in {"market_futures_core", "market_futures_binary", "context_confirmation", "multimodal_context"}:
        if family in {"market_futures_core", "market_futures_binary"} and core_set.union(expansion_set).issubset(supported_set):
            return "defensible_for_7_asset_futures_with_timeframe_limits"
        if core_set.issubset(supported_set) and not expansion_set.intersection(supported_set):
            return "defensible_for_core_assets_only"
        if core_set.issubset(supported_set):
            return "mixed_scope_needs_clear_asset_tiers"
        return "insufficient_core_coverage"

    if family == "daily_structural_confirmation":
        if {"BTCUSDT", "ETHUSDT"}.intersection(supported_set):
            return "research_only_structural_overlay"
        return "insufficient_onchain_coverage"

    if family == "structural_break_governance":
        return "governance_overlay_not_primary_strategy" if supported_assets else "missing_diagnostics"

    return "research_only"


def _required_next_steps(family, supported_assets):
    supported_set = set(supported_assets)
    missing_core = [asset for asset in CORE_ASSETS if asset not in supported_set]
    missing_expansion = [asset for asset in EXPANSION_ASSETS if asset not in supported_set]

    if family in {"market_trend_benchmark", "cross_asset_relative_strength"}:
        if missing_expansion:
            return [f"Generate market features and labels for {', '.join(missing_expansion)}."]
        return ["Keep the same date windows and feature recipe across all assets when comparing ranks or benchmark metrics."]

    if family in {"market_futures_core", "market_futures_binary"}:
        steps = []
        if missing_core:
            steps.append(f"Complete futures evaluation for core assets: {', '.join(missing_core)}.")
        if missing_expansion:
            steps.append(f"Complete futures evaluation for expansion assets: {', '.join(missing_expansion)}.")
        else:
            steps.append("Present 1h and 4h as the strongest rolling-validation futures windows across all seven assets.")
            steps.append("Present 1d as trained/backtested but limited for rolling validation until a longer daily futures history is pulled.")
        return steps

    if family in {"context_confirmation", "multimodal_context"}:
        return [
            "Keep context models as confirmation/research unless ablation evidence beats the market+futures backbone.",
            "Use broad sentiment as fallback and avoid claiming asset-news coverage where GDELT/RSS is missing.",
        ]

    if family == "daily_structural_confirmation":
        return [
            "Keep this family research-only unless on-chain source coverage is validated per asset.",
            "Do not force SOL or expansion coins into on-chain scoring when Coin Metrics coverage is unavailable.",
        ]

    if family == "structural_break_governance":
        return ["Use as a risk/governance overlay, not as a standalone forecasting strategy."]

    return ["Review generated evidence before exposing this family as user-facing."]


def build_strategy_family_scope_snapshot(project_dir):
    """Return a defensible asset-family scope map for report and UI use."""
    project_dir = Path(project_dir)
    processed_dir = project_dir / "analytics_pipeline" / "data" / "processed"
    coverage = build_evaluation_coverage_snapshot(processed_dir)
    evidence = build_strategy_family_evidence_snapshot(processed_dir)
    evidence_rows = evidence.get("rows", [])

    assets = {}
    for symbol in sorted(set(CORE_ASSETS).union(EXPANSION_ASSETS)):
        assets[symbol] = {
            "symbol": symbol,
            "asset_code": _asset_code(symbol),
            "tier": _asset_tier(symbol),
            "recommended_scope": (
                "full-stack strategy demo asset"
                if symbol in CORE_ASSETS else
                "market and futures expansion asset, context-limited"
            ),
        }

    families = {}
    for family, definition in FAMILY_DEFINITIONS.items():
        rows = _family_rows(evidence_rows, family)
        supported_assets = sorted({row.get("asset") for row in rows if row.get("asset")})
        coverage_supported_assets = _coverage_assets(coverage, family)
        if coverage_supported_assets:
            supported_assets = sorted(set(supported_assets).union(coverage_supported_assets))
        timeframes = sorted(
            {row.get("timeframe") for row in rows if row.get("timeframe")},
            key=lambda item: TIMEFRAMES.index(item) if item in TIMEFRAMES else item,
        )
        decision_timeframes = [timeframe for timeframe in timeframes if timeframe in DECISION_TIMEFRAMES]
        families[family] = {
            **definition,
            "supported_assets": supported_assets,
            "core_assets_supported": [asset for asset in CORE_ASSETS if asset in supported_assets],
            "expansion_assets_supported": [asset for asset in EXPANSION_ASSETS if asset in supported_assets],
            "supported_timeframes": timeframes,
            "decision_timeframes": decision_timeframes if family in {"market_futures_core", "market_futures_binary"} else timeframes,
            "evidence_rows": len(rows),
            "defensibility_label": _defensibility_label(family, supported_assets),
            "required_next_steps": _required_next_steps(family, supported_assets),
        }

    return {
        "assets": assets,
        "families": families,
        "defensibility_summary": (
            "Seven-asset comparison is currently defensible for market-only benchmarks, cross-asset ranking, and the "
            "1h/4h market+futures strategy families. Context and on-chain claims should still remain limited to the "
            "assets where the supporting sentiment/news/on-chain layers are actually generated and evaluated."
        ),
        "recommended_demo_framing": [
            "Use BTC, ETH, and SOL to demonstrate the deeper full-stack pipeline.",
            "Use BNB, XRP, ADA, and DOGE to demonstrate scalable market and futures coverage across popular assets.",
            "Explain that context/on-chain layers are intentionally not claimed for every asset until those source layers are validated.",
        ],
    }
