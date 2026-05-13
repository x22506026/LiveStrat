"""Governance rules for LiveStrat's market, sentiment, on-chain, and ecosystem sources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, SUPPORTED_SYMBOLS

NETWORK_ONCHAIN_EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
NETWORK_ONCHAIN_SOURCE_LIMITED_SYMBOLS = ("SOLUSDT", "BNBUSDT")
REPORT_PATH = PROCESSED_DIR / "source_governance_report.json"

SOURCE_GOVERNANCE = {
    "market_spot": {
        "label": "Spot market data",
        "source": "Binance public spot klines",
        "role": "Core price, return, trend, volatility, and volume features.",
        "decision_role": "primary_signal_input",
        "direct_replacement_for": [],
        "ui_language": "Market structure",
        "allowed_use": "Can drive market and market+futures strategy decisions after leakage-safe feature construction.",
        "limitations": "Exchange-specific trading data; it does not explain news, wallet activity, or ecosystem liquidity by itself.",
    },
    "futures_structure": {
        "label": "Futures structure data",
        "source": "Binance public futures endpoints",
        "role": "Funding, open interest, long/short pressure, taker flow, and basis context.",
        "decision_role": "primary_signal_input",
        "direct_replacement_for": [],
        "ui_language": "Futures positioning",
        "allowed_use": "Can influence market+futures strategies when endpoint coverage is complete or explicitly proxied.",
        "limitations": "Some futures endpoints are recent-window limited and basis may need a documented fallback.",
    },
    "asset_news_sentiment": {
        "label": "Asset news sentiment",
        "source": "GDELT news search and RSS-style article context",
        "role": "Event risk, attention, and asset-specific narrative confirmation.",
        "decision_role": "context_confirmation",
        "direct_replacement_for": [],
        "ui_language": "News context",
        "allowed_use": "Can confirm or caution market signals; should not be the sole trading signal.",
        "limitations": "Coverage varies by asset and time window, and refreshes are slower than market data.",
    },
    "broad_sentiment": {
        "label": "Broad market sentiment",
        "source": "Fear & Greed style market mood data",
        "role": "Market-wide fallback sentiment when asset-specific news is unavailable.",
        "decision_role": "fallback_context",
        "direct_replacement_for": ["asset_news_sentiment"],
        "ui_language": "Market mood",
        "allowed_use": "Can be shown as fallback context when asset-specific sentiment is unavailable.",
        "limitations": "Not asset-specific; should be labelled as broad market mood rather than coin-specific evidence.",
    },
    "network_onchain_coinmetrics": {
        "label": "Network on-chain data",
        "source": "Coin Metrics Community API",
        "role": "Wallet/network activity, valuation, fee, transfer, and exchange-flow structure where free community metrics support the asset.",
        "decision_role": "structural_confirmation",
        "direct_replacement_for": [],
        "ui_language": "Network on-chain",
        "allowed_use": "Can support higher-timeframe confirmation for assets with fresh, non-empty Coin Metrics community coverage.",
        "limitations": "Free community coverage is asset/metric limited; SOL and BNB are source-limited here in the current project.",
        "source_urls": [
            "https://docs.coinmetrics.io/api",
            "https://gitbook-docs.coinmetrics.io/packages/coin-metrics-community-data",
        ],
    },
    "ecosystem_defillama": {
        "label": "DeFi ecosystem data",
        "source": "DeFiLlama public API",
        "role": "Chain-level TVL, ecosystem liquidity trend, and structural DeFi participation context.",
        "decision_role": "ecosystem_confirmation",
        "direct_replacement_for": [],
        "ui_language": "Ecosystem TVL",
        "allowed_use": "Can provide all-asset ecosystem context and can improve multimodal confirmation, especially where wallet-level on-chain is unavailable.",
        "limitations": "Not a wallet/network telemetry substitute; TVL can be affected by token prices, protocol coverage, bridge accounting, and chain naming.",
        "source_urls": [
            "https://defillama.com/docs/api",
            "https://docs.llama.fi/faqs/frequently-asked-questions",
        ],
    },
}


def _safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _latest_path(pattern: str) -> Path | None:
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _onchain_coverage() -> dict:
    available_assets = []
    unavailable_assets = []
    for symbol in SUPPORTED_SYMBOLS:
        asset = symbol.replace("USDT", "")
        df = _safe_read_csv(_latest_path(f"{asset}_onchain_features_1d.csv"))
        if df.empty:
            unavailable_assets.append(symbol)
            continue
        if "onchain_data_available" in df.columns:
            is_available = df["onchain_data_available"].astype(str).str.lower().eq("true").any()
        else:
            is_available = True
        if is_available:
            available_assets.append(symbol)
        else:
            unavailable_assets.append(symbol)

    expected_missing = [
        symbol for symbol in NETWORK_ONCHAIN_EXPECTED_SYMBOLS if symbol not in available_assets
    ]
    return {
        "available_assets": sorted(available_assets),
        "unavailable_assets": sorted(unavailable_assets),
        "expected_assets": list(NETWORK_ONCHAIN_EXPECTED_SYMBOLS),
        "source_limited_assets": list(NETWORK_ONCHAIN_SOURCE_LIMITED_SYMBOLS),
        "expected_missing_assets": expected_missing,
        "status": "warning" if unavailable_assets and not expected_missing else "fail" if expected_missing else "pass",
    }


def _defillama_coverage() -> dict:
    df = _safe_read_csv(_latest_path("defillama_summary_1d.csv"))
    if df.empty or "symbol" not in df.columns:
        return {
            "available_assets": [],
            "unavailable_assets": list(SUPPORTED_SYMBOLS),
            "status": "fail",
        }
    available_df = df
    if "defi_context_available" in df.columns:
        available_df = df[df["defi_context_available"].astype(str).str.lower().eq("true")]
    available_assets = sorted({str(symbol) for symbol in available_df["symbol"].dropna().tolist()})
    unavailable_assets = [symbol for symbol in SUPPORTED_SYMBOLS if symbol not in available_assets]
    return {
        "available_assets": available_assets,
        "unavailable_assets": unavailable_assets,
        "status": "pass" if not unavailable_assets else "fail",
    }


def build_context_source_governance_snapshot() -> dict:
    """Return source roles, coverage, and the decision on whether sources should be combined."""
    onchain = _onchain_coverage()
    defi = _defillama_coverage()

    keep_both = defi["status"] == "pass" and onchain["status"] in {"pass", "warning"}
    source_decision = {
        "decision": "keep_both" if keep_both else "review_sources",
        "headline": (
            "Use Coin Metrics and DeFiLlama together, but keep their roles separate."
            if keep_both
            else "Review source coverage before presenting all context layers as ready."
        ),
        "rationale": (
            "Coin Metrics provides true network/on-chain telemetry where community coverage exists. "
            "DeFiLlama provides all-asset ecosystem TVL context, including SOL and BNB, but it is not a wallet-level on-chain substitute."
        ),
        "ui_rule": (
            "Label Coin Metrics as Network On-chain and DeFiLlama as Ecosystem TVL. "
            "Never merge both into a single generic on-chain score without showing source roles."
        ),
        "submission_rule": (
            "Describe SOL and BNB as Coin Metrics source-limited, then show that LiveStrat handles this professionally by using DeFiLlama for chain ecosystem context and by preventing unsupported on-chain claims."
        ),
    }

    layers = {}
    for key, config in SOURCE_GOVERNANCE.items():
        layer = dict(config)
        if key == "network_onchain_coinmetrics":
            layer["coverage"] = onchain
        elif key == "ecosystem_defillama":
            layer["coverage"] = defi
        else:
            layer["coverage"] = {
                "available_assets": list(SUPPORTED_SYMBOLS),
                "unavailable_assets": [],
                "status": "pass",
            }
        layers[key] = layer

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_universe": list(SUPPORTED_SYMBOLS),
        "source_decision": source_decision,
        "layers": layers,
    }


def write_context_source_governance_report(path: Path = REPORT_PATH) -> dict:
    report = build_context_source_governance_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    result = write_context_source_governance_report()
    print(f"source governance report saved to: {REPORT_PATH}")
    print(result["source_decision"]["decision"])
