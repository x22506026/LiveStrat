"""Backend contract checks for the LiveStrat submission state."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = ROOT_DIR / "analytics_pipeline"
for path in (ROOT_DIR, ANALYTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytics_pipeline.src.config import SUPPORTED_SYMBOLS
from analytics_pipeline.src.models.source_governance import (
    NETWORK_ONCHAIN_SOURCE_LIMITED_SYMBOLS,
    build_context_source_governance_snapshot,
)
from analytics_pipeline.src.reports.build_backend_readiness_report import (
    build_backend_readiness_report,
)


class BackendContractTests(unittest.TestCase):
    expected_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]

    @classmethod
    def setUpClass(cls):
        cls.source_governance = build_context_source_governance_snapshot()
        cls.readiness = build_backend_readiness_report()

    def test_supported_asset_universe_is_seven_assets(self):
        self.assertEqual(SUPPORTED_SYMBOLS, self.expected_symbols)

    def test_source_governance_keeps_coinmetrics_and_defillama_separate(self):
        decision = self.source_governance["source_decision"]
        coinmetrics = self.source_governance["layers"]["network_onchain_coinmetrics"]["coverage"]
        defillama = self.source_governance["layers"]["ecosystem_defillama"]["coverage"]

        self.assertEqual(decision["decision"], "keep_both")
        self.assertEqual(coinmetrics["status"], "warning")
        self.assertEqual(defillama["status"], "pass")
        self.assertEqual(
            set(coinmetrics["source_limited_assets"]),
            set(NETWORK_ONCHAIN_SOURCE_LIMITED_SYMBOLS),
        )
        self.assertEqual(set(defillama["available_assets"]), set(self.expected_symbols))
        self.assertIn("not a wallet-level on-chain substitute", decision["rationale"])

    def test_backend_readiness_has_no_failures(self):
        self.assertEqual(self.readiness["status_counts"]["fail"], 0)
        self.assertEqual(self.readiness["source_policy_decision"]["decision"], "keep_both")

    def test_backend_readiness_covers_all_assets_in_core_outputs(self):
        required_check_names = {
            "market_intelligence_overview_1h",
            "market_intelligence_overview_4h",
            "market_intelligence_overview_1d",
            "market_futures_signal_summary_1h",
            "market_futures_signal_summary_4h",
            "market_multimodal_strategy_summary_1h",
            "market_multimodal_strategy_summary_4h",
            "context_ablation_1h",
            "context_ablation_4h",
            "defillama_ecosystem_daily",
            "runtime_support",
        }
        checks = {check["name"]: check for check in self.readiness["checks"]}
        self.assertTrue(required_check_names.issubset(checks))

        for check_name in required_check_names - {"runtime_support"}:
            with self.subTest(check=check_name):
                check = checks[check_name]
                self.assertEqual(set(check["assets"]), set(self.expected_symbols))
                self.assertEqual(check.get("missing_assets", []), [])

        for check_name in ("context_ablation_1h", "context_ablation_4h"):
            with self.subTest(check=check_name):
                check = checks[check_name]
                self.assertIn("market_futures_plus_defi", check["variants"])
                self.assertEqual(set(check["defi_variant_assets"]), set(self.expected_symbols))
                self.assertEqual(check["missing_defi_variant_assets"], [])

    def test_app_payloads_expose_defillama_without_overclaiming_onchain(self):
        from app import build_analytics_payload, build_dashboard_payload, get_context_source_governance

        governance = get_context_source_governance()
        self.assertEqual(governance["source_decision"]["decision"], "keep_both")

        for symbol, expected_chain in (("SOLUSDT", "Solana"), ("BNBUSDT", "BSC")):
            with self.subTest(symbol=symbol):
                dashboard = build_dashboard_payload(symbol, "4h")
                analytics = build_analytics_payload(symbol, "4h")
                self.assertEqual(dashboard["source_policy_decision"]["decision"], "keep_both")
                self.assertEqual(analytics["source_policy_decision"]["decision"], "keep_both")
                self.assertEqual(dashboard["defi_context"]["chain_name"], expected_chain)
                self.assertTrue(dashboard["defi_context"]["available"])
                self.assertEqual(analytics["defi_context"]["chain_name"], expected_chain)
                self.assertTrue(analytics["defi_context"]["available"])


if __name__ == "__main__":
    unittest.main()
