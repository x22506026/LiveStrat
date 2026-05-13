"""Retrieve public DeFiLlama chain TVL context for LiveStrat assets."""

import json
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import requests

from src.config import (
    DEFILLAMA_API_BASE,
    DEFILLAMA_CHAIN_MAP,
    DEFILLAMA_FREQUENCY,
    get_raw_defillama_chain_snapshot_path,
    get_raw_defillama_chain_tvl_path,
)
from src.io_paths import DEFI_LLAMA_LOGS_DIR, ensure_dirs


REQUEST_TIMEOUT_SECONDS = 45


def fetch_chain_snapshot():
    """Fetch the current DeFiLlama chain universe snapshot."""
    response = requests.get(f"{DEFILLAMA_API_BASE}/v2/chains", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def fetch_historical_chain_tvl(chain_name):
    """Fetch historical chain TVL rows for one DeFiLlama chain name."""
    response = requests.get(
        f"{DEFILLAMA_API_BASE}/v2/historicalChainTvl/{quote(chain_name)}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def write_defillama_audit_log(asset_symbol, chain_name, output_path, rows_saved):
    """Persist a compact audit log for the DeFiLlama pull."""
    log_path = DEFI_LLAMA_LOGS_DIR / f"{asset_symbol}_defillama_audit_log.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset_symbol": asset_symbol,
                "chain_name": chain_name,
                "rows_saved": int(rows_saved),
                "output_file": str(output_path),
                "source": "defillama_public_api",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )
    return log_path


def run_defillama_audit_for_asset(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    """Retrieve one asset's mapped DeFiLlama chain TVL history."""
    ensure_dirs()
    chain_name = DEFILLAMA_CHAIN_MAP[asset_symbol]
    raw_df = fetch_historical_chain_tvl(chain_name)
    if not raw_df.empty:
        raw_df.insert(0, "asset_symbol", asset_symbol)
        raw_df.insert(1, "chain_name", chain_name)
    output_path = get_raw_defillama_chain_tvl_path(asset_symbol, frequency)
    raw_df.to_csv(output_path, index=False)
    log_path = write_defillama_audit_log(asset_symbol, chain_name, output_path, len(raw_df))

    print(f"DeFiLlama audit completed for {asset_symbol}")
    print(f"chain: {chain_name}")
    print(f"rows saved: {len(raw_df)}")
    print(f"raw data saved to: {output_path}")
    print(f"audit log saved to: {log_path}")
    return raw_df


def run_defillama_audit_for_assets(asset_symbols=None, frequency=DEFILLAMA_FREQUENCY):
    """Retrieve DeFiLlama TVL history for all configured assets by default."""
    ensure_dirs()
    snapshot_df = fetch_chain_snapshot()
    snapshot_path = get_raw_defillama_chain_snapshot_path()
    snapshot_df.to_csv(snapshot_path, index=False)

    outputs = {}
    for asset_symbol in asset_symbols or list(DEFILLAMA_CHAIN_MAP):
        outputs[asset_symbol] = run_defillama_audit_for_asset(asset_symbol, frequency=frequency)
    return outputs


if __name__ == "__main__":
    run_defillama_audit_for_assets()
