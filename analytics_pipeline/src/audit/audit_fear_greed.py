"""Retrieve the Alternative.me Fear & Greed Index for market-wide sentiment."""

import json
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import (
    SENTIMENT_FREQUENCY,
    SENTIMENT_LOOKBACK_LIMIT,
    SENTIMENT_SOURCE_URL,
    get_raw_sentiment_path,
)
from src.io_paths import SENTIMENT_LOGS_DIR, ensure_dirs


def fetch_fear_greed_history(limit=SENTIMENT_LOOKBACK_LIMIT):
    """Fetch the latest or full Fear & Greed history from Alternative.me."""
    response = requests.get(
        SENTIMENT_SOURCE_URL,
        params={"limit": limit, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def run_fear_greed_audit(frequency=SENTIMENT_FREQUENCY):
    """Retrieve and save raw market mood history."""
    ensure_dirs()
    rows = fetch_fear_greed_history()
    df = pd.DataFrame(rows)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    output_path = get_raw_sentiment_path(frequency)
    df.to_csv(output_path, index=False)

    log_path = SENTIMENT_LOGS_DIR / "fear_greed_audit_log.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "alternative_me_fear_greed",
                "rows_saved": len(df),
                "output_file": str(output_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )

    print("fear and greed audit completed")
    print(f"rows saved: {len(df)}")
    print(f"raw data saved to: {output_path}")
    print(f"audit log saved to: {log_path}")

    return df


if __name__ == "__main__":
    run_fear_greed_audit()
