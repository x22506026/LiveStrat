"""Rebuild GDELT sentiment for all configured assets in one pass.

Pulls a recent 30-day window of news articles for every asset listed in
``GDELT_ASSET_QUERY_MAP``, runs the FinBERT-or-fallback sentiment scorer,
and regenerates the daily sentiment summary CSVs that the live app
serves under ``/api/sentiment-context``.

Run from the repo root with:

    $env:PYTHONPATH = 'analytics_pipeline'
    python -m src.sentiment.rebuild_all_gdelt
"""

from __future__ import annotations

from datetime import date, timedelta

from src.audit.audit_gdelt_news import run_gdelt_audit_for_asset
from src.build.build_gdelt_sentiment_features import build_gdelt_sentiment_features_for_asset
from src.reports.build_gdelt_sentiment_summary import build_gdelt_sentiment_summary_for_asset
from src.config import GDELT_ASSET_QUERY_MAP


def rebuild_all(window_days: int = 30) -> dict:
    """Fetch + score sentiment for every configured asset.

    Returns a status dict with per-asset row counts and any errors so the
    caller can decide whether to retry.
    """
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=window_days)).isoformat()
    print(f"Rebuilding GDELT sentiment for window {start_date} -> {end_date}")
    print(f"Assets: {list(GDELT_ASSET_QUERY_MAP.keys())}")

    status = {"window": [start_date, end_date], "assets": {}}
    for asset in GDELT_ASSET_QUERY_MAP.keys():
        try:
            raw_df = run_gdelt_audit_for_asset(asset, start_date=start_date, end_date=end_date)
            row_count = int(len(raw_df))

            build_gdelt_sentiment_features_for_asset(
                asset_symbol=asset,
                start_date=start_date,
                end_date=end_date,
            )
            build_gdelt_sentiment_summary_for_asset(
                asset_symbol=asset,
                start_date=start_date,
                end_date=end_date,
            )
            status["assets"][asset] = {
                "raw_rows": row_count,
                "status": "ok" if row_count else "no_articles_returned",
            }
            print(f"[{asset}] {row_count} articles -> sentiment summary regenerated")
        except Exception as exc:
            status["assets"][asset] = {
                "raw_rows": 0,
                "status": "failed",
                "error": str(exc),
            }
            print(f"[{asset}] FAILED: {exc}")
    return status


if __name__ == "__main__":
    summary = rebuild_all()
    print("\n--- summary ---")
    for asset, info in summary["assets"].items():
        print(f"{asset:>5}: {info.get('status'):<22} ({info.get('raw_rows', 0)} rows)")
