"""Retrieve asset-specific crypto news from official public RSS feeds."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, get_raw_gdelt_articles_path
from src.io_paths import GDELT_LOGS_DIR, ensure_dirs


RSS_FEEDS = [
    {
        "source": "coindesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "domain": "coindesk.com",
    },
    {
        "source": "cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "domain": "cointelegraph.com",
    },
]

ASSET_KEYWORD_MAP = {
    "BTC": ["bitcoin", "btc", "spot bitcoin etf"],
    "ETH": ["ethereum", "ether", "eth", "ether etf"],
    "SOL": ["solana", "sol", "solana token"],
    "BNB": ["bnb", "binance coin", "bnb chain"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
}

RSS_REQUEST_INTERVAL_SECONDS = 2.0
RSS_TIMEOUT_SECONDS = 45
_LAST_RSS_REQUEST_TS = None
RSS_OUTPUT_COLUMNS = [
    "asset_symbol",
    "query_used",
    "chunk_start_utc",
    "chunk_end_utc",
    "title",
    "url",
    "url_mobile",
    "domain",
    "language",
    "sourcecountry",
    "socialimage",
    "seendate",
]


def _throttle_rss_requests():
    global _LAST_RSS_REQUEST_TS
    now = time.monotonic()
    if _LAST_RSS_REQUEST_TS is not None:
        elapsed = now - _LAST_RSS_REQUEST_TS
        if elapsed < RSS_REQUEST_INTERVAL_SECONDS:
            time.sleep(RSS_REQUEST_INTERVAL_SECONDS - elapsed)
    _LAST_RSS_REQUEST_TS = time.monotonic()


def _parse_window(start_date, end_date):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23,
        minute=59,
        second=59,
        tzinfo=timezone.utc,
    )
    return start_dt, end_dt


def _fetch_rss_xml(feed_url):
    _throttle_rss_requests()
    response = requests.get(feed_url, timeout=RSS_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _item_text(item, tag_name):
    value = item.findtext(tag_name)
    return value.strip() if isinstance(value, str) else ""


def _parse_item_datetime(item):
    for tag_name in ("pubDate", "published", "updated"):
        value = _item_text(item, tag_name)
        if not value:
            continue
        try:
            published_dt = parsedate_to_datetime(value)
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
            else:
                published_dt = published_dt.astimezone(timezone.utc)
            return published_dt
        except (TypeError, ValueError):
            continue
    return None


def _matches_asset(title, description, asset_symbol):
    haystack = f"{title} {description}".lower()
    return any(keyword in haystack for keyword in ASSET_KEYWORD_MAP.get(asset_symbol, []))


def _normalise_rss_rows(xml_text, source_name, source_domain, asset_symbol, start_dt, end_dt):
    rows = []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = _item_text(item, "title")
        description = _item_text(item, "description")
        link = _item_text(item, "link")
        published_dt = _parse_item_datetime(item)
        if not title or not link or published_dt is None:
            continue
        if not (start_dt <= published_dt <= end_dt):
            continue
        if not _matches_asset(title, description, asset_symbol):
            continue

        rows.append(
            {
                "asset_symbol": asset_symbol,
                "query_used": f"rss_fallback:{source_name}",
                "chunk_start_utc": start_dt.isoformat(),
                "chunk_end_utc": end_dt.isoformat(),
                "title": title,
                "url": link,
                "url_mobile": "",
                "domain": source_domain,
                "language": "",
                "sourcecountry": "",
                "socialimage": "",
                "seendate": published_dt.isoformat(),
            }
        )
    return rows


def _deduplicate_articles(df):
    if df.empty:
        return df
    deduped = df.drop_duplicates(subset=["url"], keep="first")
    deduped = deduped.drop_duplicates(subset=["title", "seendate"], keep="first")
    return deduped.reset_index(drop=True)


def _write_rss_audit_log(asset_symbol, start_date, end_date, output_path, rows_saved, sources_used):
    log_path = GDELT_LOGS_DIR / f"{asset_symbol}_rss_audit_{start_date}_{end_date}.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset_symbol": asset_symbol,
                "sources_used": sources_used,
                "window_start": start_date,
                "window_end": end_date,
                "rows_saved": rows_saved,
                "output_file": str(output_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )
    return log_path


def run_rss_audit_for_asset(asset_symbol, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Fetch official RSS articles for one asset and save them in the shared raw-news schema."""
    ensure_dirs()
    start_dt, end_dt = _parse_window(start_date, end_date)

    all_rows = []
    sources_used = []
    for feed in RSS_FEEDS:
        xml_text = _fetch_rss_xml(feed["url"])
        rows = _normalise_rss_rows(
            xml_text,
            feed["source"],
            feed["domain"],
            asset_symbol,
            start_dt,
            end_dt,
        )
        if rows:
            all_rows.extend(rows)
            sources_used.append(feed["source"])

    raw_df = pd.DataFrame(all_rows, columns=RSS_OUTPUT_COLUMNS)
    raw_df = _deduplicate_articles(raw_df)
    output_path = get_raw_gdelt_articles_path(asset_symbol, start_date, end_date)
    raw_df.to_csv(output_path, index=False)
    log_path = _write_rss_audit_log(
        asset_symbol,
        start_date,
        end_date,
        output_path,
        len(raw_df),
        sources_used,
    )

    print(f"RSS news audit completed for {asset_symbol}")
    print(f"window: {start_date} to {end_date}")
    print(f"rows saved: {len(raw_df)}")
    print(f"raw data saved to: {output_path}")
    print(f"audit log saved to: {log_path}")
    return raw_df


def run_rss_audit_for_assets(asset_symbols, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Fetch official RSS articles for multiple assets."""
    outputs = {}
    failures = {}
    for asset_symbol in asset_symbols:
        try:
            outputs[asset_symbol] = run_rss_audit_for_asset(
                asset_symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:  # pragma: no cover - network reliability is environment-dependent
            failures[asset_symbol] = str(exc)
            print(f"RSS audit failed for {asset_symbol}")
            print(str(exc))

    if failures:
        print("RSS audit completed with partial failures")
        print(json.dumps(failures, indent=2))
    return outputs


if __name__ == "__main__":
    run_rss_audit_for_assets(["BTC", "ETH", "SOL"])
