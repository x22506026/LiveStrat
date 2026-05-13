"""Summarize source provenance and freshness for asset-news inputs."""

from datetime import datetime, timezone

import pandas as pd

from src.config import ASSET_REGISTRY, PROCESSED_DIR, SENTIMENT_FREQUENCY, get_strategy_summary_path


def _latest_csv(pattern):
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def _age_days(timestamp_value):
    if not timestamp_value:
        return None
    try:
        timestamp = pd.Timestamp(timestamp_value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        now = pd.Timestamp(datetime.now(timezone.utc))
        return max((now - timestamp).days, 0)
    except Exception:
        return None


def _preferred_source_metrics(latest_summary, latest_feature):
    """Prefer summary values, but recover from sparse summary rows safely."""
    def _as_int(value, default=0):
        try:
            return int(float(value or default))
        except (TypeError, ValueError):
            return default

    def _as_float(value, default=0.0):
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return default

    source_count = _as_int(latest_summary.get("news_source_count"), _as_int(latest_feature.get("news_source_count")))
    rss_share = _as_float(latest_summary.get("news_rss_share"), _as_float(latest_feature.get("news_rss_share")))
    gdelt_share = _as_float(latest_summary.get("news_gdelt_share"), _as_float(latest_feature.get("news_gdelt_share")))
    primary_source_mode = str(
        latest_summary.get("news_primary_source_mode")
        or latest_feature.get("news_primary_source_mode")
        or "unknown"
    ).strip() or "unknown"

    if source_count == 0:
        source_count = _as_int(latest_feature.get("news_source_count"), source_count)
    if primary_source_mode == "unknown":
        primary_source_mode = str(latest_feature.get("news_primary_source_mode", "unknown") or "unknown")
    if rss_share == 0.0:
        rss_share = _as_float(latest_feature.get("news_rss_share"), rss_share)
    if gdelt_share == 0.0:
        gdelt_share = _as_float(latest_feature.get("news_gdelt_share"), gdelt_share)

    if source_count == 0 and (rss_share > 0 or gdelt_share > 0):
        source_count = int((rss_share > 0) + (gdelt_share > 0))
    if primary_source_mode == "unknown":
        if rss_share > gdelt_share and rss_share > 0:
            primary_source_mode = "rss_fallback"
        elif gdelt_share > 0:
            primary_source_mode = "gdelt_doc_api"

    return source_count, primary_source_mode, rss_share, gdelt_share


def evaluate_news_source_coverage():
    """Create one compact asset-level summary of source mix, freshness, and coverage."""
    summary_rows = []

    for asset_symbol, asset_config in ASSET_REGISTRY.items():
        if asset_config.get("tier") != "core":
            continue

        features_df = _latest_csv(f"{asset_symbol}_gdelt_sentiment_features_{SENTIMENT_FREQUENCY}_*.csv")
        summary_df = _latest_csv(f"{asset_symbol}_gdelt_sentiment_summary_{SENTIMENT_FREQUENCY}_*.csv")
        latest_feature = features_df.iloc[-1].to_dict() if not features_df.empty else {}
        latest_summary = summary_df.iloc[-1].to_dict() if not summary_df.empty else {}

        latest_window_end = latest_summary.get("latest_window_end", "")
        freshness_age_days = _age_days(latest_window_end)
        if freshness_age_days is None:
            freshness_label = "unavailable"
        elif freshness_age_days <= 5:
            freshness_label = "fresh"
        elif freshness_age_days <= 10:
            freshness_label = "aging"
        else:
            freshness_label = "stale"

        source_count, primary_source_mode, rss_share, gdelt_share = _preferred_source_metrics(
            latest_summary,
            latest_feature,
        )
        article_count = int(float(latest_summary.get("gdelt_article_count", 0) or 0))
        unique_domains = int(float(latest_summary.get("gdelt_unique_domain_count", 0) or 0))
        coverage_quality = float(latest_summary.get("gdelt_coverage_quality_score", 0.0) or 0.0)
        source_concentration = float(latest_feature.get("gdelt_source_concentration", 0.0) or 0.0)

        summary_rows.append(
            {
                "asset_symbol": asset_symbol,
                "market_symbol": asset_config["market_symbol"],
                "latest_window_end": latest_window_end,
                "news_article_count": article_count,
                "news_unique_domains": unique_domains,
                "news_source_count": source_count,
                "news_primary_source_mode": primary_source_mode,
                "news_rss_share": rss_share,
                "news_gdelt_share": gdelt_share,
                "news_coverage_quality_score": coverage_quality,
                "news_source_concentration": source_concentration,
                "news_freshness_age_days": freshness_age_days,
                "news_freshness_label": freshness_label,
                "news_source_summary": (
                    f"{asset_symbol} news coverage uses {primary_source_mode.replace('_', ' ')} as the primary mode, "
                    f"with {article_count} articles across {unique_domains} domains. Freshness is {freshness_label}."
                ),
            }
        )

    summary_output = pd.DataFrame(summary_rows)
    output_path = get_strategy_summary_path("news_source_coverage", SENTIMENT_FREQUENCY)
    summary_output.to_csv(output_path, index=False)

    print("news source coverage summary generated")
    print(f"rows saved: {len(summary_output)}")
    print(f"summary saved to: {output_path}")
    return summary_output


if __name__ == "__main__":
    evaluate_news_source_coverage()
