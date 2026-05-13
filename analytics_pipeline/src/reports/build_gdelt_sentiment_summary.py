"""Create latest app-facing summaries for GDELT-derived asset sentiment."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GDELT_ASSET_QUERY_MAP,
    SENTIMENT_FREQUENCY,
    get_gdelt_sentiment_features_path,
    get_gdelt_sentiment_summary_path,
)
from src.io_paths import ensure_dirs


def _coalesce_source_provenance(latest_row):
    """Keep source provenance stable even when a latest row is sparse."""
    source_count = latest_row.get("news_source_count", 0)
    primary_source_mode = latest_row.get("news_primary_source_mode", "unknown")
    rss_share = latest_row.get("news_rss_share", 0.0)
    gdelt_share = latest_row.get("news_gdelt_share", 0.0)

    try:
        source_count = int(float(source_count or 0))
    except (TypeError, ValueError):
        source_count = 0

    try:
        rss_share = float(rss_share or 0.0)
    except (TypeError, ValueError):
        rss_share = 0.0

    try:
        gdelt_share = float(gdelt_share or 0.0)
    except (TypeError, ValueError):
        gdelt_share = 0.0

    primary_source_mode = str(primary_source_mode or "unknown").strip() or "unknown"

    if source_count == 0 and (rss_share > 0 or gdelt_share > 0):
        source_count = int((rss_share > 0) + (gdelt_share > 0))

    if primary_source_mode == "unknown":
        if rss_share > gdelt_share and rss_share > 0:
            primary_source_mode = "rss_fallback"
        elif gdelt_share > 0:
            primary_source_mode = "gdelt_doc_api"

    return {
        "news_source_count": source_count,
        "news_primary_source_mode": primary_source_mode,
        "news_rss_share": rss_share,
        "news_gdelt_share": gdelt_share,
    }


def build_gdelt_sentiment_summary_for_asset(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                            start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Write one latest summary row for an asset's GDELT sentiment features."""
    ensure_dirs()
    features_path = get_gdelt_sentiment_features_path(asset_symbol, frequency, start_date, end_date)
    if not features_path.exists():
        raise FileNotFoundError(f"GDELT sentiment features not found: {features_path}")

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        summary_df = pd.DataFrame(
            [{
                "asset_symbol": asset_symbol,
                "latest_window_end": "",
                "gdelt_regime_label": "unavailable",
                "gdelt_attention_label": "quiet",
                "gdelt_sentiment_mean": 0.0,
                "gdelt_article_count": 0,
                "gdelt_unique_domain_count": 0,
                "gdelt_coverage_quality_score": 0.0,
                "gdelt_event_intensity_score": 0.0,
                "news_source_count": 0,
                "news_primary_source_mode": "unknown",
                "news_rss_share": 0.0,
                "news_gdelt_share": 0.0,
                "gdelt_dominant_event_theme": "none",
                "gdelt_risk_event_theme": "none",
                "gdelt_supportive_event_theme": "none",
                "gdelt_sentiment_model_used": "unavailable",
                "gdelt_sentiment_data_available": False,
                "gdelt_sentiment_summary": (
                    f"Asset-specific GDELT news sentiment is currently unavailable for {asset_symbol}."
                ),
            }]
        )
    else:
        latest = features_df.sort_values("window_end_utc").iloc[-1]
        source_provenance = _coalesce_source_provenance(latest)
        summary_df = pd.DataFrame(
            [{
                "asset_symbol": asset_symbol,
                "latest_window_end": latest["window_end_utc"],
                "gdelt_regime_label": latest["gdelt_regime_label"],
                "gdelt_attention_label": latest.get("gdelt_attention_label", "quiet"),
                "gdelt_sentiment_mean": latest["gdelt_sentiment_mean"],
                "gdelt_article_count": latest["gdelt_article_count"],
                "gdelt_unique_domain_count": latest.get("gdelt_unique_domain_count", 0),
                "gdelt_coverage_quality_score": latest.get("gdelt_coverage_quality_score", 0.0),
                "gdelt_event_intensity_score": latest.get("gdelt_event_intensity_score", 0.0),
                **source_provenance,
                "gdelt_dominant_event_theme": latest.get("gdelt_dominant_event_theme", "none"),
                "gdelt_risk_event_theme": latest.get("gdelt_risk_event_theme", "none"),
                "gdelt_supportive_event_theme": latest.get("gdelt_supportive_event_theme", "none"),
                "gdelt_sentiment_model_used": latest["gdelt_sentiment_model_used"],
                "gdelt_sentiment_data_available": bool(latest["gdelt_sentiment_data_available"]),
                "gdelt_sentiment_summary": (
                    f"{asset_symbol} news sentiment is currently {str(latest['gdelt_regime_label']).replace('_', ' ')} "
                    f"from {int(latest['gdelt_article_count'])} recent articles across "
                    f"{int(latest.get('gdelt_unique_domain_count', 0))} domains. "
                    f"Coverage quality is {float(latest.get('gdelt_coverage_quality_score', 0.0)):.2f}, "
                    f"attention is {str(latest.get('gdelt_attention_label', 'quiet')).replace('_', ' ')}, "
                    f"dominant theme is {str(latest.get('gdelt_dominant_event_theme', 'none')).replace('_', ' ')}, "
                    f"risk theme is {str(latest.get('gdelt_risk_event_theme', 'none')).replace('_', ' ')}, "
                    f"supportive theme is {str(latest.get('gdelt_supportive_event_theme', 'none')).replace('_', ' ')}, "
                    f"primary source mode is {str(latest.get('news_primary_source_mode', 'unknown')).replace('_', ' ')}, "
                    f"using {latest['gdelt_sentiment_model_used']}."
                ),
            }]
        )

    output_path = get_gdelt_sentiment_summary_path(asset_symbol, frequency, start_date, end_date)
    summary_df.to_csv(output_path, index=False)

    print(f"GDELT sentiment summary generated for {asset_symbol}")
    print(f"summary saved to: {output_path}")
    return summary_df


def build_gdelt_sentiment_summary_for_assets(asset_symbols=None, frequency=SENTIMENT_FREQUENCY,
                                             start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Build latest summaries for all configured GDELT assets."""
    outputs = {}
    for asset_symbol in asset_symbols or list(GDELT_ASSET_QUERY_MAP.keys()):
        try:
            outputs[asset_symbol] = build_gdelt_sentiment_summary_for_asset(
                asset_symbol,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
            )
        except FileNotFoundError as exc:
            print(f"GDELT summary build skipped for {asset_symbol}")
            print(str(exc))
    return outputs


if __name__ == "__main__":
    build_gdelt_sentiment_summary_for_assets()
