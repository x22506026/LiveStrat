"""Build daily asset-specific news sentiment features from GDELT article pulls."""

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GDELT_ASSET_QUERY_MAP,
    SENTIMENT_FREQUENCY,
    SENTIMENT_MODEL_BASELINE,
    SENTIMENT_MODEL_PRIMARY,
    get_gdelt_sentiment_features_path,
    get_raw_gdelt_articles_path,
)
from src.io_paths import ensure_dirs


POSITIVE_TERMS = {
    "adoption", "approval", "approve", "approved", "bullish", "breakout", "gain", "gains", "grew",
    "growth", "institutional", "launch", "positive", "profit", "profits", "rally", "record",
    "recover", "recovery", "rebound", "surge", "strong", "strength", "uptrend",
}
NEGATIVE_TERMS = {
    "ban", "bearish", "breach", "collapse", "crash", "cracks", "crime", "decline", "drop", "dropped",
    "dump", "fear", "fraud", "hack", "hacked", "investigation", "lawsuit", "loss", "losses",
    "negative", "risk", "selloff", "slump", "stolen", "volatility", "warning", "weak", "weakness",
}

FINBERT_LABEL_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

EVENT_THEME_KEYWORDS = {
    "regulation": {
        "ban", "compliance", "court", "etf", "judge", "law", "lawsuit", "legal",
        "policy", "regulation", "regulator", "sec", "senator", "sues", "government",
    },
    "security": {
        "attack", "breach", "exploit", "fraud", "hack", "hacked", "malware",
        "scam", "stolen", "theft", "vulnerability",
    },
    "adoption": {
        "adoption", "approval", "approved", "custody", "institutional", "merchant",
        "partnership", "payment", "payments", "treasury", "stablecoin", "tokenization",
        "mastercard", "access", "traction", "real-world", "rwa",
    },
    "network": {
        "congestion", "ecosystem", "fork", "latency", "network", "outage", "research",
        "scaling", "upgrade", "validator",
    },
    "flows": {
        "coinbase premium", "derivatives", "exchange", "inflow", "liquidation", "merger",
        "open interest", "outflow", "premium", "reserve", "resistance", "risk aversion",
        "selloff", "shorts", "treasury yield", "yield", "whale", "fomc", "hawkish", "oil",
    },
    "macro_stress": {
        "cpi", "debt ceiling", "deficit", "fomc", "hawkish", "inflation", "macro", "oil",
        "rate cut", "rates", "recession", "tariff", "treasury yield", "volatility spike", "yield",
    },
    "capital_markets": {
        "acquisition", "bond", "convertible", "equity raise", "financing", "ipo", "listing",
        "m&a", "merger", "secondary offering", "share sale", "treasury", "treasury reserve",
    },
    "stablecoin_payments": {
        "cross-border", "mastercard", "merchant", "payments", "paypal", "remittance", "settlement",
        "stablecoin", "stripe", "transfer rail", "usdc", "visa",
    },
}


def normalise_to_utc_midnight(timestamp_series):
    """Project timestamps into a daily UTC bucket."""
    return pd.to_datetime(timestamp_series, utc=True, errors="coerce").dt.floor("D")


def rule_based_title_sentiment(text):
    """Use a light lexical baseline when a transformer sentiment model is unavailable."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    words = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]
    positive_hits = sum(word in POSITIVE_TERMS for word in words)
    negative_hits = sum(word in NEGATIVE_TERMS for word in words)
    if positive_hits == 0 and negative_hits == 0:
        return 0.0

    raw_score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
    return float(max(min(raw_score, 1.0), -1.0))


def count_keyword_hits(text):
    """Count positive/negative finance-news keyword hits in a title."""
    if not isinstance(text, str) or not text.strip():
        return 0, 0

    words = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]
    positive_hits = sum(word in POSITIVE_TERMS for word in words)
    negative_hits = sum(word in NEGATIVE_TERMS for word in words)
    return positive_hits, negative_hits


def count_event_theme_hits(text):
    """Count simple event-theme hits so LiveStrat can explain what the news is about."""
    if not isinstance(text, str) or not text.strip():
        return {theme: 0 for theme in EVENT_THEME_KEYWORDS}

    normalized_text = " ".join(
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
    )
    return {
        theme: sum(keyword in normalized_text for keyword in keywords)
        for theme, keywords in EVENT_THEME_KEYWORDS.items()
    }


def classify_news_source_mode(query_used):
    """Map raw query provenance into a stable source mode label."""
    query_used = str(query_used or "").strip().lower()
    if query_used.startswith("rss_fallback:"):
        return "rss_fallback"
    if query_used:
        return "gdelt_doc_api"
    return "unknown"


def try_finbert_scores(texts):
    """Score texts with FinBERT if the local environment supports it."""
    if not texts:
        return [], SENTIMENT_MODEL_BASELINE

    try:
        from transformers import pipeline  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, "finbert_unavailable_import"

    try:
        classifier = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            framework="pt",  # force PyTorch; Keras 3 breaks the TF auto-path.
        )
    except Exception:
        return None, "finbert_unavailable_runtime"

    try:
        results = classifier(
            texts,
            truncation=True,
            max_length=256,
            batch_size=16,
        )
    except Exception:
        return None, "finbert_inference_failed"

    scores = []
    for result in results:
        label = str(result.get("label", "")).lower()
        probability = float(result.get("score", 0.0))
        direction = FINBERT_LABEL_MAP.get(label, 0.0)
        scores.append(direction * probability)
    return scores, SENTIMENT_MODEL_PRIMARY


def score_article_titles(df):
    """Apply the best available title-level sentiment scorer."""
    titles = df["title"].fillna("").astype(str).tolist()
    finbert_scores, model_used = try_finbert_scores(titles)
    if finbert_scores is not None:
        scored_df = df.copy()
        scored_df["gdelt_title_sentiment_score"] = finbert_scores
        scored_df["gdelt_sentiment_model_used"] = model_used
        return scored_df

    scored_df = df.copy()
    scored_df["gdelt_title_sentiment_score"] = scored_df["title"].apply(rule_based_title_sentiment)
    scored_df["gdelt_sentiment_model_used"] = SENTIMENT_MODEL_BASELINE
    return scored_df


def classify_gdelt_regime(score):
    """Map a continuous aggregate news score into a strategy-friendly label."""
    if pd.isna(score):
        return "unavailable"
    if score >= 0.15:
        return "supportive"
    if score <= -0.15:
        return "risk_off"
    return "mixed"


def classify_attention_label(event_intensity_score):
    """Map article volume and keyword pressure into a simple attention label."""
    if pd.isna(event_intensity_score):
        return "quiet"
    if event_intensity_score >= 0.75:
        return "event_heavy"
    if event_intensity_score >= 0.35:
        return "active"
    return "quiet"


def build_empty_feature_frame(asset_symbol, model_used):
    """Build an empty but schema-complete daily feature frame."""
    return pd.DataFrame(
        columns=[
            "asset_symbol",
            "window_end_utc",
            "gdelt_article_count",
            "gdelt_unique_domain_count",
            "gdelt_sentiment_mean",
            "gdelt_sentiment_std",
            "gdelt_positive_share",
            "gdelt_negative_share",
            "gdelt_neutral_share",
            "gdelt_sentiment_momentum_3d",
            "gdelt_sentiment_momentum_7d",
            "gdelt_article_count_zscore_30d",
            "gdelt_article_count_3d_sum",
            "gdelt_article_count_7d_sum",
            "gdelt_positive_keyword_share",
            "gdelt_negative_keyword_share",
            "gdelt_keyword_intensity_mean",
            "gdelt_source_concentration",
            "gdelt_coverage_quality_score",
            "gdelt_event_intensity_score",
            "gdelt_attention_label",
            "news_source_count",
            "news_primary_source_mode",
            "news_rss_share",
            "news_gdelt_share",
            "gdelt_regulation_share",
            "gdelt_security_share",
            "gdelt_adoption_share",
            "gdelt_network_share",
            "gdelt_flows_share",
            "gdelt_macro_stress_share",
            "gdelt_capital_markets_share",
            "gdelt_stablecoin_payments_share",
            "gdelt_dominant_event_theme",
            "gdelt_risk_event_theme",
            "gdelt_supportive_event_theme",
            "gdelt_regime_score",
            "gdelt_regime_label",
            "gdelt_supportive_flag",
            "gdelt_risk_off_flag",
            "gdelt_sentiment_data_available",
            "gdelt_sentiment_model_used",
        ]
    ).assign(asset_symbol=asset_symbol, gdelt_sentiment_model_used=model_used)


def build_daily_calendar(start_date, end_date):
    """Create a dense daily UTC index for fair chronological sentiment alignment."""
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    return pd.date_range(start=start_ts, end=end_ts, freq="D", tz="UTC")


def build_gdelt_sentiment_features_for_asset(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                             start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Aggregate raw GDELT article data into daily asset-specific sentiment features."""
    ensure_dirs()
    if asset_symbol not in GDELT_ASSET_QUERY_MAP:
        raise KeyError(f"No GDELT query configured for asset {asset_symbol}.")

    raw_path = get_raw_gdelt_articles_path(asset_symbol, start_date, end_date)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw GDELT article data not found: {raw_path}")

    try:
        raw_df = pd.read_csv(raw_path)
    except EmptyDataError:
        raw_df = pd.DataFrame()
    if raw_df.empty:
        features_df = build_empty_feature_frame(asset_symbol, SENTIMENT_MODEL_BASELINE)
    else:
        raw_df["seendate"] = pd.to_datetime(raw_df["seendate"], utc=True, errors="coerce")
        raw_df = raw_df.dropna(subset=["seendate"]).sort_values("seendate").reset_index(drop=True)
        if raw_df.empty:
            features_df = build_empty_feature_frame(asset_symbol, SENTIMENT_MODEL_BASELINE)
        else:
            scored_df = score_article_titles(raw_df)
            scored_df["news_source_mode"] = scored_df["query_used"].apply(classify_news_source_mode)
            keyword_hits = scored_df["title"].fillna("").astype(str).apply(count_keyword_hits)
            scored_df["gdelt_positive_keyword_hits"] = keyword_hits.apply(lambda value: value[0])
            scored_df["gdelt_negative_keyword_hits"] = keyword_hits.apply(lambda value: value[1])
            scored_df["gdelt_keyword_intensity"] = (
                scored_df["gdelt_positive_keyword_hits"] + scored_df["gdelt_negative_keyword_hits"]
            )
            theme_hits = scored_df["title"].fillna("").astype(str).apply(count_event_theme_hits)
            for theme in EVENT_THEME_KEYWORDS:
                scored_df[f"gdelt_{theme}_hits"] = theme_hits.apply(lambda value: value.get(theme, 0))
            scored_df["window_end_utc"] = normalise_to_utc_midnight(scored_df["seendate"])

            daily_df = (
                scored_df.groupby("window_end_utc", as_index=False)
                .agg(
                    gdelt_article_count=("title", "size"),
                    gdelt_unique_domain_count=("domain", lambda values: values.dropna().astype(str).nunique()),
                    gdelt_source_concentration=("domain", lambda values: values.dropna().astype(str).value_counts(normalize=True).max() if len(values.dropna()) else 0.0),
                    gdelt_sentiment_mean=("gdelt_title_sentiment_score", "mean"),
                    gdelt_sentiment_std=("gdelt_title_sentiment_score", "std"),
                    gdelt_positive_share=("gdelt_title_sentiment_score", lambda values: (values > 0.10).mean()),
                    gdelt_negative_share=("gdelt_title_sentiment_score", lambda values: (values < -0.10).mean()),
                    gdelt_neutral_share=("gdelt_title_sentiment_score", lambda values: (values.between(-0.10, 0.10)).mean()),
                    gdelt_positive_keyword_share=("gdelt_positive_keyword_hits", lambda values: (values > 0).mean()),
                    gdelt_negative_keyword_share=("gdelt_negative_keyword_hits", lambda values: (values > 0).mean()),
                    gdelt_keyword_intensity_mean=("gdelt_keyword_intensity", "mean"),
                    news_source_count=("news_source_mode", lambda values: values.dropna().astype(str).nunique()),
                    news_primary_source_mode=("news_source_mode", lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "unknown"),
                    news_rss_share=("news_source_mode", lambda values: values.dropna().astype(str).eq("rss_fallback").mean()),
                    news_gdelt_share=("news_source_mode", lambda values: values.dropna().astype(str).eq("gdelt_doc_api").mean()),
                    gdelt_regulation_share=("gdelt_regulation_hits", lambda values: (values > 0).mean()),
                    gdelt_security_share=("gdelt_security_hits", lambda values: (values > 0).mean()),
                    gdelt_adoption_share=("gdelt_adoption_hits", lambda values: (values > 0).mean()),
                    gdelt_network_share=("gdelt_network_hits", lambda values: (values > 0).mean()),
                    gdelt_flows_share=("gdelt_flows_hits", lambda values: (values > 0).mean()),
                    gdelt_macro_stress_share=("gdelt_macro_stress_hits", lambda values: (values > 0).mean()),
                    gdelt_capital_markets_share=("gdelt_capital_markets_hits", lambda values: (values > 0).mean()),
                    gdelt_stablecoin_payments_share=("gdelt_stablecoin_payments_hits", lambda values: (values > 0).mean()),
                    gdelt_sentiment_model_used=("gdelt_sentiment_model_used", "last"),
                )
            )

            dense_index = build_daily_calendar(start_date, end_date)
            daily_df = (
                daily_df.set_index("window_end_utc")
                .reindex(dense_index)
                .rename_axis("window_end_utc")
                .reset_index()
            )

            daily_df["asset_symbol"] = asset_symbol
            daily_df["gdelt_article_count"] = daily_df["gdelt_article_count"].fillna(0).astype(int)
            daily_df["gdelt_unique_domain_count"] = daily_df["gdelt_unique_domain_count"].fillna(0).astype(int)
            daily_df["gdelt_source_concentration"] = daily_df["gdelt_source_concentration"].fillna(0.0)
            daily_df["gdelt_sentiment_mean"] = daily_df["gdelt_sentiment_mean"].fillna(0.0)
            daily_df["gdelt_sentiment_std"] = daily_df["gdelt_sentiment_std"].fillna(0.0)
            daily_df["gdelt_positive_share"] = daily_df["gdelt_positive_share"].fillna(0.0)
            daily_df["gdelt_negative_share"] = daily_df["gdelt_negative_share"].fillna(0.0)
            daily_df["gdelt_neutral_share"] = daily_df["gdelt_neutral_share"].fillna(0.0)
            daily_df["gdelt_positive_keyword_share"] = daily_df["gdelt_positive_keyword_share"].fillna(0.0)
            daily_df["gdelt_negative_keyword_share"] = daily_df["gdelt_negative_keyword_share"].fillna(0.0)
            daily_df["gdelt_keyword_intensity_mean"] = daily_df["gdelt_keyword_intensity_mean"].fillna(0.0)
            daily_df["news_source_count"] = daily_df["news_source_count"].fillna(0).astype(int)
            daily_df["news_primary_source_mode"] = daily_df["news_primary_source_mode"].fillna("unknown")
            daily_df["news_rss_share"] = daily_df["news_rss_share"].fillna(0.0)
            daily_df["news_gdelt_share"] = daily_df["news_gdelt_share"].fillna(0.0)
            daily_df["gdelt_regulation_share"] = daily_df["gdelt_regulation_share"].fillna(0.0)
            daily_df["gdelt_security_share"] = daily_df["gdelt_security_share"].fillna(0.0)
            daily_df["gdelt_adoption_share"] = daily_df["gdelt_adoption_share"].fillna(0.0)
            daily_df["gdelt_network_share"] = daily_df["gdelt_network_share"].fillna(0.0)
            daily_df["gdelt_flows_share"] = daily_df["gdelt_flows_share"].fillna(0.0)
            daily_df["gdelt_macro_stress_share"] = daily_df["gdelt_macro_stress_share"].fillna(0.0)
            daily_df["gdelt_capital_markets_share"] = daily_df["gdelt_capital_markets_share"].fillna(0.0)
            daily_df["gdelt_stablecoin_payments_share"] = daily_df["gdelt_stablecoin_payments_share"].fillna(0.0)
            daily_df["gdelt_sentiment_model_used"] = daily_df["gdelt_sentiment_model_used"].fillna(
                daily_df["gdelt_sentiment_model_used"].dropna().iloc[-1]
                if not daily_df["gdelt_sentiment_model_used"].dropna().empty
                else SENTIMENT_MODEL_BASELINE
            )
            daily_df["gdelt_sentiment_momentum_3d"] = daily_df["gdelt_sentiment_mean"].diff(3).fillna(0.0)
            daily_df["gdelt_sentiment_momentum_7d"] = daily_df["gdelt_sentiment_mean"].diff(7).fillna(0.0)

            rolling_mean = daily_df["gdelt_article_count"].rolling(30, min_periods=5).mean()
            rolling_std = daily_df["gdelt_article_count"].rolling(30, min_periods=5).std()
            daily_df["gdelt_article_count_zscore_30d"] = (
                (daily_df["gdelt_article_count"] - rolling_mean) / rolling_std.replace(0, pd.NA)
            ).fillna(0.0)
            daily_df["gdelt_article_count_3d_sum"] = daily_df["gdelt_article_count"].rolling(3, min_periods=1).sum()
            daily_df["gdelt_article_count_7d_sum"] = daily_df["gdelt_article_count"].rolling(7, min_periods=1).sum()

            daily_df["gdelt_coverage_quality_score"] = (
                0.40 * (daily_df["gdelt_article_count"].clip(upper=8) / 8.0)
                + 0.25 * (daily_df["gdelt_unique_domain_count"].clip(upper=5) / 5.0)
                + 0.20 * (1.0 - daily_df["gdelt_source_concentration"].clip(lower=0.0, upper=1.0))
                + 0.15 * daily_df["gdelt_keyword_intensity_mean"].clip(upper=3.0).fillna(0.0) / 3.0
            ).clip(lower=0.0, upper=1.0)

            daily_df["gdelt_event_intensity_score"] = (
                0.45 * daily_df["gdelt_article_count_zscore_30d"].clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
                + 0.30 * daily_df["gdelt_keyword_intensity_mean"].clip(upper=3.0).fillna(0.0) / 3.0
                + 0.25 * daily_df["gdelt_sentiment_mean"].abs().clip(upper=0.50).fillna(0.0) / 0.50
            ).clip(lower=0.0, upper=1.0)

            daily_df["gdelt_regime_score"] = (
                0.7 * daily_df["gdelt_sentiment_mean"].fillna(0.0)
                + 0.2 * daily_df["gdelt_sentiment_momentum_3d"].clip(-1.0, 1.0).fillna(0.0)
                + 0.1 * (
                    daily_df["gdelt_positive_keyword_share"].fillna(0.0)
                    - daily_df["gdelt_negative_keyword_share"].fillna(0.0)
                )
            )
            daily_df["gdelt_regime_label"] = daily_df["gdelt_regime_score"].apply(classify_gdelt_regime)
            daily_df["gdelt_attention_label"] = daily_df["gdelt_event_intensity_score"].apply(classify_attention_label)
            theme_columns = [
                "gdelt_regulation_share",
                "gdelt_security_share",
                "gdelt_adoption_share",
                "gdelt_network_share",
                "gdelt_flows_share",
                "gdelt_macro_stress_share",
                "gdelt_capital_markets_share",
                "gdelt_stablecoin_payments_share",
            ]
            theme_labels = [
                "regulation",
                "security",
                "adoption",
                "network",
                "flows",
                "macro_stress",
                "capital_markets",
                "stablecoin_payments",
            ]
            theme_idx = daily_df[theme_columns].to_numpy().argmax(axis=1)
            max_theme_strength = daily_df[theme_columns].max(axis=1)
            daily_df["gdelt_dominant_event_theme"] = [
                theme_labels[idx] if strength > 0 else "none"
                for idx, strength in zip(theme_idx, max_theme_strength)
            ]
            daily_df["gdelt_risk_event_theme"] = "none"
            daily_df.loc[
                daily_df[["gdelt_security_share", "gdelt_flows_share", "gdelt_regulation_share", "gdelt_macro_stress_share"]].max(axis=1) > 0,
                "gdelt_risk_event_theme",
            ] = daily_df[["gdelt_security_share", "gdelt_flows_share", "gdelt_regulation_share", "gdelt_macro_stress_share"]].idxmax(axis=1).str.replace("gdelt_", "", regex=False).str.replace("_share", "", regex=False)
            daily_df["gdelt_supportive_event_theme"] = "none"
            daily_df.loc[
                daily_df[["gdelt_adoption_share", "gdelt_network_share", "gdelt_stablecoin_payments_share"]].max(axis=1) > 0,
                "gdelt_supportive_event_theme",
            ] = daily_df[["gdelt_adoption_share", "gdelt_network_share", "gdelt_stablecoin_payments_share"]].idxmax(axis=1).str.replace("gdelt_", "", regex=False).str.replace("_share", "", regex=False)
            daily_df["gdelt_supportive_flag"] = daily_df["gdelt_regime_label"].eq("supportive")
            daily_df["gdelt_risk_off_flag"] = daily_df["gdelt_regime_label"].eq("risk_off")
            daily_df["gdelt_sentiment_data_available"] = daily_df["gdelt_article_count"] > 0
            daily_df["window_end_utc"] = daily_df["window_end_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            feature_columns = [
                "asset_symbol",
                "window_end_utc",
                "gdelt_article_count",
                "gdelt_unique_domain_count",
                "gdelt_sentiment_mean",
                "gdelt_sentiment_std",
                "gdelt_positive_share",
                "gdelt_negative_share",
                "gdelt_neutral_share",
                "gdelt_sentiment_momentum_3d",
                "gdelt_sentiment_momentum_7d",
                "gdelt_article_count_zscore_30d",
                "gdelt_article_count_3d_sum",
                "gdelt_article_count_7d_sum",
                "gdelt_positive_keyword_share",
                "gdelt_negative_keyword_share",
                "gdelt_keyword_intensity_mean",
                "gdelt_source_concentration",
                "gdelt_coverage_quality_score",
                "gdelt_event_intensity_score",
                "gdelt_attention_label",
                "news_source_count",
                "news_primary_source_mode",
                "news_rss_share",
                "news_gdelt_share",
                "gdelt_regulation_share",
                "gdelt_security_share",
                "gdelt_adoption_share",
                "gdelt_network_share",
                "gdelt_flows_share",
                "gdelt_macro_stress_share",
                "gdelt_capital_markets_share",
                "gdelt_stablecoin_payments_share",
                "gdelt_dominant_event_theme",
                "gdelt_risk_event_theme",
                "gdelt_supportive_event_theme",
                "gdelt_regime_score",
                "gdelt_regime_label",
                "gdelt_supportive_flag",
                "gdelt_risk_off_flag",
                "gdelt_sentiment_data_available",
                "gdelt_sentiment_model_used",
            ]
            features_df = daily_df[feature_columns].copy()

    output_path = get_gdelt_sentiment_features_path(asset_symbol, frequency, start_date, end_date)
    features_df.to_csv(output_path, index=False)

    print(f"GDELT sentiment features built for {asset_symbol}")
    print(f"rows saved: {len(features_df)}")
    print(f"processed features saved to: {output_path}")
    return features_df


def build_gdelt_sentiment_features_for_assets(asset_symbols, frequency=SENTIMENT_FREQUENCY,
                                              start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Aggregate GDELT article data into daily features for multiple assets."""
    outputs = {}
    for asset_symbol in asset_symbols:
        try:
            outputs[asset_symbol] = build_gdelt_sentiment_features_for_asset(
                asset_symbol,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
            )
        except FileNotFoundError as exc:
            print(f"GDELT feature build skipped for {asset_symbol}")
            print(str(exc))
    return outputs


if __name__ == "__main__":
    build_gdelt_sentiment_features_for_assets(["BTC", "ETH", "SOL"])
