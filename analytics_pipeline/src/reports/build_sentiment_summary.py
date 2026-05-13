"""Create app-facing latest market mood summary outputs."""

import pandas as pd

from src.config import SENTIMENT_FREQUENCY, get_sentiment_features_path, get_sentiment_summary_path
from src.io_paths import ensure_dirs


def build_sentiment_summary(frequency=SENTIMENT_FREQUENCY):
    """Create the latest summary row for the current market mood source."""
    ensure_dirs()
    features_path = get_sentiment_features_path(frequency)
    if not features_path.exists():
        raise FileNotFoundError(f"Processed sentiment features not found: {features_path}")

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        summary_df = pd.DataFrame(
            [{
                "latest_window_end": "",
                "sentiment_value": 0.0,
                "sentiment_classification": "unavailable",
                "market_mood_label": "unavailable",
                "sentiment_change_1d": 0.0,
                "sentiment_change_7d": 0.0,
                "sentiment_data_available": False,
                "sentiment_summary": "Market mood data is currently unavailable.",
            }]
        )
    else:
        latest = features_df.sort_values("window_end_utc").iloc[-1]
        summary_df = pd.DataFrame(
            [{
                "latest_window_end": latest["window_end_utc"],
                "sentiment_value": latest["sentiment_value"],
                "sentiment_classification": latest["sentiment_classification"],
                "market_mood_label": latest["market_mood_label"],
                "sentiment_change_1d": latest["sentiment_change_1d"],
                "sentiment_change_7d": latest["sentiment_change_7d"],
                "sentiment_data_available": bool(latest["sentiment_data_available"]),
                "sentiment_summary": (
                    f"Overall crypto market mood is currently {str(latest['market_mood_label']).replace('_', ' ')} "
                    f"with Fear & Greed at {float(latest['sentiment_value']):.0f}."
                ),
            }]
        )

    output_path = get_sentiment_summary_path(frequency)
    summary_df.to_csv(output_path, index=False)

    print("sentiment summary generated")
    print(f"summary saved to: {output_path}")

    return summary_df


if __name__ == "__main__":
    build_sentiment_summary()
