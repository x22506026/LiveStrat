"""Standalone FinBERT validation pass for LiveStrat.

The main pipeline (`build_gdelt_sentiment_features.py`) tries to use
FinBERT when the local environment supports it and falls back to a lexical
scorer otherwise. This script proves the FinBERT pathway works on this
machine by running the model on a representative set of crypto-news
headlines and saving the per-headline output.

The output CSV (`finbert_validation_run.csv` under
``analytics_pipeline/data/processed``) is referenced from the project
report as direct evidence that the deep-NLP layer is functioning, not
just optionally compiled in.

Run from the repo root with:

    $env:PYTHONPATH = 'analytics_pipeline'
    python -m src.sentiment.validate_finbert
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.reproducibility import set_global_seed


# A small but representative basket of crypto headlines. The labels
# alongside each headline are the LiveStrat author's a priori expectation
# for FinBERT's classification, used only for the comparison column in the
# output CSV - they are not training data.
HEADLINE_SAMPLES: list[tuple[str, str, str]] = [
    # asset, headline, expected_label
    ("BTC", "Bitcoin surges to new all-time high as ETF inflows accelerate", "positive"),
    ("BTC", "Bitcoin tumbles below 70k amid US regulatory crackdown fears", "negative"),
    ("BTC", "Bitcoin holds steady near 80k as traders await macro data", "neutral"),
    ("ETH", "Ethereum upgrade lowers gas fees and lifts developer activity", "positive"),
    ("ETH", "Ethereum drops 8 percent after SEC files new enforcement action", "negative"),
    ("ETH", "Ethereum trades sideways as funding rates stay balanced", "neutral"),
    ("SOL", "Solana validator outage triggers sharp sell-off", "negative"),
    ("SOL", "Solana DeFi TVL hits 12-month high on Jupiter launch", "positive"),
    ("BNB", "Binance settles US legal case, lifting BNB sentiment", "positive"),
    ("BNB", "BNB chain hit by exploit, bridge funds frozen", "negative"),
    ("XRP", "Ripple wins partial summary judgement in long-running SEC case", "positive"),
    ("XRP", "XRP slides as cross-border payment rivals gain traction", "negative"),
    ("ADA", "Cardano stablecoin pilot draws institutional interest", "positive"),
    ("ADA", "Cardano price stalls amid lukewarm developer commits", "neutral"),
    ("DOGE", "Dogecoin spikes 20 percent on viral celebrity endorsement", "positive"),
    ("DOGE", "Dogecoin slumps as memecoin enthusiasm fades", "negative"),
]


def run_finbert_validation() -> Path:
    """Run FinBERT over the sample headlines and write the CSV."""
    set_global_seed(deep_learning=True)

    from transformers import pipeline  # heavy import deferred to call time

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        framework="pt",  # force the PyTorch backend; Keras 3 breaks the TF path.
    )

    titles = [t for _, t, _ in HEADLINE_SAMPLES]
    results = classifier(titles, top_k=None, truncation=True, max_length=64, batch_size=8)

    rows = []
    for (asset, title, expected_label), result in zip(HEADLINE_SAMPLES, results):
        # `result` is a list of {label, score} dicts (one per class) because
        # top_k=None. Normalise to a per-row score.
        scored = {item["label"].lower(): float(item["score"]) for item in result}
        positive = scored.get("positive", 0.0)
        negative = scored.get("negative", 0.0)
        neutral = scored.get("neutral", 0.0)

        if positive >= max(negative, neutral):
            predicted_label = "positive"
        elif negative >= max(positive, neutral):
            predicted_label = "negative"
        else:
            predicted_label = "neutral"

        signed_score = positive - negative

        rows.append(
            {
                "asset": asset,
                "title": title,
                "expected_label": expected_label,
                "finbert_label": predicted_label,
                "finbert_positive_score": positive,
                "finbert_negative_score": negative,
                "finbert_neutral_score": neutral,
                "finbert_signed_score": signed_score,
                "agrees_with_expected": predicted_label == expected_label,
            }
        )

    df = pd.DataFrame(rows)
    df["validation_run_at"] = datetime.now(timezone.utc).isoformat()
    df["model_name"] = "ProsusAI/finbert"

    output_path = PROCESSED_DIR / "finbert_validation_run.csv"
    df.to_csv(output_path, index=False)
    return output_path


def summarise_validation_csv(csv_path: Path) -> dict:
    """Return high-level agreement statistics for the project report."""
    df = pd.read_csv(csv_path)
    n = len(df)
    n_agree = int(df["agrees_with_expected"].sum())
    by_class = df.groupby("expected_label")["agrees_with_expected"].mean().to_dict()
    return {
        "n_headlines": n,
        "agreement_overall": n_agree / n if n else 0.0,
        "agreement_by_expected_label": by_class,
    }


if __name__ == "__main__":
    print("Loading FinBERT - first call will download the model (~440 MB) ...")
    out = run_finbert_validation()
    stats = summarise_validation_csv(out)
    print(f"Wrote {out}")
    print(f"Headlines scored: {stats['n_headlines']}")
    print(f"Overall agreement with a priori labels: {stats['agreement_overall']:.0%}")
    print("Per expected-class agreement:")
    for label, agreement in stats["agreement_by_expected_label"].items():
        print(f"  {label:>8}: {agreement:.0%}")
