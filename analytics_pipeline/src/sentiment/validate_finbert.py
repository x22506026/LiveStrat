"""FinBERT validation.

Loads real GDELT headlines from a labelled CSV, scores them with FinBERT
and the lexical fallback, writes a results CSV and prints per scorer
agreement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.build.build_gdelt_sentiment_features import rule_based_title_sentiment
from src.config import PROCESSED_DIR
from src.reproducibility import set_global_seed


VALIDATION_SET_PATH = PROCESSED_DIR / "finbert_validation_set.csv"
VALIDATION_RUN_PATH = PROCESSED_DIR / "finbert_validation_run.csv"


def load_headline_samples() -> list[tuple[str, str, str]]:
    # read the labelled CSV that ships in the repo
    if not VALIDATION_SET_PATH.exists():
        raise FileNotFoundError(
            f"Validation set not found at {VALIDATION_SET_PATH}. "
            "Sample real GDELT headlines into this file first."
        )
    df = pd.read_csv(VALIDATION_SET_PATH)
    required = {"asset", "title", "expected_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Validation set missing columns: {missing}")
    df = df.dropna(subset=["asset", "title", "expected_label"])
    df = df[df["expected_label"].str.strip() != ""]
    return [
        (str(row["asset"]).strip(), str(row["title"]).strip(), str(row["expected_label"]).strip().lower())
        for _, row in df.iterrows()
    ]


def lexical_label_from_score(score: float, neutral_band: float = 0.0) -> str:
    # convert the lexical scorer's signed number to a class label
    if score > neutral_band:
        return "positive"
    if score < -neutral_band:
        return "negative"
    return "neutral"


def run_finbert_validation() -> Path:
    # load model, score every headline with FinBERT and lexical, write a results CSV
    set_global_seed(deep_learning=True)
    samples = load_headline_samples()

    from transformers import pipeline  # heavy import deferred

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        framework="pt",
    )

    titles = [t for _, t, _ in samples]
    results = classifier(titles, top_k=None, truncation=True, max_length=64, batch_size=8)

    rows = []
    for (asset, title, expected_label), result in zip(samples, results):
        scored = {item["label"].lower(): float(item["score"]) for item in result}
        positive = scored.get("positive", 0.0)
        negative = scored.get("negative", 0.0)
        neutral = scored.get("neutral", 0.0)

        if positive >= max(negative, neutral):
            finbert_label = "positive"
        elif negative >= max(positive, neutral):
            finbert_label = "negative"
        else:
            finbert_label = "neutral"
        finbert_signed_score = positive - negative

        # baseline: same lexical scorer the live pipeline uses as a fallback
        lexical_signed_score = rule_based_title_sentiment(title)
        lexical_label = lexical_label_from_score(lexical_signed_score)

        rows.append(
            {
                "asset": asset,
                "title": title,
                "expected_label": expected_label,
                "finbert_label": finbert_label,
                "finbert_positive_score": positive,
                "finbert_negative_score": negative,
                "finbert_neutral_score": neutral,
                "finbert_signed_score": finbert_signed_score,
                "finbert_agrees_with_expected": finbert_label == expected_label,
                "lexical_label": lexical_label,
                "lexical_signed_score": lexical_signed_score,
                "lexical_agrees_with_expected": lexical_label == expected_label,
                "finbert_agrees_with_lexical": finbert_label == lexical_label,
            }
        )

    df = pd.DataFrame(rows)
    df["validation_run_at"] = datetime.now(timezone.utc).isoformat()
    df["model_name"] = "ProsusAI/finbert"
    df.to_csv(VALIDATION_RUN_PATH, index=False)
    return VALIDATION_RUN_PATH


def summarise_validation_csv(csv_path: Path) -> dict:
    # produce per scorer overall and per class agreement statistics
    df = pd.read_csv(csv_path)
    n = len(df)
    finbert_overall = float(df["finbert_agrees_with_expected"].mean()) if n else 0.0
    lexical_overall = float(df["lexical_agrees_with_expected"].mean()) if n else 0.0
    return {
        "n_headlines": n,
        "finbert_overall": finbert_overall,
        "lexical_overall": lexical_overall,
        "lift_finbert_over_lexical_pp": (finbert_overall - lexical_overall) * 100.0,
        "finbert_by_expected_label": (
            df.groupby("expected_label")["finbert_agrees_with_expected"].mean().to_dict() if n else {}
        ),
        "lexical_by_expected_label": (
            df.groupby("expected_label")["lexical_agrees_with_expected"].mean().to_dict() if n else {}
        ),
    }


if __name__ == "__main__":
    print("Loading FinBERT. First call downloads the model (~440 MB).")
    out = run_finbert_validation()
    stats = summarise_validation_csv(out)

    print(f"Wrote {out}")
    print(f"Headlines scored: {stats['n_headlines']}")
    print()
    print("Overall agreement with labels:")
    print(f"  FinBERT : {stats['finbert_overall']:.1%}")
    print(f"  Lexical : {stats['lexical_overall']:.1%}")
    print(f"  Lift    : {stats['lift_finbert_over_lexical_pp']:+.1f} percentage points")
    print()
    print("Per expected class agreement:")
    print(f"  {'class':>10} {'FinBERT':>10} {'Lexical':>10}")
    for label in ("positive", "neutral", "negative"):
        fb = stats["finbert_by_expected_label"].get(label, 0.0)
        lx = stats["lexical_by_expected_label"].get(label, 0.0)
        print(f"  {label:>10} {fb:>9.1%}  {lx:>9.1%}")
