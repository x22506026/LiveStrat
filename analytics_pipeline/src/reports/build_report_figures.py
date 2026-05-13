"""Generate publication-quality matplotlib figures for the final-year report.

Reads the most recent evaluation outputs in ``data/processed`` and writes
PNG figures to ``report_figures/`` at the repo root, ready to drop into
the Word report. Every figure has a consistent visual style (LiveStrat
green primary, neutral grey grid, no chart-junk) and an explanatory
caption text printed to stdout when the script runs, so the report author
can copy the recommended caption straight into the figure label.

Run from the repo root with:

    $env:PYTHONPATH = 'analytics_pipeline'
    python -m src.reports.build_report_figures

Figures produced:
* fig_01_per_asset_accuracy.png        - per-asset accuracy on the 4h window
* fig_02_strategy_vs_hold.png          - strategy return vs buy-and-hold per asset
* fig_03_sharpe_per_asset.png          - Sharpe ratio per asset (capped)
* fig_04_cost_sensitivity.png          - excess return at 1x, 2x, 3x, 5x fee
* fig_05_ablation_macro_f1.png         - macro-F1 by data-layer ablation step
* fig_06_finbert_confusion.png         - FinBERT label agreement
* fig_07_walkforward_stability.png     - per-fold accuracy stability
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; do not require a display server.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LIVESTRAT_GREEN = "#008B5A"
LIVESTRAT_AMBER = "#D49500"
LIVESTRAT_RED = "#B84B4B"
LIVESTRAT_GREY = "#6B7280"


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _latest(pattern: str) -> Path | None:
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _short_asset(symbol: str) -> str:
    return symbol.replace("USDT", "")


def figure_per_asset_accuracy() -> Path | None:
    overview_path = _latest("market_intelligence_overview_4h_*.csv")
    if overview_path is None:
        return None
    df = pd.read_csv(overview_path)
    df = df.sort_values("baseline_scaled_test_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    assets = df["symbol"].map(_short_asset).tolist()
    accuracy = df["baseline_scaled_test_accuracy"].astype(float).fillna(0.0) * 100
    rule_acc = df["rule_based_test_accuracy"].astype(float).fillna(0.0) * 100

    x = np.arange(len(assets))
    width = 0.38
    ax.bar(x - width / 2, accuracy, width, label="Scaled market baseline", color=LIVESTRAT_GREEN)
    ax.bar(x + width / 2, rule_acc, width, label="Rule-based benchmark", color=LIVESTRAT_GREY)
    ax.axhline(33.33, color=LIVESTRAT_RED, linestyle="--", linewidth=1, label="Random ternary baseline (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_title("Per-asset classification accuracy on 4h evaluation window")
    ax.legend(loc="lower right", frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = FIG_DIR / "fig_01_per_asset_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_strategy_vs_hold() -> Path | None:
    summary_path = _latest("market_futures_backtest_summary_4h_*.csv")
    if summary_path is None:
        return None
    df = pd.read_csv(summary_path)

    assets = df["symbol"].map(_short_asset).tolist()
    strategy_ret = df["strategy_total_return"].astype(float) * 100
    hold_ret = df["buy_hold_total_return"].astype(float) * 100

    x = np.arange(len(assets))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, strategy_ret, width, label="Strategy", color=LIVESTRAT_GREEN)
    ax.bar(x + width / 2, hold_ret, width, label="Buy & hold", color=LIVESTRAT_GREY)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Total return over window (%)")
    ax.set_title("Strategy return vs buy-and-hold (4h window)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_02_strategy_vs_hold.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_sharpe_per_asset() -> Path | None:
    summary_path = _latest("market_futures_backtest_summary_4h_*.csv")
    if summary_path is None:
        return None
    df = pd.read_csv(summary_path).copy()
    df["sharpe_capped"] = df["sharpe_ratio"].astype(float).clip(-3, 8)
    df = df.sort_values("sharpe_capped", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [LIVESTRAT_GREEN if v >= 1 else (LIVESTRAT_AMBER if v >= 0 else LIVESTRAT_RED) for v in df["sharpe_capped"]]
    ax.bar(df["symbol"].map(_short_asset), df["sharpe_capped"], color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1, color=LIVESTRAT_GREEN, linestyle="--", linewidth=0.8)
    ax.set_ylabel("Sharpe ratio (capped at +/-3, +8 for display)")
    ax.set_title("Per-asset Sharpe ratio over the 4h window")
    fig.tight_layout()
    out = FIG_DIR / "fig_03_sharpe_per_asset.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_cost_sensitivity() -> Path | None:
    sens_path = _latest("market_futures_cost_sensitivity_summary_4h*.csv")
    if sens_path is None:
        return None
    df = pd.read_csv(sens_path)
    pivot = df.pivot_table(
        index="symbol",
        columns="cost_multiplier",
        values="excess_return",
        aggfunc="last",
    ) * 100
    pivot.index = pivot.index.map(_short_asset)
    pivot = pivot.reindex(["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    multipliers = sorted(df["cost_multiplier"].unique())
    bar_width = 0.18
    x = np.arange(len(pivot.index))
    for i, m in enumerate(multipliers):
        ax.bar(x + (i - len(multipliers) / 2 + 0.5) * bar_width, pivot[m],
               bar_width,
               label=f"{m:.0f}x fee" if float(m).is_integer() else f"{m}x fee",
               color={1.0: LIVESTRAT_GREEN, 2.0: LIVESTRAT_AMBER, 3.0: "#A05E1E", 5.0: LIVESTRAT_RED}.get(m, LIVESTRAT_GREY))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Excess return over buy-and-hold (%)")
    ax.set_title("Transaction-cost sensitivity: strategy edge survives x-fold fee scaling?")
    ax.legend(loc="lower right", frameon=False, ncol=4)
    fig.tight_layout()
    out = FIG_DIR / "fig_04_cost_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_ablation_macro_f1() -> Path | None:
    ablation_path = _latest("market_context_ablation_summary_4h_*.csv")
    if ablation_path is None:
        return None
    df = pd.read_csv(ablation_path)
    variant_col = "variant_name" if "variant_name" in df.columns else df.columns[1]
    metric_col = "macro_f1" if "macro_f1" in df.columns else "test_macro_f1"

    if metric_col not in df.columns:
        # fall back to any column containing 'macro'
        candidates = [c for c in df.columns if "macro" in c.lower()]
        if not candidates:
            return None
        metric_col = candidates[0]

    grouped = df.groupby(variant_col)[metric_col].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(grouped.index.astype(str), grouped.values * 100, color=LIVESTRAT_GREEN)
    ax.set_xlabel("Mean held-out macro-F1 (%)")
    ax.set_title("Ablation: macro-F1 across data-layer combinations (4h, all assets)")
    fig.tight_layout()
    out = FIG_DIR / "fig_05_ablation_macro_f1.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_finbert_confusion() -> Path | None:
    validation_path = PROCESSED_DIR / "finbert_validation_run.csv"
    if not validation_path.exists():
        return None
    df = pd.read_csv(validation_path)
    labels = ["positive", "neutral", "negative"]
    matrix = pd.crosstab(df["expected_label"], df["finbert_label"]).reindex(index=labels, columns=labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix.values, cmap="Greens")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.title() for l in labels])
    ax.set_yticklabels([l.title() for l in labels])
    ax.set_xlabel("FinBERT-predicted label")
    ax.set_ylabel("A priori expected label")
    ax.set_title("FinBERT validation: agreement matrix (16-headline sample)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix.values[i, j]),
                    ha="center", va="center",
                    color="white" if matrix.values[i, j] > matrix.values.max() / 2 else "black",
                    fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = FIG_DIR / "fig_06_finbert_confusion.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_walkforward_stability() -> Path | None:
    detail_path = _latest("market_futures_binary_walkforward_detail_4h_*.csv")
    if detail_path is None:
        return None
    df = pd.read_csv(detail_path)
    if "fold_accuracy" not in df.columns or "symbol" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = {
        "BTC": LIVESTRAT_GREEN, "ETH": "#0066CC", "SOL": "#7C3AED",
        "BNB": LIVESTRAT_AMBER, "XRP": "#0EA5E9", "ADA": "#10B981", "DOGE": "#F59E0B",
    }
    for symbol, group in df.groupby("symbol"):
        short = _short_asset(symbol)
        ax.plot(range(1, len(group) + 1), group["fold_accuracy"].astype(float) * 100,
                marker="o", linewidth=1.4, label=short,
                color=palette.get(short, LIVESTRAT_GREY))
    ax.axhline(33.33, color=LIVESTRAT_RED, linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_xlabel("Walk-forward fold number")
    ax.set_ylabel("Fold accuracy (%)")
    ax.set_title("Per-fold accuracy across walk-forward folds (binary, 4h)")
    ax.legend(loc="lower left", frameon=False, ncol=4)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = FIG_DIR / "fig_07_walkforward_stability.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    _apply_style()
    produced: list[tuple[str, Path | None]] = [
        ("Figure 6.1 - Per-asset accuracy", figure_per_asset_accuracy()),
        ("Figure 6.2 - Strategy vs buy-and-hold", figure_strategy_vs_hold()),
        ("Figure 6.3 - Sharpe per asset", figure_sharpe_per_asset()),
        ("Figure 6.4 - Transaction cost sensitivity", figure_cost_sensitivity()),
        ("Figure 6.5 - Ablation macro-F1", figure_ablation_macro_f1()),
        ("Figure 6.6 - FinBERT confusion matrix", figure_finbert_confusion()),
        ("Figure 6.7 - Walk-forward stability", figure_walkforward_stability()),
    ]
    print(f"Output directory: {FIG_DIR}")
    for label, path in produced:
        if path is None:
            print(f"  [skipped] {label} - source CSV not found")
        else:
            print(f"  [ok]      {label} -> {path.name}")


if __name__ == "__main__":
    main()
