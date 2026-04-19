"""
Generate report figures: confusion matrix heatmap and metrics bar chart (JPG).
Reuses the same 80/20 evaluation path as the project report (seed=42, SpamAssassin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from naive_bayes_spam_classifier import (
    MultinomialNaiveBayesClassifier,
    evaluate_classifier,
    load_spamassassin,
    train_test_split,
)


def _run_eval(seed: int = 42, test_fraction: float = 0.2):
    documents, labels = load_spamassassin()
    train_docs, train_labels, test_docs, test_labels = train_test_split(
        documents, labels, test_fraction=test_fraction, seed=seed
    )
    clf = MultinomialNaiveBayesClassifier()
    clf.fit(train_docs, train_labels)
    return evaluate_classifier(clf, test_docs, test_labels)


def save_confusion_matrix_figure(cm, out_path: Path, dpi: int = 150) -> None:
    """cm: ConfusionMatrix with TP/FP/TN/FN (spam positive). Rows = actual class."""
    grid = np.array(
        [
            [cm.true_positive, cm.false_negative],
            [cm.false_positive, cm.true_negative],
        ],
        dtype=float,
    )
    row_labels = ["Actual spam", "Actual ham"]
    col_labels = ["Pred spam", "Pred ham"]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(grid, cmap="Blues", vmin=0)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(grid[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                color="#111",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(row_labels)
    ax.set_title("Confusion matrix (Spam = positive class)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fmt = "png" if out_path.suffix.lower() == ".png" else "jpeg"
    fig.savefig(out_path, format=fmt, dpi=dpi)
    plt.close(fig)


def save_metrics_bar_figure(metrics, out_path: Path, dpi: int = 150) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(x, vals, color=["#2e7d32", "#1565c0", "#6a1b9a", "#ad1457"])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Hold-out test metrics (80/20 split, seed=42)")
    ax.axhline(1.0, color="#ccc", linestyle="--", linewidth=1)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fmt = "png" if out_path.suffix.lower() == ".png" else "jpeg"
    fig.savefig(out_path, format=fmt, dpi=dpi)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running evaluation for figures...", file=sys.stderr)
    m = _run_eval(seed=42, test_fraction=0.2)

    save_confusion_matrix_figure(m.confusion, out_dir / "confusion_matrix.jpg")
    save_metrics_bar_figure(m, out_dir / "metrics_bar.jpg")
    save_confusion_matrix_figure(m.confusion, out_dir / "confusion_matrix.png")
    save_metrics_bar_figure(m, out_dir / "metrics_bar.png")

    print(f"Wrote JPG/PNG under: {out_dir}", file=sys.stderr)
    print(m.summary())


if __name__ == "__main__":
    main()
