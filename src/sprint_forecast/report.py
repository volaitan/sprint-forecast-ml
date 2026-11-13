"""Charts and a Markdown model report for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .model import TrainingReport  # noqa: E402

BLUE, ORANGE, GREY = "#2f5fd0", "#e8890c", "#9aa4b2"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_charts(df: pd.DataFrame, report: TrainingReport, test: pd.DataFrame, out: str | Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Delivery rate by commitment ratio (the key insight)
    ratio = df["committed_points"] / df["velocity_last_3"].clip(lower=1)
    bins = pd.cut(ratio, [0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 5], right=False)
    rate = df.groupby(bins, observed=True)["delivered"].mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    labels = ["<0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0", "1.0–1.1", "1.1–1.2", "1.2–1.3", "1.3+"][: len(rate)]
    ax.bar(labels, rate.values, color=[BLUE if r >= 50 else ORANGE for r in rate.values])
    ax.set_xlabel("Commitment ÷ average velocity of last 3 sprints")
    ax.set_ylabel("Sprints delivered (%)")
    ax.set_title("Over-committing is the biggest risk")
    ax.set_ylim(0, 100)
    paths.append(out / "delivery_by_commitment.png")
    _save(fig, paths[-1])

    # 2. Feature importance
    imp = pd.Series(report.importances).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(imp.index.str.replace("_", " "), imp.values, color=BLUE)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What drives the prediction (permutation importance)")
    paths.append(out / "feature_importance.png")
    _save(fig, paths[-1])

    # 3. ROC + calibration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fpr, tpr, _ = roc_curve(test["actual"], test["predicted"])
    a1.plot(fpr, tpr, color=BLUE, label=f"AUC = {report.test_metrics['roc_auc']:.2f}")
    a1.plot([0, 1], [0, 1], "--", color=GREY)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title("ROC curve (unseen teams)")
    a1.legend(loc="lower right", frameon=False)
    frac, mean_pred = calibration_curve(test["actual"], test["predicted"], n_bins=10, strategy="quantile")
    a2.plot(mean_pred, frac, "o-", color=ORANGE, label="Model")
    a2.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    a2.set_xlabel("Predicted probability"); a2.set_ylabel("Observed delivery rate"); a2.set_title("Calibration")
    a2.legend(loc="upper left", frameon=False)
    paths.append(out / "roc_calibration.png")
    _save(fig, paths[-1])
    return paths


def markdown(report: TrainingReport) -> str:
    m = report.test_metrics
    lines = [
        "| Model | Cross-validated ROC-AUC |", "|---|---|",
        *[f"| {k.replace('_', ' ')}{' **(selected)**' if k == report.best_model else ''} | {v:.3f} |" for k, v in report.cv_auc.items()],
        "",
        "| Test metric (unseen teams) | Value |", "|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.3f} |",
        f"| Accuracy | {m['accuracy']:.1%} (always guessing the majority class: {m['baseline_accuracy']:.1%}) |",
        f"| F1 score | {m['f1']:.3f} |",
        f"| Brier score (lower is better) | {m['brier']:.3f} |",
    ]
    return "\n".join(lines)


__all__ = ["make_charts", "markdown", "np"]
