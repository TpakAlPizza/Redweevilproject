"""Charts for the report. Every figure is saved as a PNG and returned as base64
so the HTML report is a single self-contained file that survives being emailed.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _finish(fig, out_path: Optional[Path]) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def roc_plot(y_true, y_prob, out_path: Optional[Path] = None, title: str = "ROC curve") -> str:
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate (healthy flagged)")
    ax.set_ylabel("True positive rate (infected found)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _finish(fig, out_path)


def pr_plot(y_true, y_prob, out_path: Optional[Path] = None, title: str = "Precision-recall curve") -> str:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    base = float(np.mean(y_true))
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(base, ls="--", c="k", lw=1, label=f"prevalence = {base:.2f}")
    ax.set_xlabel("Recall (infected)")
    ax.set_ylabel("Precision (infected)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    return _finish(fig, out_path)


def calibration_plot(bins: List[dict], out_path: Optional[Path] = None, title: str = "Reliability diagram") -> str:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(4.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    xs = [b["mean_predicted"] for b in bins if b["count"]]
    ys = [b["observed_rate"] for b in bins if b["count"]]
    counts = [b["count"] for b in bins if b["count"]]
    centres = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins if b["count"]]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    if xs:
        ax.plot(xs, ys, "o-", lw=2, label="observed")
    ax.set_ylabel("Observed infected rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.bar(centres, counts, width=0.08)
    ax2.set_xlabel("Predicted probability (calibrated)")
    ax2.set_ylabel("n")
    return _finish(fig, out_path)


def confusion_plot(counts: Dict[str, int], out_path: Optional[Path] = None, title: str = "Confusion matrix") -> str:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["predicted healthy", "predicted infected"])
    ax.set_yticks([0, 1], ["actual healthy", "actual infected"])
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=15, color="black" if matrix[i, j] < matrix.max() / 2 else "white")
    ax.set_title(f"{title} (raw counts)")
    return _finish(fig, out_path)


def learning_curves(fold_histories: Dict[str, List[dict]], out_path: Optional[Path] = None) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for tag, history in fold_histories.items():
        epochs = [h["epoch"] for h in history]
        ax1.plot(epochs, [h.get("train_loss") for h in history], lw=1.4, label=f"{tag} train")
        val = [h.get("val_loss") for h in history]
        if any(v is not None for v in val):
            ax1.plot(epochs, val, lw=1.4, ls="--", label=f"{tag} val")
        pr = [h.get("val_pr_auc") for h in history]
        if any(v is not None for v in pr):
            ax2.plot(epochs, pr, lw=1.4, label=tag)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Loss")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation PR-AUC"); ax2.set_title("Validation PR-AUC")
    if len(fold_histories) <= 6:
        ax1.legend(fontsize=6); ax2.legend(fontsize=6)
    return _finish(fig, out_path)


def threshold_sweep(y_true, y_prob, chosen: float, out_path: Optional[Path] = None) -> str:
    ts = np.linspace(0.01, 0.99, 99)
    sens, spec = [], []
    y_true = np.asarray(y_true)
    for t in ts:
        pred = np.asarray(y_prob) >= t
        tp = int((pred & (y_true == 1)).sum()); fn = int((~pred & (y_true == 1)).sum())
        tn = int((~pred & (y_true == 0)).sum()); fp = int((pred & (y_true == 0)).sum())
        sens.append(tp / (tp + fn) if tp + fn else np.nan)
        spec.append(tn / (tn + fp) if tn + fp else np.nan)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(ts, sens, lw=2, label="sensitivity (infected)")
    ax.plot(ts, spec, lw=2, label="specificity (healthy)")
    ax.axvline(chosen, color="crimson", ls="--", lw=1.5, label=f"chosen = {chosen:.3f}")
    ax.set_xlabel("threshold on calibrated probability")
    ax.set_ylabel("rate")
    ax.set_title("Operating point (development OOF)")
    ax.legend(fontsize=7)
    return _finish(fig, out_path)


def score_histogram(y_true, y_prob, threshold: float, out_path: Optional[Path] = None) -> str:
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bins = np.linspace(0, 1, 21)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.65, label="actual healthy")
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.65, label="actual infected")
    ax.axvline(threshold, color="crimson", ls="--", lw=1.5, label=f"threshold {threshold:.3f}")
    ax.set_xlabel("calibrated probability of infection")
    ax.set_ylabel("images")
    ax.set_title("Score distribution")
    ax.legend(fontsize=7)
    return _finish(fig, out_path)


def error_grid(
    records: Sequence[dict], out_path: Optional[Path] = None, title: str = "Errors", max_images: int = 12
) -> Optional[str]:
    """Thumbnails of the most confident mistakes, for human review."""
    from PIL import Image

    records = list(records)[:max_images]
    if not records:
        return None
    cols = min(4, len(records))
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, records):
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img.thumbnail((320, 320))
            ax.imshow(np.asarray(img))
        except Exception:
            ax.text(0.5, 0.5, "unreadable", ha="center")
        ax.set_title(
            f"true={rec.get('label')}  p={float(rec.get('probability_infected', float('nan'))):.2f}\n"
            f"{Path(str(rec.get('image_path', ''))).name[:26]}",
            fontsize=7,
        )
    fig.suptitle(title, fontsize=10)
    return _finish(fig, out_path)
