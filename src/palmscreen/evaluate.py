"""Final test-set evaluation. Runs once, after everything else is frozen.

An append-only ledger records every evaluation of a given weights hash. Running
it a second time and picking the nicer number is exactly how a locked test set
stops being locked, so a repeat requires an explicit override flag and is
recorded in the ledger and the report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .bundle import InferenceBundle
from .config import Config
from .decision import needs_review
from .metrics import (
    bootstrap_metric_bundle,
    expected_calibration_error,
    full_metrics,
    prevalence_adjusted,
    wilson_for_rates,
)
from .splits import load_manifest

LEDGER = "artifacts/test_evaluation_ledger.jsonl"


class TestSetLockError(RuntimeError):
    """Raised on a repeat evaluation of the same weights without an override."""

    __test__ = False  # stops pytest trying to collect this as a test class


def _ledger_entries(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _append_ledger(path: str | Path, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _batch_predict(bundle: InferenceBundle, frame: pd.DataFrame, batch_size: int = 16) -> np.ndarray:
    """Raw probabilities for a frame, using the bundle's own preprocessing."""
    import torch

    from .data import make_loader

    model = bundle.model()
    loader = make_loader(frame, bundle.spec, batch_size, train=False, num_workers=0)
    out = np.zeros(len(frame), dtype=float)
    with torch.no_grad():
        for batch_x, _, batch_idx in loader:
            logits = model(batch_x).squeeze(1)
            out[np.asarray(batch_idx)] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def evaluate_locked_test(
    cfg: Config,
    bundle_dir: str | Path,
    manifest_dir: Optional[str] = None,
    ledger_path: str = LEDGER,
    allow_repeat: bool = False,
    out_dir: Optional[str] = None,
) -> dict:
    bundle = InferenceBundle.load(bundle_dir)
    weights_hash = bundle.meta["weights_sha256"]

    previous = [e for e in _ledger_entries(ledger_path) if e.get("weights_sha256") == weights_hash]
    if previous and not allow_repeat:
        raise TestSetLockError(
            f"These weights were already evaluated on the locked test set on "
            f"{previous[-1]['evaluated_at']} ({len(previous)} time(s)).\n"
            "Re-running and reporting the better result turns the test set into a "
            "validation set. If you genuinely need to repeat it (for example after "
            "fixing a data bug), pass --allow-repeat; the repeat is recorded in the "
            "ledger and printed in the report."
        )

    manifest, table = load_manifest(manifest_dir or cfg.split.manifest_dir)
    test = table[table["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("Manifest contains no test rows.")

    dev_groups = set(table[table["split"] == "dev"]["group_id"].astype(str))
    overlap = dev_groups & set(test["group_id"].astype(str))
    if overlap:
        raise RuntimeError(f"Test groups also appear in development: {sorted(overlap)[:10]}")

    print(f"Evaluating {len(test)} locked test images "
          f"({test['group_id'].nunique()} groups) with model {bundle.meta['model_version']}")

    raw = _batch_predict(bundle, test, cfg.train.batch_size)
    calibrated = bundle.calibrator.apply(raw)
    threshold = bundle.threshold.threshold
    y_true = test["label_binary"].values

    test = test.assign(
        prob_raw=raw,
        probability_infected=calibrated,
        predicted_label=np.where(calibrated >= threshold, "infected", "healthy"),
        correct=lambda d: (d["probability_infected"] >= threshold).astype(int) == y_true,
        needs_review=[needs_review(p, threshold, bundle.threshold.review_band) for p in calibrated],
    )

    metrics = full_metrics(y_true, calibrated, threshold, cfg.report.ece_bins)
    calibration = expected_calibration_error(y_true, calibrated, cfg.report.ece_bins)
    wilson = wilson_for_rates(y_true, calibrated, threshold)
    boot = bootstrap_metric_bundle(
        y_true, calibrated, test["group_id"].values, threshold,
        n_boot=cfg.report.bootstrap_iterations, seed=cfg.split.seed,
    )
    prevalence = prevalence_adjusted(
        metrics["sensitivity_infected"], metrics["specificity_healthy"], cfg.report.prevalence_grid
    )

    errors = test[~test["correct"]].copy()
    false_negatives = errors[errors["label"] == "infected"].sort_values("probability_infected")
    false_positives = errors[errors["label"] == "healthy"].sort_values("probability_infected", ascending=False)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or Path(cfg.report.output_dir) / "test" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    pred_cols = [
        "image_path", "label", "label_binary", "group_id", "prob_raw",
        "probability_infected", "predicted_label", "correct", "needs_review",
    ]
    test[pred_cols].to_csv(out / "test_predictions.csv", index=False)

    result = {
        "evaluated_at": stamp,
        "model_version": bundle.meta["model_version"],
        "weights_sha256": weights_hash,
        "bundle_dir": str(bundle_dir),
        "manifest_version": manifest["manifest_version"],
        "is_repeat_evaluation": bool(previous),
        "n_previous_evaluations": len(previous),
        "threshold": threshold,
        "threshold_source": bundle.threshold.to_dict(),
        "calibration": bundle.calibrator.to_dict(),
        "n_test_images": int(len(test)),
        "n_test_groups": int(test["group_id"].nunique()),
        "test_class_counts": test["label"].value_counts().to_dict(),
        "metrics": {k: v for k, v in metrics.items() if k != "calibration_bins"},
        "calibration_bins": calibration["bins"],
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "wilson_intervals": wilson,
        "bootstrap_intervals": boot,
        "prevalence_sensitivity": prevalence,
        "n_needs_review": int(test["needs_review"].sum()),
        "false_negatives": false_negatives[pred_cols].to_dict("records"),
        "false_positives": false_positives[pred_cols].to_dict("records"),
        "predictions_csv": str(out / "test_predictions.csv"),
        "output_dir": str(out),
    }
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2, default=str))

    _append_ledger(
        ledger_path,
        {
            "evaluated_at": stamp,
            "weights_sha256": weights_hash,
            "model_version": bundle.meta["model_version"],
            "manifest_version": manifest["manifest_version"],
            "n_test_images": int(len(test)),
            "allow_repeat_used": bool(allow_repeat),
            "headline": {
                "sensitivity_infected": metrics["sensitivity_infected"],
                "specificity_healthy": metrics["specificity_healthy"],
                "roc_auc": metrics["roc_auc"],
            },
        },
    )

    print(
        f"\nLocked test results (n={len(test)}): "
        f"sensitivity={metrics['sensitivity_infected']:.3f} "
        f"specificity={metrics['specificity_healthy']:.3f} "
        f"balanced_acc={metrics['balanced_accuracy']:.3f}"
    )
    print(f"Written to {out}")
    return result
