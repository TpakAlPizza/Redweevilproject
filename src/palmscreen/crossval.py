"""Grouped cross-validation on development data, model selection, final fit.

Everything in this file touches development data only. Architecture choice,
hyperparameters, early stopping, calibration and the decision threshold are all
derived from out-of-fold predictions. The locked test split is filtered out at
the top of every entry point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .decision import fit_calibrator, select_threshold
from .metrics import full_metrics, group_bootstrap_ci, ranking_metrics
from .preprocess import PreprocessSpec
from .splits import load_manifest
from .train import train_model


def _spec_from_cfg(cfg: Config) -> PreprocessSpec:
    return PreprocessSpec(
        image_size=cfg.train.image_size,
        resize_shorter_to=int(round(cfg.train.image_size * 256 / 224)),
    )


def _dev_only(table: pd.DataFrame) -> pd.DataFrame:
    dev = table[table["split"] == "dev"].copy()
    if dev.empty:
        raise ValueError("No development rows in the manifest.")
    if "fold" not in dev.columns or (dev["fold"] < 0).any():
        raise ValueError("Manifest has no fold assignments. Re-run: palmscreen split")
    return dev


def run_cross_validation(
    cfg: Config,
    manifest_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """Train one model per fold, collect out-of-fold predictions."""
    manifest, table = load_manifest(manifest_dir or cfg.split.manifest_dir)
    dev = _dev_only(table)
    spec = _spec_from_cfg(cfg)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or Path(cfg.train.checkpoint_dir) / f"cv_{cfg.train.arch}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    folds = sorted(dev["fold"].unique())
    oof = np.full(len(dev), np.nan)
    dev = dev.reset_index(drop=True)
    fold_results: List[dict] = []

    for fold in folds:
        val_mask = dev["fold"] == fold
        train_df, val_df = dev[~val_mask], dev[val_mask]
        overlap = set(train_df["group_id"]) & set(val_df["group_id"])
        if overlap:
            raise RuntimeError(f"Fold {fold} has group overlap: {sorted(overlap)[:5]}")

        print(f"\n=== fold {fold}: train={len(train_df)} val={len(val_df)} "
              f"({train_df['group_id'].nunique()} vs {val_df['group_id'].nunique()} groups) ===")
        _, result = train_model(
            train_df, val_df, cfg, spec, out, tag=f"fold{fold}", resume=resume
        )
        # val_index refers to positions within val_df
        probs = np.full(len(val_df), np.nan)
        probs[result.val_index.astype(int)] = result.val_probs
        oof[np.flatnonzero(val_mask.values)] = probs

        fold_metrics = full_metrics(val_df["label_binary"].values, probs, 0.5, cfg.report.ece_bins)
        fold_results.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_train_groups": int(train_df["group_id"].nunique()),
                "n_val_groups": int(val_df["group_id"].nunique()),
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "stopped_early": result.stopped_early,
                "seconds": round(result.seconds, 1),
                "device": result.device,
                "checkpoint": result.checkpoint_path,
                "metrics_at_0.5": {k: v for k, v in fold_metrics.items() if k != "calibration_bins"},
                "history": result.history,
            }
        )

    if np.isnan(oof).any():
        raise RuntimeError("Some development images have no out-of-fold prediction.")

    dev["oof_prob_raw"] = oof
    oof_path = out / "oof_predictions.csv"
    dev[["image_path", "label", "label_binary", "group_id", "fold", "oof_prob_raw"]].to_csv(
        oof_path, index=False
    )

    oof_rank = ranking_metrics(dev["label_binary"].values, oof)
    summary = {
        "arch": cfg.train.arch,
        "created_at": stamp,
        "manifest_version": manifest["manifest_version"],
        "config_fingerprint": cfg.fingerprint(),
        "n_dev_images": int(len(dev)),
        "n_dev_groups": int(dev["group_id"].nunique()),
        "n_folds": len(folds),
        "folds": fold_results,
        "oof_csv": str(oof_path),
        "oof_ranking_metrics": oof_rank,
        "oof_pr_auc_ci": group_bootstrap_ci(
            dev["label_binary"].values, oof, dev["group_id"].values,
            lambda yt, yp: float(ranking_metrics(yt, yp)["pr_auc"]),
            n_boot=cfg.report.bootstrap_iterations, seed=cfg.split.seed,
        ),
        "per_fold_summary": _per_fold_stats(fold_results),
        "note": (
            "These are development cross-validation numbers. They are used to "
            "choose the model, threshold and calibrator, and are therefore "
            "optimistic. They are not test results and not field performance."
        ),
        "output_dir": str(out),
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nCV complete. OOF PR-AUC={oof_rank['pr_auc']:.3f} ROC-AUC={oof_rank['roc_auc']:.3f}")
    print(f"Artifacts: {out}")
    return summary


def _per_fold_stats(fold_results: List[dict]) -> dict:
    def collect(key: str) -> List[float]:
        return [f["metrics_at_0.5"].get(key, float("nan")) for f in fold_results]

    out = {}
    for key in ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_infected", "specificity_healthy"):
        values = np.array(collect(key), dtype=float)
        out[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "per_fold": [float(v) for v in values],
        }
    out["median_best_epoch"] = int(np.median([f["best_epoch"] for f in fold_results]))
    return out


# --------------------------------------------------------------------------
# Model selection (development data only)
# --------------------------------------------------------------------------

PARAM_COUNTS = {"efficientnet_b0": 5_288_548, "mobilenet_v3_large": 5_483_032}


def compare_candidates(
    summaries: List[dict], metric: str = "pr_auc", n_boot: int = 2000, seed: int = 1337
) -> dict:
    """Paired, group-aware bootstrap over out-of-fold predictions.

    If the interval for the difference includes zero, the candidates are
    statistically indistinguishable on this dataset and the smaller/faster model
    wins by default.
    """
    if len(summaries) < 2:
        return {"decision": summaries[0]["arch"] if summaries else None, "reason": "only one candidate"}

    frames = {s["arch"]: pd.read_csv(s["oof_csv"]) for s in summaries}
    archs = list(frames)
    base = frames[archs[0]][["image_path", "label_binary", "group_id"]]
    merged = base.copy()
    for arch, frame in frames.items():
        merged = merged.merge(
            frame[["image_path", "oof_prob_raw"]].rename(columns={"oof_prob_raw": arch}),
            on="image_path", how="inner",
        )
    if len(merged) != len(base):
        raise ValueError("Candidates were evaluated on different images; use the same split manifest.")

    def score(y, p):
        return float(ranking_metrics(y, p)[metric])

    rng = np.random.default_rng(seed)
    groups = merged["group_id"].values
    unique = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    y = merged["label_binary"].values

    point = {a: score(y, merged[a].values) for a in archs}
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        try:
            diffs.append(score(y[idx], merged[archs[0]].values[idx]) - score(y[idx], merged[archs[1]].values[idx]))
        except Exception:
            continue
    diffs = np.array(diffs)
    lo, hi = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) else (float("nan"),) * 2
    indistinguishable = not (lo > 0 or hi < 0)

    if indistinguishable:
        winner = min(archs, key=lambda a: PARAM_COUNTS.get(a, float("inf")))
        reason = (
            f"The 95% interval for the difference in {metric} ({lo:+.3f}, {hi:+.3f}) includes zero, "
            f"so the candidates are statistically indistinguishable on {len(merged)} development "
            f"images across {len(unique)} groups. Chose the smaller/faster model."
        )
    else:
        winner = archs[0] if point[archs[0]] > point[archs[1]] else archs[1]
        reason = (
            f"{winner} is better on development out-of-fold {metric}; the 95% interval for the "
            f"difference ({lo:+.3f}, {hi:+.3f}) excludes zero."
        )

    return {
        "decision": winner,
        "reason": reason,
        "metric": metric,
        "point_estimates": point,
        "paired_difference_ci": {"comparison": f"{archs[0]} - {archs[1]}", "lower": lo, "upper": hi},
        "n_images": int(len(merged)),
        "n_groups": int(len(unique)),
        "selected_on": "development_out_of_fold_predictions",
    }


# --------------------------------------------------------------------------
# Final fit + calibration + threshold -> bundle
# --------------------------------------------------------------------------

def finalize(
    cfg: Config,
    cv_summary: dict,
    bundles_root: str = "artifacts/bundles",
    manifest_dir: Optional[str] = None,
) -> Tuple[Path, dict]:
    """Refit on all development data, then freeze calibration and threshold.

    Epoch count comes from the median best epoch across folds rather than early
    stopping, because holding out a validation set here would either shrink the
    training data or reuse data already used for selection.
    """
    from .bundle import InferenceBundle, promote_to_latest

    manifest, table = load_manifest(manifest_dir or cfg.split.manifest_dir)
    dev = _dev_only(table).reset_index(drop=True)
    spec = _spec_from_cfg(cfg)

    oof = pd.read_csv(cv_summary["oof_csv"])
    calibrator = fit_calibrator(
        oof["label_binary"].values, oof["oof_prob_raw"].values, cfg.calibration.method
    )
    calibrated_oof = calibrator.apply(oof["oof_prob_raw"].values)
    decision = select_threshold(
        oof["label_binary"].values,
        calibrated_oof,
        policy=cfg.threshold.policy,
        target_recall=cfg.threshold.target_recall,
        fixed_threshold=cfg.threshold.fixed_threshold,
        review_band=cfg.threshold.review_band,
    )
    print(f"Calibration: {calibrator.method} (n={calibrator.n_fit})")
    print(f"Threshold: {decision.threshold:.4f} via {decision.policy} "
          f"(dev OOF recall={decision.achieved_recall:.3f}, specificity={decision.achieved_specificity:.3f})")

    median_epochs = cv_summary["per_fold_summary"]["median_best_epoch"]
    finetune_epochs = max(1, median_epochs - cfg.train.head_epochs + 1)
    run_dir = Path(cfg.train.checkpoint_dir) / f"final_{cfg.train.arch}_{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Refitting on all {len(dev)} development images for {cfg.train.head_epochs}+{finetune_epochs} epochs")
    model, result = train_model(
        dev, None, cfg, spec, run_dir, tag="final", max_epochs_override=finetune_epochs
    )

    version = f"{cfg.model_version}+{cfg.train.arch}.{time.strftime('%Y%m%d%H%M%S')}"
    bundle_dir = Path(bundles_root) / f"palmscreen_{version}"
    oof_with_cal = oof.assign(oof_prob_calibrated=calibrated_oof)
    InferenceBundle.save(
        bundle_dir, model, cfg, spec, calibrator, decision, version,
        extra={
            "training_summary": {
                "n_dev_images": int(len(dev)),
                "n_dev_groups": int(dev["group_id"].nunique()),
                "class_counts": dev["label"].value_counts().to_dict(),
                "manifest_version": manifest["manifest_version"],
                "grouping": manifest["grouping"],
                "cv_summary_dir": cv_summary["output_dir"],
                "final_fit": result.summary(),
                "dev_oof_pr_auc": cv_summary["oof_ranking_metrics"]["pr_auc"],
            },
            "selection_note": (
                "Architecture, calibration and threshold were selected using "
                "development out-of-fold predictions only. The locked test set "
                "was not consulted."
            ),
        },
    )
    oof_with_cal.to_csv(bundle_dir / "dev_oof_predictions.csv", index=False)
    promote_to_latest(bundle_dir, bundles_root)
    print(f"Bundle written: {bundle_dir}")
    return bundle_dir, {
        "bundle_dir": str(bundle_dir),
        "model_version": version,
        "calibration": calibrator.to_dict(),
        "threshold": decision.to_dict(),
        "final_fit": result.summary(),
    }
