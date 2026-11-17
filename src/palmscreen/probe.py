"""Shortcut probes.

If a linear model on average colour, or on an 8x8 thumbnail, or on nothing but
the file's dimensions, separates your classes as well as the trained network
does, then the network is not detecting disease. It is detecting how the two
sets of photographs were produced.

These probes deliberately destroy fine detail. Frond lesions, weevil damage and
crown texture cannot survive an 8x8 greyscale downsample. Whatever still
separates the classes after that is not the disease.

Run on development data only, using the folds from the split manifest, so the
numbers are directly comparable to the CV numbers from `palmscreen crossval`.

    palmscreen probe --config configs/effnet_b0.yaml
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .metrics import ranking_metrics
from .preprocess import load_image
from .splits import load_manifest

PROBES = ("mean_rgb", "colour_histogram", "tiny_greyscale_8x8", "acquisition_metadata")


def _image_features(path: str) -> Dict[str, np.ndarray]:
    """Every probe feature for one image, computed in a single decode."""
    img = load_image(path)
    size_bytes = Path(path).stat().st_size
    width, height = img.size
    rgb = img.convert("RGB")

    small = np.asarray(rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    mean_rgb = small.reshape(-1, 3).mean(axis=0)

    hist = np.concatenate([
        np.histogram(small[..., c], bins=16, range=(0, 255), density=True)[0]
        for c in range(3)
    ])

    tiny = np.asarray(
        rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32
    ).flatten() / 255.0

    acquisition = np.array([
        width, height, width / max(height, 1), size_bytes / 1000.0,
        width * height / 1e6,
    ], dtype=np.float32)

    return {
        "mean_rgb": mean_rgb,
        "colour_histogram": hist,
        "tiny_greyscale_8x8": tiny,
        "acquisition_metadata": acquisition,
    }


def _build_matrices(frame: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
    rows: Dict[str, List[np.ndarray]] = {name: [] for name in PROBES}
    failed: List[str] = []
    for path in frame["image_path"]:
        try:
            features = _image_features(path)
        except Exception as exc:
            failed.append(f"{path}: {type(exc).__name__}")
            features = {name: None for name in PROBES}
        for name in PROBES:
            rows[name].append(features[name])

    matrices = {}
    for name in PROBES:
        valid = [v for v in rows[name] if v is not None]
        if not valid:
            continue
        width = len(valid[0])
        filled = [v if v is not None else np.zeros(width, dtype=np.float32) for v in rows[name]]
        matrices[name] = np.vstack(filled)
    return matrices, failed


def _oof_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold probabilities from a small logistic regression.

    The same grouped folds the network used, so the comparison is like for like.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        train_mask = folds != fold
        val_mask = ~train_mask
        if len(np.unique(y[train_mask])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        model.fit(X[train_mask], y[train_mask])
        oof[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    return oof


def _verdict(pr_auc: float) -> str:
    if pr_auc != pr_auc:
        return "not computed"
    if pr_auc >= 0.90:
        return "SHORTCUT: this alone nearly solves the task"
    if pr_auc >= 0.75:
        return "strong non-disease signal present"
    if pr_auc >= 0.65:
        return "some non-disease signal"
    return "little signal (expected)"


def run_probes(
    cfg: Config, manifest_dir: Optional[str] = None, out_dir: Optional[str] = None
) -> dict:
    manifest, table = load_manifest(manifest_dir or cfg.split.manifest_dir)
    dev = table[table["split"] == "dev"].reset_index(drop=True)
    if dev.empty:
        raise ValueError("No development rows in the manifest. Run: palmscreen split")
    if "fold" not in dev.columns or (dev["fold"] < 0).any():
        raise ValueError("Manifest has no fold assignments. Run: palmscreen split")

    y = dev["label_binary"].to_numpy()
    folds = dev["fold"].to_numpy()
    groups = dev["group_id"].to_numpy()

    print(f"Probing {len(dev)} development images across {len(np.unique(folds))} grouped folds...")
    matrices, failed = _build_matrices(dev)

    results = []
    for name, X in matrices.items():
        oof = _oof_scores(X, y, folds)
        usable = ~np.isnan(oof)
        rank = ranking_metrics(y[usable], oof[usable])
        results.append({
            "probe": name,
            "n_features": int(X.shape[1]),
            "oof_roc_auc": rank["roc_auc"],
            "oof_pr_auc": rank["pr_auc"],
            "verdict": _verdict(rank["pr_auc"]),
        })

    # Negative control: shuffle labels between groups. A working evaluation must
    # collapse to chance here. If it does not, the split or the folds are broken
    # and every other number in the project is suspect.
    #
    # Group sizes are often very unequal (one 44-photo burst against seventy
    # single frames). A naive permutation can then hand 97% of the images one
    # label, and a PR-AUC of 0.99 follows from the prevalence alone rather than
    # from any leakage. So candidate permutations are drawn and the one whose
    # image-level prevalence is closest to the real data is used.
    rng = np.random.default_rng(cfg.split.seed)
    unique_groups = np.unique(groups)
    group_labels = {g: y[groups == g][0] for g in unique_groups}
    target_rate = float(y.mean())

    best_map, best_gap = None, None
    for _ in range(500):
        values = rng.permutation(list(group_labels.values()))
        candidate = dict(zip(unique_groups, values))
        rate = float(np.array([candidate[g] for g in groups]).mean())
        gap = abs(rate - target_rate)
        if best_gap is None or gap < best_gap:
            best_map, best_gap = candidate, gap
        if gap < 0.02:
            break
    shuffled_map = best_map
    y_shuffled = np.array([shuffled_map[g] for g in groups])

    control_name = "colour_histogram" if "colour_histogram" in matrices else next(iter(matrices))
    control_oof = _oof_scores(matrices[control_name], y_shuffled, folds)
    usable = ~np.isnan(control_oof)
    control_rank = ranking_metrics(y_shuffled[usable], control_oof[usable])
    n_groups_per_class = {
        "infected": int(len({g for g in unique_groups if group_labels[g] == 1})),
        "healthy": int(len({g for g in unique_groups if group_labels[g] == 0})),
    }
    too_few_groups = min(n_groups_per_class.values()) < int(len(np.unique(folds)))

    control = {
        "probe": f"{control_name} (labels shuffled - negative control)",
        "oof_roc_auc": control_rank["roc_auc"],
        "oof_pr_auc": control_rank["pr_auc"],
        "shuffled_positive_rate": float(y_shuffled.mean()),
        "real_positive_rate": target_rate,
        "groups_per_class": n_groups_per_class,
        "expected": "ROC-AUC near 0.50",
        "ok": bool(0.30 <= control_rank["roc_auc"] <= 0.70)
        if control_rank["roc_auc"] == control_rank["roc_auc"] else False,
        "unreliable": bool(too_few_groups),
    }

    worst = max((r["oof_pr_auc"] for r in results if r["oof_pr_auc"] == r["oof_pr_auc"]), default=float("nan"))
    report = {
        "generated_at": time.strftime("%Y%m%d-%H%M%S"),
        "manifest_version": manifest["manifest_version"],
        "grouping": manifest["grouping"],
        "n_dev_images": int(len(dev)),
        "n_dev_groups": int(len(unique_groups)),
        "n_folds": int(len(np.unique(folds))),
        "probes": results,
        "negative_control": control,
        "highest_probe_pr_auc": float(worst),
        "unreadable": failed,
        "interpretation": _interpretation(worst, manifest["grouping"], control),
    }

    out = Path(out_dir or Path(cfg.report.output_dir) / "probe" / report["generated_at"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(results).to_csv(out / "probe_results.csv", index=False)

    _print_report(report)
    print(f"\nWritten to {out}")
    return report


def _interpretation(worst: float, grouping: dict, control: dict) -> List[str]:
    notes: List[str] = []
    if control.get("unreliable"):
        counts = control.get("groups_per_class", {})
        notes.append(
            f"NEGATIVE CONTROL IS UNRELIABLE HERE: one class has very few independent "
            f"groups ({counts}). With so few groups the shuffled control cannot reach a "
            "balanced label assignment, so its score says nothing about leakage. Fix the "
            "grouping first: this is a data problem, not a pipeline fault."
        )
    elif not control.get("ok"):
        notes.append(
            "NEGATIVE CONTROL FAILED: shuffled labels did not collapse to chance. "
            "Something is wrong with the split or the folds themselves. Fix this "
            "before interpreting anything else."
        )
    if worst != worst:
        return notes or ["No probe produced a score."]

    if worst >= 0.90:
        notes.append(
            f"A trivial linear model reaches {worst:.3f} PR-AUC. The classes are separable "
            "without any disease-level detail, so a high score from the network is not "
            "evidence that it detects infection. Fix the dataset before tuning the model: "
            "matching the photography conditions between classes matters far more than "
            "architecture, epochs or augmentation."
        )
    elif worst >= 0.75:
        notes.append(
            f"A trivial model reaches {worst:.3f} PR-AUC. Some of the network's performance "
            "is attributable to acquisition differences rather than disease. Treat the gap "
            "between this and the network's score as the part that might be real."
        )
    else:
        notes.append(
            f"Trivial baselines top out at {worst:.3f} PR-AUC, so global colour, coarse layout "
            "and file properties do not solve the task on their own. That is a good sign, "
            "though it does not prove the network uses disease morphology either."
        )

    if grouping.get("trust") != "strong":
        notes.append(
            f"Grouping trust is '{grouping.get('trust')}' ({grouping.get('strategy')}). "
            "Photos of the same palm may sit in both training and validation folds, which "
            "inflates the probe scores and the network scores alike. Supply tree_id metadata "
            "and re-split before drawing conclusions from either."
        )
    return notes


def _print_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print("SHORTCUT PROBES - development out-of-fold, same grouped folds as CV")
    print("=" * 72)
    print(f"{'probe':<34}{'ROC-AUC':>9}{'PR-AUC':>9}   verdict")
    print("-" * 72)
    for row in report["probes"]:
        print(f"{row['probe']:<34}{row['oof_roc_auc']:>9.3f}{row['oof_pr_auc']:>9.3f}   {row['verdict']}")
    control = report["negative_control"]
    print("-" * 72)
    if control.get("unreliable"):
        status = "UNRELIABLE - too few groups in one class"
    elif control["ok"]:
        status = "ok"
    else:
        status = "FAILED - expected ~0.50"
    print(f"{control['probe']:<34}{control['oof_roc_auc']:>9.3f}{control['oof_pr_auc']:>9.3f}   {status}")
    print(f"{'  (shuffled prevalence)':<34}{control['shuffled_positive_rate']:>9.3f}"
          f"{control['real_positive_rate']:>9.3f}   shuffled vs real positive rate")
    print("=" * 72)
    for note in report["interpretation"]:
        print("\n! " + note)
