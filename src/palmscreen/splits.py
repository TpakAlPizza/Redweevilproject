"""Splitting, leakage assertions, and the versioned split manifest.

Rules enforced here (they are assertions, not comments):
  * no group_id appears in more than one split;
  * no SHA-256 appears in more than one split;
  * no perceptual-hash near-duplicate pair straddles a split;
  * the test split is written once and locked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .hashing import hamming


class LeakageError(RuntimeError):
    """Raised when train/test independence is violated. Never caught."""


# --------------------------------------------------------------------------
# Test-set construction
# --------------------------------------------------------------------------

def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group_id").agg(
        n=("label_binary", "size"), n_pos=("label_binary", "sum")
    )
    g["pos_rate"] = g["n_pos"] / g["n"]
    return g.reset_index()


def group_stratified_holdout(
    df: pd.DataFrame, test_fraction: float, seed: int, min_per_class: int = 5
) -> Tuple[List[str], Dict[str, float]]:
    """Greedily choose whole groups for the test set, keeping class balance.

    Whole groups move together, so a tree photographed eight times contributes
    all eight images to exactly one side of the split.

    A test set missing a class cannot produce a sensitivity or a specificity, so
    both classes are guaranteed a minimum presence even when that means
    overshooting the target size. When group sizes are very unequal -- one group
    of 44 photos against seventy groups of one -- the size target alone will
    happily select a single group and call it a test set.
    """
    summary = _group_summary(df).sample(frac=1.0, random_state=seed)
    n_total = len(df)
    n_pos_total = int(df["label_binary"].sum())
    target_n = max(1, int(round(n_total * test_fraction)))
    target_pos_rate = n_pos_total / n_total if n_total else 0.0

    chosen: List[str] = []
    cur_n = cur_pos = 0
    remaining = summary.to_dict("records")

    while cur_n < target_n and remaining:
        best_idx, best_cost = None, None
        for i, grp in enumerate(remaining):
            new_n = cur_n + grp["n"]
            if new_n > target_n * 1.35:  # never wildly overshoot the target size
                continue
            new_pos = cur_pos + grp["n_pos"]
            rate = new_pos / new_n
            # Cost: distance from the target prevalence + distance from target size.
            cost = abs(rate - target_pos_rate) + abs(new_n - target_n) / max(target_n, 1)
            if best_cost is None or cost < best_cost:
                best_idx, best_cost = i, cost
        if best_idx is None:
            break
        grp = remaining.pop(best_idx)
        chosen.append(str(grp["group_id"]))
        cur_n += int(grp["n"])
        cur_pos += int(grp["n_pos"])

    # Guarantee both classes are represented, adding the smallest useful groups
    # of any missing class rather than leaving the split unusable.
    chosen_set = set(chosen)
    forced: List[str] = []
    for class_value in (1, 0):
        present = int(
            df[df["group_id"].astype(str).isin(chosen_set)]["label_binary"].eq(class_value).sum()
        )
        if present >= min_per_class:
            continue
        candidates = (
            _group_summary(df[df["label_binary"] == class_value])
            .assign(gid=lambda d: d["group_id"].astype(str))
            .query("gid not in @chosen_set")
            .sort_values("n", ascending=False)
        )
        for _, grp in candidates.iterrows():
            if present >= min_per_class:
                break
            gid = str(grp["group_id"])
            chosen.append(gid)
            chosen_set.add(gid)
            forced.append(gid)
            present += int(grp["n"])
            cur_n += int(grp["n"])
            cur_pos += int(grp["n_pos"])

    final = df[df["group_id"].astype(str).isin(chosen_set)]
    n_pos_test = int(final["label_binary"].sum())
    n_neg_test = int(len(final) - n_pos_test)
    if n_pos_test == 0 or n_neg_test == 0:
        raise ValueError(
            f"Cannot build a two-class test set: {n_pos_test} infected and {n_neg_test} "
            "healthy images available after grouping. Both classes need at least "
            f"{min_per_class} images across separate groups. This usually means one "
            "class collapsed into very few groups -- check the group counts per class."
        )

    stats = {
        "target_n": target_n,
        "actual_n": cur_n,
        "n_infected": n_pos_test,
        "n_healthy": n_neg_test,
        "forced_groups_for_class_balance": forced,
        "actual_pos": cur_pos,
        "actual_pos_rate": (cur_pos / cur_n) if cur_n else 0.0,
        "overall_pos_rate": target_pos_rate,
    }
    return chosen, stats


def make_test_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    """Assign every row to 'dev' or 'test'. Returns df with a 'split' column."""
    df = df.copy()
    info: Dict[str, object] = {}

    col, values = cfg.data.holdout_column, cfg.data.holdout_values
    if col and values:
        if col not in df.columns:
            raise ValueError(f"holdout_column '{col}' is not in the catalog")
        mask = df[col].astype(str).isin([str(v) for v in values])
        if not mask.any():
            raise ValueError(f"No rows match holdout {col} in {values}")
        df["split"] = np.where(mask, "test", "dev")
        info["strategy"] = f"explicit_holdout:{col}={values}"
        info["note"] = (
            "Test set is an entire unseen "
            f"{col}. This is the strongest split available and the only one that "
            "approaches an estimate of generalisation to new collection conditions."
        )
    else:
        chosen, stats = group_stratified_holdout(df, cfg.split.test_fraction, cfg.split.seed)
        df["split"] = np.where(df["group_id"].astype(str).isin(chosen), "test", "dev")
        info["strategy"] = "group_stratified_random_holdout"
        info["stats"] = stats
        info["note"] = (
            "Test groups were drawn at random from the same collection. Unseen "
            "trees, yes; unseen farm/camera/season, no. Performance on a genuinely "
            "new site will typically be lower."
        )

    for split in ("dev", "test"):
        sub = df[df["split"] == split]
        info[f"n_{split}"] = int(len(sub))
        info[f"n_{split}_infected"] = int(sub["label_binary"].sum())
        info[f"n_{split}_groups"] = int(sub["group_id"].nunique())
    return df, info


# --------------------------------------------------------------------------
# Development folds
# --------------------------------------------------------------------------

def make_folds(dev_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Grouped, stratified K folds over the development set only."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = dev_df["group_id"].nunique()
    n_splits = min(cfg.split.n_folds, n_groups)
    if n_splits < 2:
        raise ValueError(f"Need >= 2 dev groups to cross-validate, found {n_groups}")
    if n_splits < cfg.split.n_folds:
        import warnings

        warnings.warn(f"Only {n_groups} dev groups; reduced to {n_splits} folds.")

    per_class_groups = dev_df.groupby("label_binary")["group_id"].nunique()
    scarce = per_class_groups[per_class_groups < n_splits]
    if len(scarce):
        import warnings

        names = {1: "infected", 0: "healthy"}
        detail = ", ".join(f"{names.get(k, k)}: {v} groups" for k, v in scarce.items())
        warnings.warn(
            f"A class has fewer independent groups than folds ({detail}, {n_splits} folds). "
            "Some folds will contain almost no examples of that class, which makes "
            "per-fold metrics unstable and the negative control unreliable. Either "
            "collect more independent examples or reduce n_folds."
        )

    dev = dev_df.copy().reset_index(drop=True)
    dev["fold"] = -1
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.split.seed
    )
    for fold, (_, val_idx) in enumerate(
        splitter.split(dev, dev["label_binary"], groups=dev["group_id"])
    ):
        dev.loc[val_idx, "fold"] = fold
    if (dev["fold"] < 0).any():
        raise RuntimeError("Some dev rows were not assigned a fold")
    return dev


# --------------------------------------------------------------------------
# Leakage assertions
# --------------------------------------------------------------------------

@dataclass
class LeakageReport:
    group_overlap: List[str]
    sha256_overlap: List[str]
    near_duplicate_pairs: List[Tuple[str, str, int]]
    fold_group_overlap: List[str]

    @property
    def clean(self) -> bool:
        return not (
            self.group_overlap
            or self.sha256_overlap
            or self.near_duplicate_pairs
            or self.fold_group_overlap
        )

    def as_dict(self) -> dict:
        return {
            "clean": self.clean,
            "group_overlap": self.group_overlap,
            "sha256_overlap": self.sha256_overlap,
            "near_duplicate_pairs_across_splits": [
                {"a": a, "b": b, "hamming": d} for a, b, d in self.near_duplicate_pairs
            ],
            "fold_group_overlap": self.fold_group_overlap,
        }


def audit_leakage(df: pd.DataFrame, near_dup_distance: int = 6) -> LeakageReport:
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]

    group_overlap = sorted(
        set(dev["group_id"].astype(str)) & set(test["group_id"].astype(str))
    )
    sha_overlap = []
    if "sha256" in df.columns:
        sha_overlap = sorted(
            {s for s in dev["sha256"].dropna()} & {s for s in test["sha256"].dropna()}
        )

    near_pairs: List[Tuple[str, str, int]] = []
    if "phash" in df.columns:
        dev_h = dev[["image_path", "phash"]].dropna().values.tolist()
        test_h = test[["image_path", "phash"]].dropna().values.tolist()
        for p1, h1 in dev_h:
            for p2, h2 in test_h:
                d = hamming(h1, h2)
                if d <= near_dup_distance:
                    near_pairs.append((p1, p2, d))

    fold_overlap: List[str] = []
    if "fold" in dev.columns:
        by_fold = {
            f: set(sub["group_id"].astype(str))
            for f, sub in dev[dev["fold"] >= 0].groupby("fold")
        }
        folds = sorted(by_fold)
        for i, fi in enumerate(folds):
            for fj in folds[i + 1 :]:
                shared = by_fold[fi] & by_fold[fj]
                fold_overlap.extend(f"fold{fi}~fold{fj}:{g}" for g in sorted(shared))

    return LeakageReport(group_overlap, sha_overlap, near_pairs, fold_overlap)


def assert_no_leakage(df: pd.DataFrame, near_dup_distance: int = 6) -> LeakageReport:
    report = audit_leakage(df, near_dup_distance)
    if not report.clean:
        raise LeakageError(
            "Train/test independence violated:\n"
            + json.dumps(report.as_dict(), indent=2)[:4000]
        )
    return report


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "image_path",
    "label",
    "label_binary",
    "group_id",
    "split",
    "fold",
    "sha256",
    "phash",
    "width",
    "height",
]


def write_manifest(
    df: pd.DataFrame,
    cfg: Config,
    group_resolution: dict,
    split_info: dict,
    leakage: LeakageReport,
    manifest_dir: Optional[str] = None,
) -> Path:
    """Versioned, immutable record of exactly which image went where."""
    from .runtime import provenance

    out_dir = Path(manifest_dir or cfg.split.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    version = f"{stamp}_seed{cfg.split.seed}_{cfg.fingerprint()}"

    cols = [c for c in MANIFEST_COLUMNS if c in df.columns]
    table = df[cols].sort_values("image_path")
    csv_path = out_dir / f"split_{version}.csv"
    table.to_csv(csv_path, index=False)

    manifest = {
        "manifest_version": version,
        "created_at": stamp,
        "seed": cfg.split.seed,
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.to_dict(),
        "grouping": group_resolution,
        "split_info": split_info,
        "leakage_audit": leakage.as_dict(),
        "counts": _counts(df),
        "csv": csv_path.name,
        "provenance": provenance(),
    }
    json_path = out_dir / f"split_{version}.json"
    json_path.write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "latest.json").write_text(json.dumps({"manifest": json_path.name}, indent=2))
    return json_path


def _counts(df: pd.DataFrame) -> dict:
    out: Dict[str, object] = {"total_images": int(len(df))}
    for split, sub in df.groupby("split"):
        entry = {
            "n_images": int(len(sub)),
            "n_groups": int(sub["group_id"].nunique()),
            "by_class": sub["label"].value_counts().to_dict(),
            "groups_by_class": {
                str(label): int(g["group_id"].nunique())
                for label, g in sub.groupby("label")
            },
        }
        if "fold" in sub.columns and (sub["fold"] >= 0).any():
            entry["folds"] = {
                str(int(f)): {
                    "n_images": int(len(s)),
                    "n_groups": int(s["group_id"].nunique()),
                    "n_infected": int(s["label_binary"].sum()),
                }
                for f, s in sub[sub["fold"] >= 0].groupby("fold")
            }
        out[str(split)] = entry
    return out


def load_manifest(manifest_dir: str | Path, version: Optional[str] = None) -> Tuple[dict, pd.DataFrame]:
    d = Path(manifest_dir)
    if version:
        json_path = d / f"split_{version}.json"
    else:
        latest = d / "latest.json"
        if not latest.exists():
            raise FileNotFoundError(
                f"No split manifest in {d}. Run: palmscreen split --config <cfg>"
            )
        json_path = d / json.loads(latest.read_text())["manifest"]
    manifest = json.loads(json_path.read_text())
    table = pd.read_csv(d / manifest["csv"])
    return manifest, table
