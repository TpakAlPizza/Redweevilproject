"""Metrics with honest uncertainty.

Two things this module refuses to do:
  * report a point estimate without an interval;
  * report precision/NPV as if field prevalence equalled dataset prevalence.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

EPS = 1e-12


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Counts and every threshold-dependent rate. `infected` is the positive class."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1_pos = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_div(2 * tn, 2 * tn + fn + fp)

    out: Dict[str, float] = dict(c)
    out.update(
        {
            "threshold": float(threshold),
            "n": int(tp + fp + tn + fn),
            "sensitivity_infected": sensitivity,
            "false_negative_rate": 1.0 - sensitivity if sensitivity == sensitivity else float("nan"),
            "specificity_healthy": specificity,
            "false_positive_rate": 1.0 - specificity if specificity == specificity else float("nan"),
            "precision_infected": precision,
            "npv_healthy": npv,
            "f1_infected": f1_pos,
            "macro_f1": float(np.nanmean([f1_pos, f1_neg])),
            "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
            "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
            "mcc": _mcc(y_true, y_pred),
        }
    )
    return out


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y_true, y_pred))


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Threshold-free discrimination and probability quality."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_true.mean()),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, object]:
    """Equal-width binned ECE plus the bins needed for a reliability plot."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    bins, ece, mce, n = [], 0.0, 0.0, len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
                 "count": 0, "mean_predicted": None, "observed_rate": None}
            )
            continue
        mean_pred = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        gap = abs(mean_pred - observed)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
            }
        )
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins, "bins": bins}


def full_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, ece_bins: int = 10
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out.update(threshold_metrics(y_true, y_prob, threshold))
    out.update(ranking_metrics(y_true, y_prob))
    cal = expected_calibration_error(y_true, y_prob, ece_bins)
    out["ece"] = cal["ece"]
    out["mce"] = cal["mce"]
    out["calibration_bins"] = cal["bins"]
    return out


# --------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """Wilson score interval: the right choice for small-n proportions such as
    sensitivity on ~26 infected test images."""
    if total == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return {
        "point": float(p),
        "lower": float(max(0.0, centre - half)),
        "upper": float(min(1.0, centre + half)),
        "n": int(total),
        "successes": int(successes),
        "method": "wilson",
        "confidence": confidence,
    }


def wilson_for_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, dict]:
    c = threshold_metrics(y_true, y_prob, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    return {
        "sensitivity_infected": wilson_interval(tp, tp + fn),
        "specificity_healthy": wilson_interval(tn, tn + fp),
        "precision_infected": wilson_interval(tp, tp + fp),
        "npv_healthy": wilson_interval(tn, tn + fn),
        "accuracy": wilson_interval(tp + tn, tp + tn + fp + fn),
    }


def group_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: Optional[Sequence] = None,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> Dict[str, float]:
    """Percentile bootstrap that resamples *groups*, not images.

    Resampling images would treat eight photos of one tree as eight independent
    observations and produce intervals that are far too narrow.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if metric_fn is None:
        raise ValueError("metric_fn is required")
    rng = np.random.default_rng(seed)

    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}

    stats_out: List[float] = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        try:
            value = metric_fn(y_true[idx], y_prob[idx])
        except Exception:
            value = float("nan")
        if value == value:
            stats_out.append(value)

    if not stats_out:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}
    arr = np.array(stats_out)
    lo = (1 - confidence) / 2 * 100
    return {
        "point": float(metric_fn(y_true, y_prob)),
        "lower": float(np.percentile(arr, lo)),
        "upper": float(np.percentile(arr, 100 - lo)),
        "n_boot": len(arr),
        "n_groups": int(len(unique)),
        "method": "group_bootstrap_percentile" if len(unique) < len(y_true) else "bootstrap_percentile",
        "confidence": confidence,
    }


def bootstrap_metric_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: Optional[Sequence],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 1337,
) -> Dict[str, dict]:
    """Group-aware intervals for the headline metrics."""

    def at(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda yt, yp: float(threshold_metrics(yt, yp, threshold)[name])

    metric_fns: Dict[str, Callable] = {
        "sensitivity_infected": at("sensitivity_infected"),
        "specificity_healthy": at("specificity_healthy"),
        "precision_infected": at("precision_infected"),
        "balanced_accuracy": at("balanced_accuracy"),
        "macro_f1": at("macro_f1"),
        "mcc": at("mcc"),
        "roc_auc": lambda yt, yp: float(ranking_metrics(yt, yp)["roc_auc"]),
        "pr_auc": lambda yt, yp: float(ranking_metrics(yt, yp)["pr_auc"]),
        "brier": lambda yt, yp: float(ranking_metrics(yt, yp)["brier"]),
    }
    return {
        name: group_bootstrap_ci(y_true, y_prob, groups, fn, n_boot=n_boot, seed=seed)
        for name, fn in metric_fns.items()
    }


# --------------------------------------------------------------------------
# Prevalence adjustment
# --------------------------------------------------------------------------

def prevalence_adjusted(
    sensitivity: float, specificity: float, prevalences: Sequence[float]
) -> List[Dict[str, float]]:
    """PPV/NPV implied by Bayes at other disease prevalences.

    Dataset prevalence here is ~50% by construction. A grove with 5% infection
    gives a very different precision for the same model.
    """
    rows = []
    for prev in prevalences:
        tp = sensitivity * prev
        fn = (1 - sensitivity) * prev
        tn = specificity * (1 - prev)
        fp = (1 - specificity) * (1 - prev)
        rows.append(
            {
                "prevalence": float(prev),
                "ppv_precision_infected": _safe_div(tp, tp + fp),
                "npv_healthy": _safe_div(tn, tn + fn),
                "expected_false_alarms_per_100_healthy": float((1 - specificity) * 100),
                "expected_missed_per_100_infected": float((1 - sensitivity) * 100),
            }
        )
    return rows
