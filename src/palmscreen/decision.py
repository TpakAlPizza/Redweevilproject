"""Calibration and threshold selection.

Both are fitted on development out-of-fold predictions, serialised as plain
JSON numbers (no pickles), and shipped inside the inference bundle so the API
applies exactly the same mapping the report describes.

Raw sigmoid output is *not* confidence. It becomes a probability only after
the calibrator here, and even then it is a calibrated estimate on this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass
class Calibrator:
    """JSON-serialisable probability calibrator."""

    method: str  # platt | temperature | isotonic | none
    params: Dict[str, object]
    fitted_on: str = "dev_out_of_fold"
    n_fit: int = 0

    def apply(self, prob: np.ndarray) -> np.ndarray:
        prob = np.asarray(prob, dtype=float)
        if self.method == "none":
            return prob
        if self.method == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            return _sigmoid(a * _logit(prob) + b)
        if self.method == "temperature":
            t = float(self.params["temperature"])
            return _sigmoid(_logit(prob) / max(t, EPS))
        if self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=float)
            ys = np.asarray(self.params["y"], dtype=float)
            return np.interp(prob, xs, ys, left=ys[0], right=ys[-1])
        raise ValueError(f"Unknown calibration method: {self.method}")

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "params": self.params,
            "fitted_on": self.fitted_on,
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(
            method=d["method"],
            params=d.get("params", {}),
            fitted_on=d.get("fitted_on", "unknown"),
            n_fit=int(d.get("n_fit", 0)),
        )


def fit_calibrator(
    y_true: Sequence[int], y_prob: Sequence[float], method: str = "platt"
) -> Calibrator:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    if method == "none" or len(np.unique(y_true)) < 2:
        return Calibrator("none", {}, n_fit=n)

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(y_prob).reshape(-1, 1), y_true)
        return Calibrator(
            "platt",
            {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])},
            n_fit=n,
        )

    if method == "temperature":
        from scipy.optimize import minimize_scalar

        logits = _logit(y_prob)

        def nll(t: float) -> float:
            p = np.clip(_sigmoid(logits / max(t, EPS)), EPS, 1 - EPS)
            return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        return Calibrator("temperature", {"temperature": float(res.x)}, n_fit=n)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        if n < 300:
            import warnings

            warnings.warn(
                f"Isotonic calibration on {n} points overfits badly. "
                "Platt scaling is recommended below ~300 development images."
            )
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(y_prob, y_true)
        grid = np.linspace(0.0, 1.0, 201)
        return Calibrator(
            "isotonic",
            {"x": grid.tolist(), "y": iso.predict(grid).tolist()},
            n_fit=n,
        )

    raise ValueError(f"Unknown calibration method: {method}")


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------

@dataclass
class ThresholdDecision:
    threshold: float
    policy: str
    target_recall: Optional[float]
    achieved_recall: float
    achieved_specificity: float
    review_band: float
    selected_on: str = "dev_out_of_fold"
    notes: List[str] = None

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "policy": self.policy,
            "target_recall": self.target_recall,
            "achieved_recall_dev_oof": self.achieved_recall,
            "achieved_specificity_dev_oof": self.achieved_specificity,
            "review_band": self.review_band,
            "selected_on": self.selected_on,
            "notes": self.notes or [],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdDecision":
        return cls(
            threshold=float(d["threshold"]),
            policy=d.get("policy", "unknown"),
            target_recall=d.get("target_recall"),
            achieved_recall=float(d.get("achieved_recall_dev_oof", float("nan"))),
            achieved_specificity=float(d.get("achieved_specificity_dev_oof", float("nan"))),
            review_band=float(d.get("review_band", 0.0)),
            selected_on=d.get("selected_on", "unknown"),
            notes=d.get("notes", []),
        )


def _rates(y_true: np.ndarray, y_prob: np.ndarray, t: float):
    pred = y_prob >= t
    tp = int((pred & (y_true == 1)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    fp = int((pred & (y_true == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return sens, spec, tp, fp


def candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    """Midpoints between observed scores, plus the endpoints."""
    uniq = np.unique(np.asarray(y_prob, dtype=float))
    mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
    return np.unique(np.concatenate([[0.0], mids, uniq, [1.0]]))


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    policy: str = "max_specificity_at_recall",
    target_recall: float = 0.90,
    fixed_threshold: float = 0.50,
    review_band: float = 0.10,
) -> ThresholdDecision:
    """Choose the operating point from development OOF predictions only.

    Default policy: among thresholds meeting the infected-recall target, take the
    one with the highest specificity. Missing an infected palm costs a tree;
    a false alarm costs an inspection.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    notes: List[str] = []

    if policy == "fixed":
        sens, spec, _, _ = _rates(y_true, y_prob, fixed_threshold)
        notes.append("Fixed threshold supplied by config; not data-driven.")
        return ThresholdDecision(
            float(fixed_threshold), policy, None, sens, spec, review_band, notes=notes
        )

    grid = candidate_thresholds(y_prob)
    scored = [(t, *_rates(y_true, y_prob, t)) for t in grid]

    if policy == "max_specificity_at_recall":
        eligible = [s for s in scored if s[1] == s[1] and s[1] >= target_recall]
        if eligible:
            # Highest specificity; break ties by the higher threshold (fewer alarms).
            best = max(eligible, key=lambda s: (s[2], s[0]))
            notes.append(
                f"Chosen as the most specific threshold reaching >= {target_recall:.0%} "
                "infected recall on development out-of-fold predictions."
            )
        else:
            best = max(scored, key=lambda s: (s[1] if s[1] == s[1] else -1, s[2]))
            notes.append(
                f"WARNING: no threshold reached the {target_recall:.0%} recall target "
                "on development data. Fell back to the highest achievable recall. "
                "The model is probably not good enough for this policy yet."
            )
    elif policy == "youden":
        best = max(scored, key=lambda s: (s[1] + s[2] - 1) if s[1] == s[1] and s[2] == s[2] else -9)
        notes.append("Youden J (sensitivity + specificity - 1) maximised on dev OOF.")
    elif policy == "max_f1":
        def f1(s):
            t, sens, spec, tp, fp = s
            fn = int(((y_prob < t) & (y_true == 1)).sum())
            return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else -1
        best = max(scored, key=f1)
        notes.append("F1 for the infected class maximised on dev OOF.")
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")

    t, sens, spec, _, _ = best
    if spec == spec and spec < 0.10:
        # Recall targets are always reachable by flagging everything; that is a
        # smoke alarm that screams constantly, not a screening tool.
        notes.append(
            f"WARNING: the selected threshold reaches {sens:.0%} recall only by flagging "
            f"almost everything (specificity {spec:.0%}). The model is not separating the "
            "classes usefully at this operating point. Lower target_recall or improve the "
            "model rather than shipping this."
        )
    notes.append(
        "0.50 was not assumed. Applying this threshold to a population with a "
        "different infection prevalence changes precision and NPV, not "
        "sensitivity or specificity."
    )
    return ThresholdDecision(
        float(t), policy, target_recall, float(sens), float(spec), review_band, notes=notes
    )


def needs_review(prob: float, threshold: float, band: float) -> bool:
    """True when a probability sits close enough to the boundary that a human
    should look at the image."""
    return bool(abs(float(prob) - float(threshold)) <= band)
