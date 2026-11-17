"""The inference bundle: one directory that fully determines a prediction.

Weights, preprocessing spec, class mapping, calibrator, threshold, config and
provenance travel together and are hash-verified on load. If any one of them
changed without the others, the model version is a lie -- so they are versioned
as a unit and the server refuses a bundle whose weights do not match the recorded
hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .decision import Calibrator, ThresholdDecision, needs_review
from .preprocess import PreprocessSpec, load_image, prepare_pil

BUNDLE_FORMAT_VERSION = 1
DISCLAIMER = (
    "Screening aid only. This is not an agricultural diagnosis. "
    "Confirm any result with a qualified agronomist or plant-health inspection."
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InferenceBundle:
    root: Path
    meta: Dict[str, Any]
    spec: PreprocessSpec
    calibrator: Calibrator
    threshold: ThresholdDecision
    _model: Any = None

    # ------------------------------------------------------------------ save
    @staticmethod
    def save(
        out_dir: str | Path,
        model,
        cfg,
        spec: PreprocessSpec,
        calibrator: Calibrator,
        threshold: ThresholdDecision,
        model_version: str,
        extra: Optional[dict] = None,
    ) -> Path:
        import torch

        from .models import count_params, model_state_hash
        from .runtime import provenance

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights_path = out / "model.pt"
        torch.save(model.state_dict(), weights_path)

        meta = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "model_version": model_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arch": cfg.train.arch,
            "weights_file": weights_path.name,
            "weights_sha256": _sha256_file(weights_path),
            "state_dict_hash": model_state_hash(model),
            "n_parameters": count_params(model),
            "preprocess": spec.to_dict(),
            "class_map": {"0": "healthy", "1": "infected"},
            "positive_class": "infected",
            "calibration": calibrator.to_dict(),
            "threshold": threshold.to_dict(),
            "disclaimer": DISCLAIMER,
            "config": cfg.to_dict(),
            "config_fingerprint": cfg.fingerprint(),
            "provenance": provenance(),
        }
        meta.update(extra or {})
        (out / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))
        cfg.save(out / "config.yaml")
        return out

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, bundle_dir: str | Path, verify_hash: bool = True) -> "InferenceBundle":
        root = Path(bundle_dir)
        meta_path = root / "bundle.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No bundle.json in {root}. Train a model first "
                "(see README: 'Train and evaluate')."
            )
        meta = json.loads(meta_path.read_text())
        weights = root / meta.get("weights_file", "model.pt")
        if verify_hash and weights.exists():
            actual = _sha256_file(weights)
            if actual != meta.get("weights_sha256"):
                raise RuntimeError(
                    f"Bundle integrity check failed for {weights}.\n"
                    f"  expected {meta.get('weights_sha256')}\n  found    {actual}\n"
                    "The weights changed after the bundle was written; refusing to serve."
                )
        return cls(
            root=root,
            meta=meta,
            spec=PreprocessSpec.from_dict(meta["preprocess"]),
            calibrator=Calibrator.from_dict(meta["calibration"]),
            threshold=ThresholdDecision.from_dict(meta["threshold"]),
        )

    # --------------------------------------------------------------- predict
    def model(self):
        """Lazily construct the network and load weights (CPU by default)."""
        if self._model is None:
            import torch

            from .models import build_model

            net = build_model(self.meta["arch"], pretrained=False)
            state = torch.load(
                self.root / self.meta.get("weights_file", "model.pt"),
                map_location="cpu",
                weights_only=True,
            )
            net.load_state_dict(state)
            net.eval()
            self._model = net
        return self._model

    def raw_probability(self, image_source) -> float:
        """Uncalibrated sigmoid output. Not a confidence. Internal use only."""
        import torch

        from .preprocess import to_tensor

        pil = prepare_pil(load_image(image_source), self.spec)
        tensor = to_tensor(pil, self.spec).unsqueeze(0)
        with torch.no_grad():
            logit = self.model()(tensor).squeeze()
        return float(torch.sigmoid(logit))

    def predict(self, image_source) -> Dict[str, Any]:
        """The single prediction path used by the API and the tests."""
        raw = self.raw_probability(image_source)
        calibrated = float(self.calibrator.apply([raw])[0])
        threshold = float(self.threshold.threshold)
        label = "infected" if calibrated >= threshold else "healthy"
        return {
            "label": label,
            "probability_infected": round(calibrated, 6),
            "threshold": threshold,
            "needs_review": needs_review(calibrated, threshold, self.threshold.review_band),
            "model_version": self.meta.get("model_version", "unknown"),
            "disclaimer": self.meta.get("disclaimer", DISCLAIMER),
        }

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        """Public, non-sensitive description for GET /api/v1/model-info."""
        meta = self.meta
        return {
            "model_version": meta.get("model_version"),
            "arch": meta.get("arch"),
            "created_at": meta.get("created_at"),
            "image_size": self.spec.image_size,
            "preprocess": self.spec.to_dict(),
            "class_map": meta.get("class_map"),
            "positive_class": meta.get("positive_class"),
            "calibration_method": self.calibrator.method,
            "calibration_fitted_on": self.calibrator.fitted_on,
            "threshold": self.threshold.to_dict(),
            "weights_sha256": meta.get("weights_sha256"),
            "config_fingerprint": meta.get("config_fingerprint"),
            "n_parameters": meta.get("n_parameters"),
            "training_summary": meta.get("training_summary"),
            "disclaimer": meta.get("disclaimer", DISCLAIMER),
        }


def promote_to_latest(bundle_dir: str | Path, bundles_root: str | Path) -> Path:
    """Point 'latest' at a specific version without copying huge files twice."""
    root = Path(bundles_root)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "latest.json"
    pointer.write_text(
        json.dumps({"bundle": Path(bundle_dir).name, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2)
    )
    return pointer


def resolve_latest(bundles_root: str | Path) -> Optional[Path]:
    root = Path(bundles_root)
    pointer = root / "latest.json"
    if pointer.exists():
        candidate = root / json.loads(pointer.read_text())["bundle"]
        if (candidate / "bundle.json").exists():
            return candidate
    candidates = sorted(p.parent for p in root.glob("*/bundle.json"))
    return candidates[-1] if candidates else None
