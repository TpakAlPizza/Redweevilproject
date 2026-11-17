"""Configuration objects, loaded from YAML. No hidden defaults in the code:
every knob that changes a result lives here and is snapshotted into the report
and the inference bundle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _build(cls, data: Optional[dict]):
    data = data or {}
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    metadata_csv: str = "data/metadata.csv"
    classes: List[str] = field(default_factory=lambda: ["healthy", "infected"])
    positive_class: str = "infected"
    # Group key priority. The first column present and fully populated wins.
    group_key_priority: List[str] = field(
        default_factory=lambda: ["tree_id", "session_id", "site_id"]
    )
    # If no grouping metadata exists, cluster near-duplicates by perceptual hash
    # and use the clusters as pseudo-groups. Weaker than real metadata, but far
    # better than a naive per-image split.
    phash_pseudo_groups: bool = True
    phash_group_distance: int = 6
    allow_image_level_split: bool = True  # emits a loud warning if used
    # Optionally hold out an entire site/session as the test set (strongest split).
    holdout_column: Optional[str] = None
    holdout_values: List[str] = field(default_factory=list)


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    n_folds: int = 5
    seed: int = 1337
    manifest_dir: str = "artifacts/splits"


@dataclass
class AugmentationConfig:
    """Conservative, biologically motivated only. Aggressive transforms are off
    by default and each carries a comment explaining why."""

    random_resized_crop: bool = True
    crop_scale_min: float = 0.80  # mild; not 0.08 like ImageNet defaults
    crop_scale_max: float = 1.0
    rotation_degrees: float = 12.0
    horizontal_flip: bool = True  # a mirrored palm is still a valid palm
    vertical_flip: bool = False  # palms are not upside down in the field
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    hue: float = 0.02  # tiny: leaf yellowing IS the signal, do not recolour it
    mixup: bool = False  # off: mixes labels, hurts calibration on ~260 images
    cutmix: bool = False


@dataclass
class TrainConfig:
    arch: str = "efficientnet_b0"  # or mobilenet_v3_large
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    head_epochs: int = 6  # phase 1: frozen backbone
    finetune_epochs: int = 25  # phase 2: partial unfreeze
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    unfreeze_last_n_blocks: int = 3
    freeze_batchnorm: bool = True  # BN stats are unreliable at batch size 8-16
    early_stopping_metric: str = "val_pr_auc"  # or val_loss, val_roc_auc
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    grad_clip_norm: float = 1.0
    use_class_weights: bool = False  # default off; only enable with evidence
    amp: bool = False  # MPS autocast is inconsistent; off by default
    seed: int = 1337
    checkpoint_dir: str = "artifacts/checkpoints"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ThresholdConfig:
    """Decision policy. Default prioritises catching infected palms."""

    policy: str = "max_specificity_at_recall"  # or youden, max_f1, fixed
    target_recall: float = 0.90
    fixed_threshold: float = 0.50
    # Predictions within this band around the threshold are flagged for review.
    review_band: float = 0.10


@dataclass
class CalibrationConfig:
    method: str = "platt"  # platt | temperature | isotonic | none
    # isotonic is available but discouraged below ~500 dev images.


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    bootstrap_iterations: int = 2000
    ece_bins: int = 10
    prevalence_grid: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.50]
    )


@dataclass
class Config:
    project_name: str = "palm-screen"
    model_version: str = "0.1.0"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        train_raw = dict(raw.get("train") or {})
        aug = _build(AugmentationConfig, train_raw.pop("augmentation", None))
        train = _build(TrainConfig, train_raw)
        train.augmentation = aug
        return cls(
            project_name=raw.get("project_name", "palm-screen"),
            model_version=raw.get("model_version", "0.1.0"),
            data=_build(DataConfig, raw.get("data")),
            split=_build(SplitConfig, raw.get("split")),
            train=train,
            threshold=_build(ThresholdConfig, raw.get("threshold")),
            calibration=_build(CalibrationConfig, raw.get("calibration")),
            report=_build(ReportConfig, raw.get("report")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the whole config, recorded in every artifact."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
