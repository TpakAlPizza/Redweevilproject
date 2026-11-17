"""Builds the canonical table of images: path, label, metadata, hashes, group.

The grouping decision made here determines whether the reported numbers mean
anything. It is therefore explicit, recorded, and impossible to silently skip.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .hashing import hash_image_file, phash_cluster_ids
from .preprocess import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

METADATA_COLUMNS = [
    "image_path",
    "label",
    "tree_id",
    "site_id",
    "session_id",
    "capture_date",
    "camera_id",
]

GROUPING_WARNING = (
    "NO GROUPING METADATA FOUND. Splits cannot guarantee that photos of the "
    "same palm/session appear in only one split. Every metric produced from "
    "this run must be labelled POTENTIALLY OPTIMISTIC INTERNAL VALIDATION."
)


@dataclass
class GroupResolution:
    """How groups were derived, and how much to trust them."""

    strategy: str  # metadata_column | phash_pseudo_groups | image_level
    column: Optional[str]
    n_groups: int
    trust: str  # strong | weak | none
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "column": self.column,
            "n_groups": self.n_groups,
            "trust": self.trust,
            "warnings": self.warnings,
        }


def scan_raw_directory(raw_dir: str | Path, classes: List[str]) -> pd.DataFrame:
    """Walk data/raw/<class>/ and list every candidate image."""
    raw = Path(raw_dir)
    rows = []
    for label in classes:
        class_dir = raw / label
        if not class_dir.is_dir():
            warnings.warn(f"Missing class directory: {class_dir}")
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "image_path": str(path.as_posix()),
                        "filename": path.name,
                        "label": label,
                        "size_bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(rows, columns=["image_path", "filename", "label", "size_bytes"])


def read_image_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Decode every image once: dimensions, mode, EXIF presence, readability."""
    records = []
    for path in df["image_path"]:
        rec = {
            "image_path": path,
            "readable": False,
            "width": None,
            "height": None,
            "mode": None,
            "has_exif": False,
            "error": None,
            "sha256": None,
            "phash": None,
            "dhash": None,
        }
        try:
            img = load_image(path)
            rec.update(
                readable=True,
                width=img.width,
                height=img.height,
                mode=img.mode,
                has_exif=bool(getattr(img, "getexif", lambda: None)()),
            )
            rec.update(hash_image_file(path))
        except Exception as exc:  # corrupt, truncated, or not an image
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
    return pd.DataFrame(records)


def load_metadata(metadata_csv: str | Path) -> Optional[pd.DataFrame]:
    path = Path(metadata_csv)
    if not path.exists():
        return None
    meta = pd.read_csv(path, dtype=str).fillna("")
    if "image_path" not in meta.columns:
        raise ValueError(f"{path} must contain an 'image_path' column")
    meta["image_path"] = meta["image_path"].map(lambda p: Path(p).as_posix())
    return meta


def write_metadata_template(out_path: str | Path, df: pd.DataFrame) -> Path:
    """Emit a pre-filled CSV the user only has to complete the group columns in."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame({"image_path": df["image_path"], "label": df["label"]})
    for col in METADATA_COLUMNS:
        if col not in template.columns:
            template[col] = ""
    template[METADATA_COLUMNS].to_csv(out, index=False)
    return out


def resolve_groups(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.Series, GroupResolution]:
    """Pick the strongest available grouping and say so out loud.

    Priority:
      1. A metadata column (tree_id > session_id > site_id) that is fully populated.
      2. Perceptual-hash near-duplicate clusters as pseudo-groups.
      3. Image-level split -- only if explicitly allowed, always flagged.
    """
    msgs: List[str] = []

    for col in cfg.data.group_key_priority:
        if col in df.columns:
            values = df[col].astype(str).str.strip()
            populated = (values != "") & (values.str.lower() != "nan")
            if populated.all():
                n = values.nunique()
                if n < 2:
                    msgs.append(f"Column '{col}' has a single value; skipping it.")
                    continue
                return values, GroupResolution(
                    "metadata_column", col, n, "strong", msgs
                )
            if populated.any():
                msgs.append(
                    f"Column '{col}' is only {populated.mean():.0%} populated "
                    "and was skipped. Partial grouping is worse than none "
                    "because it hides which rows are unprotected."
                )

    if cfg.data.phash_pseudo_groups:
        records = df[["image_path", "phash"]].dropna().to_dict("records")
        if records:
            lookup = phash_cluster_ids(records, cfg.data.phash_group_distance)
            groups = df["image_path"].map(lookup).fillna("ungrouped")
            msgs.append(GROUPING_WARNING)
            msgs.append(
                "Falling back to perceptual-hash pseudo-groups: this stops "
                "obvious near-duplicates from straddling the split, but it does "
                "NOT stop two different photos of the same tree from doing so."
            )
            return groups, GroupResolution(
                "phash_pseudo_groups", None, int(groups.nunique()), "weak", msgs
            )

    if not cfg.data.allow_image_level_split:
        raise RuntimeError(
            GROUPING_WARNING + " allow_image_level_split is false, so this run is aborted."
        )
    msgs.append(GROUPING_WARNING)
    msgs.append("Using an image-level split. Treat all results as an upper bound.")
    return (
        pd.Series([f"image_{i:05d}" for i in range(len(df))], index=df.index),
        GroupResolution("image_level", None, len(df), "none", msgs),
    )


def build_catalog(cfg: Config, compute_hashes: bool = True) -> Tuple[pd.DataFrame, GroupResolution]:
    """Full catalog: files + properties + metadata + resolved group ids."""
    df = scan_raw_directory(cfg.data.raw_dir, cfg.data.classes)
    if df.empty:
        raise FileNotFoundError(
            f"No images found under {cfg.data.raw_dir}/. Expected "
            f"{cfg.data.raw_dir}/infected/ and {cfg.data.raw_dir}/healthy/."
        )
    if compute_hashes:
        df = df.merge(read_image_properties(df), on="image_path", how="left")

    meta = load_metadata(cfg.data.metadata_csv)
    if meta is not None:
        extra = [c for c in meta.columns if c not in {"image_path", "label"}]
        df = df.merge(meta[["image_path"] + extra], on="image_path", how="left")
        missing = df[extra].isna().all(axis=1).sum() if extra else 0
        if missing:
            warnings.warn(
                f"{missing} images have no row in {cfg.data.metadata_csv}. "
                "Check that image_path values match exactly."
            )

    df["label_binary"] = (df["label"] == cfg.data.positive_class).astype(int)
    groups, resolution = resolve_groups(df, cfg)
    df["group_id"] = groups.values
    return df, resolution
