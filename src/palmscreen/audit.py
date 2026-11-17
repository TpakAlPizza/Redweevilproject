"""Dataset audit.

On ~260 images the dataset is the experiment. This module looks for the things
that silently decide the outcome: unreadable files, duplicates, near-duplicates
that will straddle a split, and *shortcuts* -- class-correlated artefacts such as
one camera per class, one background per class, or one resolution per class. A
model that learns "photos from camera B are infected" scores well and is useless.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .catalog import build_catalog, write_metadata_template
from .config import Config
from .hashing import near_duplicate_clusters
from .preprocess import load_image

SHORTCUT_CHECKLIST = [
    "Is the diseased tissue actually visible in the frame, or is the palm a distant speck?",
    "Do infected and healthy photos come from different farms, seasons, or times of day?",
    "Are there watermarks, logos, date stamps, or borders on one class only?",
    "Do multiple palms appear in one frame, so the label is ambiguous?",
    "Were infected images downloaded from the web while healthy ones were shot on a phone?",
    "Do backgrounds differ systematically (sand vs irrigation, sky vs buildings)?",
    "Are labels assigned per-tree by an agronomist, or guessed from appearance?",
]


def _describe_dimensions(df: pd.DataFrame) -> dict:
    ok = df[df["readable"] == True]  # noqa: E712
    if ok.empty:
        return {}
    return {
        "width": {"min": int(ok.width.min()), "median": float(ok.width.median()), "max": int(ok.width.max())},
        "height": {"min": int(ok.height.min()), "median": float(ok.height.median()), "max": int(ok.height.max())},
        "megapixels_median": float((ok.width * ok.height).median() / 1e6),
        "modes": Counter(ok["mode"].tolist()),
        "aspect_ratio": {
            "min": float((ok.width / ok.height).min()),
            "max": float((ok.width / ok.height).max()),
        },
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def _mean_colour_stats(df: pd.DataFrame, positive_class: str, sample_size: int = 40) -> dict:
    """Average colour per class. A large gap suggests lighting/background shortcut."""
    rows = []
    for label, sub in df[df["readable"] == True].groupby("label"):  # noqa: E712
        sample = sub.sample(min(len(sub), sample_size), random_state=0)
        for path in sample["image_path"]:
            try:
                arr = np.asarray(load_image(path).convert("RGB").resize((64, 64)), dtype=np.float32)
                rows.append(
                    {"label": label, "r": arr[..., 0].mean(), "g": arr[..., 1].mean(),
                     "b": arr[..., 2].mean(), "brightness": arr.mean()}
                )
            except Exception:
                continue
    if not rows:
        return {}
    table = pd.DataFrame(rows)
    stats: Dict[str, object] = {"per_class_mean": table.groupby("label").mean(numeric_only=True).round(2).to_dict()}
    labels = sorted(table["label"].unique())
    if len(labels) == 2:
        a = table[table.label == labels[0]]
        b = table[table.label == labels[1]]
        stats["effect_size_cohens_d"] = {
            ch: round(_cohens_d(a[ch].values, b[ch].values), 3)
            for ch in ("r", "g", "b", "brightness")
        }
        stats["sampled_per_class"] = int(min(len(a), len(b)))
    return stats


def _class_correlated_metadata(df: pd.DataFrame) -> List[dict]:
    """Any metadata field that predicts the label on its own is a shortcut risk."""
    findings = []
    candidates = [c for c in ("camera_id", "site_id", "capture_date", "session_id") if c in df.columns]
    ok = df[df["readable"] == True]  # noqa: E712
    for col in candidates:
        values = ok[col].astype(str)
        if values.replace("", np.nan).isna().all() or values.nunique() < 2:
            continue
        cross = pd.crosstab(values, ok["label"])
        purity = (cross.max(axis=1) / cross.sum(axis=1))
        pure = purity[(purity >= 0.95) & (cross.sum(axis=1) >= 3)]
        if len(pure):
            findings.append(
                {
                    "field": col,
                    "single_class_values": pure.index.tolist()[:20],
                    "n_single_class_values": int(len(pure)),
                    "message": (
                        f"{len(pure)} value(s) of '{col}' map to (almost) one class only. "
                        "The model can learn the acquisition condition instead of the disease."
                    ),
                }
            )
    # Resolution as a proxy for source
    dims = ok.assign(dim=ok.width.astype(str) + "x" + ok.height.astype(str))
    cross = pd.crosstab(dims["dim"], dims["label"])
    if len(cross):
        purity = cross.max(axis=1) / cross.sum(axis=1)
        pure = purity[(purity >= 0.95) & (cross.sum(axis=1) >= 5)]
        if len(pure):
            findings.append(
                {
                    "field": "image_resolution",
                    "single_class_values": pure.index.tolist()[:20],
                    "n_single_class_values": int(len(pure)),
                    "message": (
                        "Some exact resolutions occur in one class only, which usually "
                        "means the two classes came from different devices or sources."
                    ),
                }
            )
    return findings


def _filename_pattern_check(df: pd.DataFrame) -> List[dict]:
    findings = []
    for label, sub in df.groupby("label"):
        prefixes = Counter(str(n)[:4].lower() for n in sub["filename"])
        top, count = prefixes.most_common(1)[0] if prefixes else ("", 0)
        if count / max(len(sub), 1) > 0.8:
            findings.append(
                {
                    "field": "filename_prefix",
                    "label": label,
                    "pattern": top,
                    "message": (
                        f"{count}/{len(sub)} '{label}' filenames start with '{top}'. "
                        "Harmless for the model (it never sees filenames) but a strong "
                        "hint that the two classes were collected separately."
                    ),
                }
            )
    return findings


def build_montage(
    df: pd.DataFrame, out_path: str | Path, cols: int = 8, thumb: int = 160, max_images: int = 64,
    title_field: str = "filename",
) -> Optional[Path]:
    """Contact sheet for eyeball label inspection."""
    from PIL import ImageDraw

    rows_df = df[df["readable"] == True].head(max_images)  # noqa: E712
    if rows_df.empty:
        return None
    n = len(rows_df)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    label_h = 14
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        try:
            img = load_image(row["image_path"]).convert("RGB")
            img.thumbnail((thumb, thumb))
        except Exception:
            continue
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        sheet.paste(img, (x + (thumb - img.width) // 2, y + (thumb - img.height) // 2))
        caption = str(row.get(title_field, ""))[:22]
        draw.text((x + 2, y + thumb + 1), caption, fill="black")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def run_audit(cfg: Config, out_dir: Optional[str] = None) -> dict:
    """Full audit. Writes JSON + CSV + montages and returns the report dict."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or Path(cfg.report.output_dir) / "audit" / stamp)
    out.mkdir(parents=True, exist_ok=True)

    df, resolution = build_catalog(cfg, compute_hashes=True)
    readable = df[df["readable"] == True]  # noqa: E712
    unreadable = df[df["readable"] != True]  # noqa: E712

    records = readable[["image_path", "sha256", "phash", "label"]].to_dict("records")
    exact_map: Dict[str, List[str]] = {}
    for r in records:
        exact_map.setdefault(r["sha256"], []).append(r["image_path"])
    exact_dups = [sorted(v) for v in exact_map.values() if len(v) > 1]

    near_clusters = near_duplicate_clusters(records, cfg.data.phash_group_distance)
    label_by_path = dict(zip(readable["image_path"], readable["label"]))
    cross_class_near = [
        c for c in near_clusters if len({label_by_path.get(p) for p in c}) > 1
    ]
    conflicting_labels = [
        {"sha256": h, "paths": v, "labels": sorted({label_by_path.get(p) for p in v})}
        for h, v in exact_map.items()
        if len(v) > 1 and len({label_by_path.get(p) for p in v}) > 1
    ]

    counts = readable["label"].value_counts().to_dict()
    minority = min(counts.values()) if counts else 0
    majority = max(counts.values()) if counts else 0

    warnings_list: List[str] = list(resolution.warnings)
    if len(unreadable):
        warnings_list.append(f"{len(unreadable)} unreadable/corrupt file(s); they are excluded from all splits.")
    if exact_dups:
        warnings_list.append(
            f"{len(exact_dups)} exact-duplicate group(s). Duplicates inflate metrics and must not straddle a split."
        )
    if near_clusters:
        warnings_list.append(
            f"{len(near_clusters)} near-duplicate cluster(s) at Hamming <= {cfg.data.phash_group_distance}. "
            "These are treated as one group when metadata is missing."
        )
    if cross_class_near:
        warnings_list.append(
            f"{len(cross_class_near)} near-duplicate cluster(s) span BOTH classes -- likely mislabelled images. Inspect first."
        )
    if conflicting_labels:
        warnings_list.append(
            f"{len(conflicting_labels)} identical file(s) carry conflicting labels. Fix before training."
        )
    if majority and minority / majority < 0.8:
        warnings_list.append(f"Class imbalance {counts}. Use balanced metrics, not accuracy.")

    colour = _mean_colour_stats(readable, cfg.data.positive_class)
    effects = colour.get("effect_size_cohens_d") or {}
    extreme = {ch: d for ch, d in effects.items() if abs(d) >= 1.5}
    if extreme:
        warnings_list.append(
            "Large global colour/brightness difference between classes "
            f"(Cohen's d {extreme}). A model can separate these classes on average "
            "colour alone, without ever looking at a frond. Check whether the two "
            "classes were photographed in different light, at different times of "
            "day, or from different sources."
        )
    if len(readable) < 400:
        warnings_list.append(
            f"Only {len(readable)} usable images. Confidence intervals will be wide; "
            "no result from this dataset can establish field performance."
        )

    montages = {}
    for label, sub in readable.groupby("label"):
        p = build_montage(sub, out / f"montage_{label}.png")
        if p:
            montages[str(label)] = p.name
    if cross_class_near:
        flat = [p for c in cross_class_near for p in c][:64]
        p = build_montage(readable[readable.image_path.isin(flat)].assign(
            filename=lambda d: d.label.str[:4] + "|" + d.filename), out / "montage_cross_class_near_dupes.png")
        if p:
            montages["cross_class_near_duplicates"] = p.name

    report = {
        "generated_at": stamp,
        "output_dir": str(out),
        "raw_dir": cfg.data.raw_dir,
        "n_files_found": int(len(df)),
        "n_readable": int(len(readable)),
        "unreadable": unreadable[["image_path", "error"]].to_dict("records"),
        "class_counts": counts,
        "grouping": resolution.as_dict(),
        "dimensions": _describe_dimensions(df),
        "duplicates": {
            "exact_groups": exact_dups,
            "n_exact_groups": len(exact_dups),
            "near_duplicate_clusters": near_clusters[:200],
            "n_near_duplicate_clusters": len(near_clusters),
            "cross_class_near_duplicates": cross_class_near,
            "conflicting_labels": conflicting_labels,
        },
        "shortcut_signals": {
            "class_correlated_metadata": _class_correlated_metadata(df),
            "filename_patterns": _filename_pattern_check(readable),
            "colour_statistics": _mean_colour_stats(readable, cfg.data.positive_class),
        },
        "manual_review_checklist": SHORTCUT_CHECKLIST,
        "montages": montages,
        "warnings": warnings_list,
        "roi_guidance": (
            "If the diseased fronds occupy a small part of the frame, whole-image "
            "classification is the wrong tool: at 224x224 the symptom may cover a "
            "handful of pixels. In that case annotate crops/ROIs around symptomatic "
            "tissue, or run a palm detector first and classify each detected crown."
        ),
    }

    (out / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    df.to_csv(out / "catalog.csv", index=False)
    pd.DataFrame(
        [{"issue": w} for w in warnings_list]
    ).to_csv(out / "warnings.csv", index=False)

    if resolution.strategy != "metadata_column":
        template = write_metadata_template(cfg.data.metadata_csv + ".template", df)
        report["metadata_template"] = str(template)
        print(f"\nGrouping metadata template written to: {template}")
        print("Fill in tree_id/site_id/session_id and save it as", cfg.data.metadata_csv)

    print(f"Audit written to {out}")
    for w in warnings_list:
        print("  ! " + w)
    return report
