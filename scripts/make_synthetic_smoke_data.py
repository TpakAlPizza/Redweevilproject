#!/usr/bin/env python3
"""Generate a synthetic dataset so the pipeline can be exercised end to end
before real photographs exist.

These are coloured noise images, not palms. Any metric produced from them is
meaningless by construction -- the point is to prove the plumbing works
(grouping, splitting, leakage checks, reporting), not to estimate performance.

    python scripts/make_synthetic_smoke_data.py --per-class 130 --photos-per-tree 3
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(rng: np.random.Generator, base_hue: tuple, size=(640, 480)) -> Image.Image:
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        canvas[..., c] = base_hue[c]
    canvas += rng.normal(0, 18, size=(h, w, 3))
    gradient = np.linspace(-25, 25, h)[:, None, None]
    canvas += gradient
    for _ in range(rng.integers(4, 9)):  # crude "fronds"
        y, x = rng.integers(0, h), rng.integers(0, w)
        length, thickness = rng.integers(60, 200), rng.integers(3, 9)
        canvas[max(0, y - thickness):y + thickness, max(0, x - length):x + length] += rng.normal(20, 10)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--per-class", type=int, default=130)
    parser.add_argument("--photos-per-tree", type=int, default=3)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    rows = []
    out_root = Path(args.out)

    for label, hue in (("healthy", (60, 120, 70)), ("infected", (120, 110, 60))):
        class_dir = out_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        n_trees = max(1, args.per_class // args.photos_per_tree)
        made = 0
        for tree in range(n_trees):
            tree_id = f"{label[:3]}_tree_{tree:03d}"
            site_id = f"site_{tree % args.sites}"
            session_id = f"{site_id}_session_{tree % 7}"
            for shot in range(args.photos_per_tree):
                if made >= args.per_class:
                    break
                # Photos of one tree are deliberately similar: this is exactly the
                # correlation that makes a naive random split dishonest.
                jitter = tuple(c + rng.normal(0, 4) for c in hue)
                img = make_image(rng, jitter)
                path = class_dir / f"{tree_id}_{shot}.jpg"
                img.save(path, quality=88)
                rows.append(
                    {
                        "image_path": path.as_posix(),
                        "label": label,
                        "tree_id": tree_id,
                        "site_id": site_id,
                        "session_id": session_id,
                        "capture_date": f"2025-0{1 + (tree % 6)}-1{shot}",
                        "camera_id": f"cam_{tree % 3}",
                    }
                )
                made += 1

    meta_path = Path(args.metadata)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic images to {out_root} and metadata to {meta_path}")
    print("These are NOT palms. Do not report any metric computed from them.")


if __name__ == "__main__":
    main()
