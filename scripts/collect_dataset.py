#!/usr/bin/env python3
"""Find the images wherever they are and merge them into data/raw/.

Roboflow classification exports usually look like this:

    export/train/infected/  export/train/healthy/
    export/valid/infected/  export/valid/healthy/
    export/test/infected/   export/test/healthy/

That train/valid/test split is discarded on purpose. It splits at the image
level, so the augmented copies of one source frame get scattered across train
AND valid -- which is the leak that produces a perfect validation score. This
script flattens everything back into two class folders so the grouped splitter
can do the job properly.

    python3 scripts/collect_dataset.py --source ~/Downloads/palm-export
    python3 scripts/collect_dataset.py --source ~/Downloads --dry-run

Exact duplicates (same SHA-256) are copied once. Files whose class cannot be
determined from the path are listed rather than guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Path fragments that identify a class. Lowercased substring match.
CLASS_HINTS: Dict[str, List[str]] = {
    "infected": ["infected", "infect", "diseased", "disease", "sick", "unhealthy",
                 "positive", "weevil", "مصاب", "مصابة"],
    "healthy": ["healthy", "health", "normal", "negative", "good", "سليم", "سليمة"],
}

# Folder names that are Roboflow's split, not a class.
SPLIT_FOLDERS = {"train", "training", "valid", "validation", "val", "test", "testing"}


def classify_path(path: Path) -> Optional[str]:
    """Decide the class from the folder names, nearest folder first.

    'unhealthy' contains 'healthy', so the deepest folder wins and 'infected'
    hints are checked before 'healthy' ones.
    """
    for part in reversed(path.parts[:-1]):
        lowered = part.lower()
        if lowered in SPLIT_FOLDERS:
            continue
        for hint in CLASS_HINTS["infected"]:
            if hint in lowered:
                return "infected"
        for hint in CLASS_HINTS["healthy"]:
            if hint in lowered:
                return "healthy"
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _warn_about_split_leak(found: Dict[str, List[Path]]) -> None:
    """Report source frames whose augmented copies were spread across the
    original train/valid/test folders. This is the specific leak that makes a
    validation score look perfect."""
    import re

    roboflow_suffix = re.compile(r"[._](jpe?g|png|webp)\.rf\.[0-9a-f]+\.\w+$", re.IGNORECASE)
    variant_index = re.compile(r"[._-]\d{1,2}$")

    by_stem: Dict[str, set] = defaultdict(set)
    for paths in found.values():
        for path in paths:
            stem = variant_index.sub("", roboflow_suffix.sub("", path.name))
            split = next((p.lower() for p in path.parts if p.lower() in SPLIT_FOLDERS), None)
            if split:
                by_stem[stem].add(split)

    straddling = {stem: splits for stem, splits in by_stem.items() if len(splits) > 1}
    if not straddling:
        return

    print(f"\n!! {len(straddling)} source frame(s) had augmented copies spread across the")
    print("   original train/valid/test folders. Examples:")
    for stem, splits in list(straddling.items())[:5]:
        print(f"       {stem}  ->  {sorted(splits)}")
    print("   That is the leak: the same photograph was in training AND validation,")
    print("   so a perfect validation score measured memorisation, not detection.")
    print("   Merging here fixes it, provided you now split with grouping metadata.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="folder to search (searched recursively)")
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument("--dry-run", action="store_true", help="report what would happen, copy nothing")
    parser.add_argument("--move", action="store_true", help="move instead of copy")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    dest = Path(args.dest)
    found: Dict[str, List[Path]] = defaultdict(list)
    unclassified: List[Path] = []
    origins: Dict[str, Counter] = defaultdict(Counter)

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if dest.resolve() in path.resolve().parents:
            continue  # already collected
        label = classify_path(path)
        if label is None:
            unclassified.append(path)
            continue
        found[label].append(path)
        # Record which Roboflow split it came from, purely for the summary.
        split = next((p.lower() for p in path.parts if p.lower() in SPLIT_FOLDERS), "(no split folder)")
        origins[label][split] += 1

    if not found:
        print(f"No class folders found under {source}.")
        print("Expected folder names containing 'infected'/'healthy'. Found instead:")
        for d in sorted({p.parent for p in source.rglob("*") if p.is_file()})[:15]:
            print("   ", d)
        raise SystemExit(1)

    print(f"Searched: {source}\n")
    for label, paths in found.items():
        print(f"{label:<12} {len(paths):>5} images   from {dict(origins[label])}")
    if unclassified:
        print(f"\n{len(unclassified)} image(s) could not be assigned to a class. Examples:")
        for path in unclassified[:5]:
            print("   ", path)
        print("   Move these into the right folder by hand, or rerun with the folder renamed.")

    if args.dry_run:
        print("\nDry run: nothing copied. Remove --dry-run to proceed.")
        return

    print()
    seen: Dict[str, Path] = {}
    duplicates = 0
    collisions = 0
    for label, paths in found.items():
        target_dir = dest / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            digest = sha256(path)
            if digest in seen:
                duplicates += 1
                continue
            seen[digest] = path
            target = target_dir / path.name
            if target.exists():
                # Same filename, different content: keep both, disambiguate.
                collisions += 1
                target = target_dir / f"{path.stem}__{digest[:8]}{path.suffix}"
            if args.move:
                shutil.move(str(path), target)
            else:
                shutil.copy2(path, target)

    def count_images(directory: Path) -> int:
        return sum(1 for p in directory.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    print(f"Collected into {dest}/")
    for label in found:
        print(f"    {label:<12} {count_images(dest / label):>5} images")
    if duplicates:
        print(f"\n{duplicates} exact duplicate(s) skipped (identical SHA-256).")
    if collisions:
        print(f"{collisions} filename collision(s) renamed to keep both files.")

    _warn_about_split_leak(found)

    counts = {label: count_images(dest / label) for label in found}
    if len(counts) < 2:
        print("\n!! Only one class was collected. Binary classification needs both.")
    else:
        smallest, largest = min(counts.values()), max(counts.values())
        if largest and smallest / largest < 0.5:
            print(f"\n!! Class imbalance: {counts}. Use balanced metrics, never accuracy.")

    print("\nNext:")
    print("    python3 scripts/build_metadata_from_filenames.py")
    print("    python3 -m palmscreen.cli audit  --config configs/effnet_b0.yaml")
    print("    python3 -m palmscreen.cli split  --config configs/effnet_b0.yaml")
    print("    python3 -m palmscreen.cli probe  --config configs/effnet_b0.yaml")


if __name__ == "__main__":
    main()
