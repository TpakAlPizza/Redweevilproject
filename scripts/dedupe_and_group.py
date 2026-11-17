#!/usr/bin/env python3
"""Remove exact duplicates and put visually identical images in the same group.

Filenames are not a reliable guide to what is the same photograph:

    photo_1448-03-27 11.20.11.jpeg
    photo_1448-03-27 11.20.11 (1).jpeg     <- byte-identical copy
    photo_1448-03-27 11.19.22.jpeg
    photo_1448-03-27 11.17.59.jpeg         <- identical pixels, different clock

Two fixes, in order:

  1. Exact duplicates (same SHA-256) are moved out of the dataset. A file that
     appears twice is not two observations; keeping both weights that palm
     double during training and inflates the apparent dataset size.

  2. Near-duplicates that survive are merged into one group, so the splitter
     keeps them on the same side. Filename grouping alone misses these, which
     is what the leakage check caught.

Duplicates are MOVED, never deleted, so nothing is lost if the call was wrong.

    python3 scripts/dedupe_and_group.py
    python3 scripts/dedupe_and_group.py --dry-run
    python3 scripts/dedupe_and_group.py --distance 8      # stricter merging
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from palmscreen.hashing import hamming, phash, sha256_file  # noqa: E402
from palmscreen.preprocess import load_image  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

ROBOFLOW_SUFFIX = re.compile(r"[._](jpe?g|png|webp)\.rf\.[0-9a-f]{16,}\.\w+$", re.IGNORECASE)
VARIANT_INDEX = re.compile(r"[._-]\d{1,2}$")
COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")  # "photo (1)", "photo (2)"
TIMESTAMP = re.compile(
    r"(?P<y>\d{4})[-_]?(?P<mo>\d{2})[-_]?(?P<d>\d{2})[-_ ]?"
    r"(?P<h>\d{2})[-_.:]?(?P<mi>\d{2})[-_.:]?(?P<s>\d{2})"
)


class Union:
    """Minimal union-find, so every merge rule can be applied independently."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


def source_stem(filename: str) -> str:
    """Reduce a filename to the source frame it came from.

    The variant index (_1 .. _4) is stripped ONLY for Roboflow exports. On a
    plain photo the trailing digits are usually the seconds of a timestamp
    ("photo_1448-03-27 11.20.11"), and removing them silently merges every
    photograph taken in the same minute.
    """
    is_roboflow = ".rf." in filename.lower()
    stem = ROBOFLOW_SUFFIX.sub("", filename)
    stem = re.sub(r"\.(jpe?g|png|webp|bmp|tiff?)$", "", stem, flags=re.IGNORECASE)
    stem = COPY_SUFFIX.sub("", stem)          # "(1)" copies are the same file
    if is_roboflow:
        stem = VARIANT_INDEX.sub("", stem)    # augmented copies of one frame
    return stem


def parse_timestamp(filename: str) -> Optional[datetime]:
    match = TIMESTAMP.search(filename)
    if not match:
        return None
    try:
        p = {k: int(v) for k, v in match.groupdict().items()}
        if not (1900 <= p["y"] <= 2100):  # Hijri-style years appear in some exports
            return None
        return datetime(p["y"], p["mo"], p["d"], p["h"], p["mi"], p["s"])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/metadata.csv")
    parser.add_argument("--quarantine", default="data/removed_duplicates")
    parser.add_argument("--classes", nargs="+", default=["healthy", "infected"])
    parser.add_argument("--distance", type=int, default=6,
                        help="perceptual-hash Hamming distance treated as the same photo")
    parser.add_argument("--session-gap-minutes", type=int, default=5)
    parser.add_argument(
        "--group-mode", choices=["frame", "burst"], default="frame",
        help=(
            "frame: one group per source photograph (plus its augmented copies). "
            "burst: also merge photos taken within --burst-seconds of each other, "
            "which are almost certainly the same palm from a slightly different angle."
        ),
    )
    parser.add_argument("--burst-seconds", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    records = []
    for label in args.classes:
        class_dir = raw / label
        if not class_dir.is_dir():
            print(f"!! missing directory: {class_dir}")
            continue
        for path in sorted(p for p in class_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                records.append({
                    "path": path.as_posix(),
                    "filename": path.name,
                    "label": label,
                    "sha256": sha256_file(path),
                    "phash": phash(load_image(path)),
                    "stem": source_stem(path.name),
                    "timestamp": parse_timestamp(path.name),
                })
            except Exception as exc:
                print(f"!! unreadable, skipped: {path.name} ({type(exc).__name__})")

    if not records:
        raise SystemExit(f"No readable images under {raw}/")
    print(f"Read {len(records)} images.\n")

    # ---- 1. exact duplicates -------------------------------------------
    by_hash: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)

    removed = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        keep, extras = group[0], group[1:]
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            print(f"!! CONFLICT: identical file appears in BOTH classes {sorted(labels)}:")
            for r in group:
                print(f"       {r['path']}")
            print("   One of these labels is wrong. Fix by hand before training.\n")
            continue
        for extra in extras:
            removed.append((extra, keep))

    if removed:
        print(f"{len(removed)} exact duplicate file(s) found:")
        for extra, keep in removed[:8]:
            print(f"    {Path(extra['path']).name}")
            print(f"      duplicate of {Path(keep['path']).name}")
        if len(removed) > 8:
            print(f"    ... and {len(removed) - 8} more")

        if not args.dry_run:
            quarantine = Path(args.quarantine)
            for extra, _ in removed:
                target_dir = quarantine / extra["label"]
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(extra["path"], target_dir / Path(extra["path"]).name)
            print(f"\nMoved to {quarantine}/ (not deleted).")
        removed_paths = {extra["path"] for extra, _ in removed}
        records = [r for r in records if r["path"] not in removed_paths]
    else:
        print("No exact duplicates.")

    # ---- 2. merge groups ------------------------------------------------
    paths = [r["path"] for r in records]
    union = Union(paths)
    by_path = {r["path"]: r for r in records}

    # a) same source frame per filename (Roboflow variants, "(1)" copies)
    by_stem: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        by_stem[f"{record['label']}::{record['stem']}"].append(record["path"])
    for group in by_stem.values():
        for other in group[1:]:
            union.join(group[0], other)

    # b) visually identical images the filenames disagreed about
    merged_by_phash = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["label"] != records[j]["label"]:
                continue  # cross-class similarity is a labelling question, not a group
            if hamming(records[i]["phash"], records[j]["phash"]) <= args.distance:
                if union.find(records[i]["path"]) != union.find(records[j]["path"]):
                    merged_by_phash += 1
                union.join(records[i]["path"], records[j]["path"])
    if merged_by_phash:
        print(f"\n{merged_by_phash} additional merge(s) from visual similarity "
              f"(Hamming <= {args.distance}) that filenames alone missed.")

    # c) optional: photographs taken seconds apart are the same palm
    if args.group_mode == "burst":
        merged_by_burst = 0
        for label in {r["label"] for r in records}:
            timed = sorted(
                (r for r in records if r["label"] == label and r["timestamp"]),
                key=lambda r: r["timestamp"],
            )
            for previous, current in zip(timed, timed[1:]):
                gap = (current["timestamp"] - previous["timestamp"]).total_seconds()
                if gap <= args.burst_seconds:
                    if union.find(previous["path"]) != union.find(current["path"]):
                        merged_by_burst += 1
                    union.join(previous["path"], current["path"])
        if merged_by_burst:
            print(f"\n{merged_by_burst} merge(s) from burst timing "
                  f"(<= {args.burst_seconds}s apart, same class).")

    groups = union.groups()
    group_id = {}
    for index, (_, members) in enumerate(sorted(groups.items())):
        label = by_path[members[0]]["label"]
        for path in members:
            group_id[path] = f"{label[:3]}__grp_{index:04d}"

    # ---- 3. sessions ----------------------------------------------------
    timed = sorted((r for r in records if r["timestamp"]), key=lambda r: r["timestamp"])
    sessions: Dict[str, str] = {}
    index, previous = 0, None
    for record in timed:
        if previous and record["timestamp"] - previous > timedelta(minutes=args.session_gap_minutes):
            index += 1
        sessions[record["path"]] = f"session_{index:03d}"
        previous = record["timestamp"]
    for i, record in enumerate(r for r in records if not r["timestamp"]):
        sessions[record["path"]] = f"session_untimed_{i:04d}"

    # ---- 4. write -------------------------------------------------------
    rows = [{
        "image_path": r["path"],
        "label": r["label"],
        "tree_id": group_id[r["path"]],
        "site_id": "roboflow_export" if ".rf." in r["filename"].lower() else "direct_photo",
        "session_id": sessions[r["path"]],
        "capture_date": r["timestamp"].strftime("%Y-%m-%d") if r["timestamp"] else "",
        "camera_id": "roboflow_export" if ".rf." in r["filename"].lower() else "direct_photo",
    } for r in records]

    if not args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")
    else:
        print(f"\nDry run: would write {len(rows)} rows.")

    _summary(rows)


def _summary(rows: List[dict]) -> None:
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    print(f"\n{'class':<12}{'images':>8}{'groups':>9}   images/group")
    print("-" * 46)
    for label, subset in by_label.items():
        groups = len({r["tree_id"] for r in subset})
        print(f"{label:<12}{len(subset):>8}{groups:>9}   {len(subset) / max(groups, 1):.2f}")

    families: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        families[row["site_id"]][row["label"]] += 1
    if len(families) > 1:
        print("\nsource x class")
        print("-" * 46)
        flagged = False
        for family, counts in families.items():
            total = sum(counts.values())
            flag = ""
            if max(counts.values()) / total >= 0.95 and total >= 3:
                flag = "  <-- one class only"
                flagged = True
            print(f"{family:<20}{dict(counts)}{flag}")
        if flagged:
            print("\n!! A source maps to a single class. The model can learn the source")
            print("   instead of the disease. The probe will measure how much.")

    largest: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        largest[row["tree_id"]].append(Path(row["image_path"]).name)
    biggest = sorted(largest.items(), key=lambda kv: -len(kv[1]))[:3]
    if biggest and len(biggest[0][1]) > 1:
        print("\nlargest groups (check these are really the same palm)")
        print("-" * 46)
        for group, names in biggest:
            print(f"  {group}  ({len(names)} images)")
            for name in names[:4]:
                print(f"      {name[:64]}")
            if len(names) > 4:
                print(f"      ... {len(names) - 4} more")

    print("\nNext:")
    print("    python3 -m palmscreen.cli split --config configs/effnet_b0.yaml")
    print("    python3 -m palmscreen.cli probe --config configs/effnet_b0.yaml")


if __name__ == "__main__":
    main()
