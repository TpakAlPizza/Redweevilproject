#!/usr/bin/env python3
"""Build data/metadata.csv by reading structure out of the filenames.

Written for datasets exported from Roboflow and assembled from screen captures,
where the filenames still carry the information the split needs:

    Screenshot-2024-02-19-204218_4_jpg.rf.98ae0482...jpg
    |__________________________|  |                |
     capture timestamp            variant index    Roboflow content hash

Three things get recovered:

  * `tree_id`   -- all augmented variants of ONE source frame become one group.
                   This is the leak that perceptual hashing misses: rotation and
                   brightness jitter change the hash enough to slip through,
                   while the underlying palm is identical.
  * `session_id`-- captures taken within a few minutes of each other are almost
                   certainly the same video, the same visit, and often the same
                   tree from a slightly different angle.
  * `source_family` -- which pipeline the file came from (Roboflow export,
                   messaging app, camera). If this correlates with the label,
                   the model can learn the source instead of the disease, and
                   the script says so.

This does not invent information. Two frames of genuinely different trees taken
30 seconds apart will be grouped together, which is conservative: it can only
make the split stricter, never looser. Real `tree_id` values from the field are
still better. Use this when those do not exist.

    python3 scripts/build_metadata_from_filenames.py
    python3 scripts/build_metadata_from_filenames.py --session-gap-minutes 10
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Roboflow appends "_jpg.rf.<32 hex chars>.jpg" to every exported file.
ROBOFLOW_SUFFIX = re.compile(r"[._](jpe?g|png|webp)\.rf\.[0-9a-f]{16,}\.\w+$", re.IGNORECASE)
# A trailing "_4" or "-3" before that suffix is the augmentation variant index.
VARIANT_INDEX = re.compile(r"[._-]\d{1,2}$")
# Screenshot-2024-02-19-204218  /  Screenshot_2024-02-19-20-42-18
TIMESTAMP = re.compile(
    r"(?P<y>20\d{2})[-_]?(?P<mo>\d{2})[-_]?(?P<d>\d{2})[-_]?"
    r"(?P<h>\d{2})[-_]?(?P<mi>\d{2})[-_]?(?P<s>\d{2})"
)


def source_stem(filename: str) -> str:
    """Strip the export suffix and the augmentation index down to the frame id."""
    stem = ROBOFLOW_SUFFIX.sub("", filename)
    stem = re.sub(r"\.(jpe?g|png|webp|bmp|tiff?)$", "", stem, flags=re.IGNORECASE)
    stem = VARIANT_INDEX.sub("", stem)
    return stem


def parse_timestamp(filename: str) -> Optional[datetime]:
    match = TIMESTAMP.search(filename)
    if not match:
        return None
    try:
        parts = {k: int(v) for k, v in match.groupdict().items()}
        return datetime(parts["y"], parts["mo"], parts["d"], parts["h"], parts["mi"], parts["s"])
    except ValueError:
        return None


def source_family(filename: str) -> str:
    """Which pipeline produced this file. A shortcut risk when it tracks the label."""
    lower = filename.lower()
    if ".rf." in lower:
        return "roboflow_export"
    if lower.startswith("photo_"):
        return "messaging_app"
    if lower.startswith(("img_", "dsc", "dscn", "p10", "20")):
        return "camera_or_phone"
    if "screenshot" in lower:
        return "screenshot"
    return "other"


def assign_sessions(
    records: List[dict], gap_minutes: int
) -> Dict[str, str]:
    """Cluster timestamped captures into sessions separated by a quiet gap."""
    timed = sorted(
        (r for r in records if r["timestamp"]), key=lambda r: r["timestamp"]
    )
    sessions: Dict[str, str] = {}
    index = 0
    previous: Optional[datetime] = None
    for record in timed:
        if previous is not None and record["timestamp"] - previous > timedelta(minutes=gap_minutes):
            index += 1
        sessions[record["path"]] = f"session_{index:03d}"
        previous = record["timestamp"]

    # Files with no timestamp cannot be placed in a session; each becomes its own,
    # which is the conservative choice.
    for i, record in enumerate(r for r in records if not r["timestamp"]):
        sessions[record["path"]] = f"session_untimed_{i:04d}"
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/metadata.csv")
    parser.add_argument("--classes", nargs="+", default=["healthy", "infected"])
    parser.add_argument("--session-gap-minutes", type=int, default=5,
                        help="captures separated by less than this are one session")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    records: List[dict] = []
    for label in args.classes:
        class_dir = raw / label
        if not class_dir.is_dir():
            print(f"!! missing directory: {class_dir}")
            continue
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            print(f"!! no images found in {class_dir}")
        for path in files:
            records.append({
                "path": path.as_posix(),
                "filename": path.name,
                "label": label,
                "stem": source_stem(path.name),
                "timestamp": parse_timestamp(path.name),
                "family": source_family(path.name),
            })

    if not records:
        raise SystemExit(f"No images found under {raw}/")

    sessions = assign_sessions(records, args.session_gap_minutes)

    rows = []
    for record in records:
        timestamp = record["timestamp"]
        rows.append({
            "image_path": record["path"],
            "label": record["label"],
            # All augmented variants of one source frame share a group. Prefixed
            # with the label so identical stems across classes never merge.
            "tree_id": f"{record['label'][:3]}__{record['stem']}",
            "site_id": record["family"],
            "session_id": sessions[record["path"]],
            "capture_date": timestamp.strftime("%Y-%m-%d") if timestamp else "",
            "camera_id": record["family"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "label", "tree_id", "site_id",
            "session_id", "capture_date", "camera_id",
        ])
        writer.writeheader()
        writer.writerows(rows)

    _summarise(rows, records, out)


def _summarise(rows: List[dict], records: List[dict], out: Path) -> None:
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    print(f"\nWrote {len(rows)} rows to {out}\n")
    print(f"{'class':<12}{'images':>8}{'groups':>9}{'sessions':>10}   collapse")
    print("-" * 56)
    for label, subset in by_label.items():
        groups = len({r["tree_id"] for r in subset})
        sessions = len({r["session_id"] for r in subset})
        ratio = len(subset) / max(groups, 1)
        print(f"{label:<12}{len(subset):>8}{groups:>9}{sessions:>10}   {ratio:.2f} images/group")

    duplicates = Counter(r["tree_id"] for r in rows)
    repeated = [(g, n) for g, n in duplicates.most_common() if n > 1]
    if repeated:
        total_extra = sum(n - 1 for _, n in repeated)
        print(f"\n{len(repeated)} source frame(s) appear more than once "
              f"({total_extra} duplicate images in total).")
        print("These are augmented variants of the same photo. Before this file existed")
        print("they could land in both training and validation, which inflates every score.")
        for group, count in repeated[:5]:
            print(f"    {count}x  {group}")

    # Source family vs label: the shortcut that no split can fix.
    families: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        families[row["site_id"]][row["label"]] += 1
    print("\nsource family x class")
    print("-" * 56)
    problem = False
    for family, counts in families.items():
        total = sum(counts.values())
        purity = max(counts.values()) / total
        flag = ""
        if purity >= 0.95 and total >= 3:
            flag = "  <-- one class only"
            problem = True
        print(f"{family:<24}{dict(counts)}{flag}")
    if problem:
        print("\n!! At least one source family maps to a single class. The model can")
        print("   learn 'images that came from this pipeline are infected' without ever")
        print("   looking at a palm. No splitting strategy fixes this; it needs both")
        print("   classes collected the same way.")

    print("\nNext:")
    print("    python3 -m palmscreen.cli split --config configs/effnet_b0.yaml")
    print("    python3 -m palmscreen.cli probe --config configs/effnet_b0.yaml")
    print("\nThese groups are recovered from filenames, not from the field. Real")
    print("tree_id values recorded at capture time remain the better answer.")


if __name__ == "__main__":
    main()
