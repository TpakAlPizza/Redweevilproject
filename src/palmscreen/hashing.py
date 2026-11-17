"""SHA-256 (exact duplicates) and perceptual hashes (near duplicates).

Near-duplicate detection is a leakage control, not a nicety: two photos of the
same frond taken two seconds apart will land in train and test under a random
split and inflate every metric in the report.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .preprocess import load_image


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _grey(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(
        img.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float64
    )


def dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Difference hash: robust to mild scaling/compression."""
    px = _grey(img, (hash_size + 1, hash_size))
    bits = px[:, 1:] > px[:, :-1]
    return _bits_to_hex(bits.flatten())


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """DCT-based perceptual hash: robust to brightness/contrast shifts."""
    from scipy.fftpack import dct

    size = hash_size * highfreq_factor
    px = _grey(img, (size, size))
    coeffs = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude DC term from the median
    return _bits_to_hex((low > med).flatten())


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return format(value, f"0{len(bits) // 4}x")


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def hash_image_file(path: str | Path) -> Dict[str, str]:
    img = load_image(path)
    return {
        "sha256": sha256_file(path),
        "phash": phash(img),
        "dhash": dhash(img),
    }


def exact_duplicate_groups(records: Sequence[dict]) -> List[List[str]]:
    """Group image paths that share a SHA-256."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        buckets[r["sha256"]].append(r["image_path"])
    return [sorted(v) for v in buckets.values() if len(v) > 1]


def near_duplicate_clusters(
    records: Sequence[dict], max_distance: int = 6, hash_field: str = "phash"
) -> List[List[str]]:
    """Single-linkage clustering over perceptual-hash Hamming distance.

    O(n^2), which is entirely fine for a few thousand images and keeps the
    dependency footprint at zero. Returns clusters of size >= 2, sorted.
    """
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    hashes = [r[hash_field] for r in records]
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashes[i], hashes[j]) <= max_distance:
                union(i, j)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for idx, rec in enumerate(records):
        clusters[find(idx)].append(rec["image_path"])
    return sorted(
        (sorted(v) for v in clusters.values() if len(v) > 1), key=lambda c: c[0]
    )


def phash_cluster_ids(
    records: Sequence[dict], max_distance: int = 6
) -> Dict[str, str]:
    """Map every image to a pseudo-group id derived from near-duplicate clusters.

    Used only when real tree/session metadata is missing. Singletons get their
    own group so they are still splittable.
    """
    clusters = near_duplicate_clusters(records, max_distance=max_distance)
    lookup: Dict[str, str] = {}
    for i, cluster in enumerate(clusters):
        for path in cluster:
            lookup[path] = f"phash_cluster_{i:04d}"
    for i, rec in enumerate(records):
        lookup.setdefault(rec["image_path"], f"phash_single_{i:04d}")
    return lookup
