"""Single source of truth for image preprocessing.

Both training and the FastAPI server import from this module. Nothing else is
allowed to resize, normalize or decode images. The `PreprocessSpec` is persisted
inside the inference bundle so the server reproduces training byte-for-byte.

Test coverage: tests/test_preprocess_parity.py
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

from PIL import Image, ImageOps

# Decompression-bomb guard. PIL's default is ~89M pixels; we are stricter.
Image.MAX_IMAGE_PIXELS = 80_000_000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PIL resampling filter names -> enum. Stored by name so the bundle stays JSON.
_INTERPOLATIONS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


@dataclass
class PreprocessSpec:
    """Everything needed to turn raw bytes into a model-ready tensor."""

    image_size: int = 224
    resize_shorter_to: int = 256  # resize shorter side, then center crop
    mean: list = field(default_factory=lambda: list(IMAGENET_MEAN))
    std: list = field(default_factory=lambda: list(IMAGENET_STD))
    interpolation: str = "bilinear"
    apply_exif_transpose: bool = True
    color_mode: str = "RGB"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessSpec":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def pil_interpolation(self):
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation: {self.interpolation}")
        return _INTERPOLATIONS[self.interpolation]


def load_image(source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """Decode an image from a path or raw bytes.

    Raises PIL.UnidentifiedImageError (or OSError) on corrupt input; callers
    are expected to handle that and report the file as unreadable.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        img = Image.open(source)
    else:
        raise TypeError(f"Unsupported image source type: {type(source)!r}")
    img.load()  # force full decode so truncated files fail here, not later
    return img


def prepare_pil(image: Image.Image, spec: PreprocessSpec) -> Image.Image:
    """EXIF orientation fix -> colour conversion -> resize -> center crop.

    This is the deterministic geometric/photometric path shared by training
    evaluation and serving. Training-time random augmentation happens *before*
    this in the training transform, never here.
    """
    img = image
    if spec.apply_exif_transpose:
        # exif_transpose bakes the orientation tag into pixels and drops the tag,
        # so a phone photo looks the same in training and in the API.
        img = ImageOps.exif_transpose(img)
    if img.mode != spec.color_mode:
        img = img.convert(spec.color_mode)

    # Resize shorter side, preserving aspect ratio.
    w, h = img.size
    target = spec.resize_shorter_to
    if w <= h:
        new_w = target
        new_h = max(target, int(round(h * target / w)))
    else:
        new_h = target
        new_w = max(target, int(round(w * target / h)))
    img = img.resize((new_w, new_h), spec.pil_interpolation)

    # Center crop to the square model input.
    s = spec.image_size
    left = (new_w - s) // 2
    top = (new_h - s) // 2
    return img.crop((left, top, left + s, top + s))


def to_tensor(image: Image.Image, spec: PreprocessSpec):
    """PIL image (already `prepare_pil`-ed) -> normalized CHW float tensor.

    Imported lazily so that the audit/split/metrics code paths work without
    torch installed.
    """
    import numpy as np
    import torch

    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # defensive: greyscale that slipped through
        arr = np.stack([arr] * 3, axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(spec.mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(spec.std, dtype=torch.float32).view(-1, 1, 1)
    return (tensor - mean) / std


def preprocess(source: Any, spec: PreprocessSpec):
    """Full deterministic path: raw source -> normalized tensor (CHW)."""
    return to_tensor(prepare_pil(load_image(source), spec), spec)
