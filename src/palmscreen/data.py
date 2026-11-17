"""Torch datasets and transforms.

The evaluation path deliberately calls `preprocess.prepare_pil` / `to_tensor`
instead of torchvision's Resize/CenterCrop so that training-time evaluation and
the FastAPI server run literally the same code. Augmentation exists only on the
training path and is never applied to validation or test images.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import AugmentationConfig
from .preprocess import PreprocessSpec, load_image, prepare_pil, to_tensor


class EvalTransform:
    """Deterministic. Identical to what the API applies at inference time."""

    def __init__(self, spec: PreprocessSpec):
        self.spec = spec

    def __call__(self, img: Image.Image):
        return to_tensor(prepare_pil(img, self.spec), self.spec)


class TrainTransform:
    """Conservative augmentation, then the same normalization as evaluation.

    Every transform here is defended in configs/*.yaml. Anything that could
    invent or destroy the disease signal (heavy hue shifts, vertical flips,
    aggressive crops, MixUp/CutMix) is off unless the data justifies it.
    """

    def __init__(self, spec: PreprocessSpec, aug: AugmentationConfig):
        from torchvision import transforms as T

        self.spec = spec
        self.aug = aug
        interp = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
            "lanczos": T.InterpolationMode.LANCZOS,
            "nearest": T.InterpolationMode.NEAREST,
        }[spec.interpolation]

        steps = [T.Resize(spec.resize_shorter_to, interpolation=interp)]
        if aug.rotation_degrees > 0:
            steps.append(T.RandomRotation(aug.rotation_degrees, interpolation=interp))
        if aug.random_resized_crop:
            steps.append(
                T.RandomResizedCrop(
                    spec.image_size,
                    scale=(aug.crop_scale_min, aug.crop_scale_max),
                    ratio=(0.9, 1.111),  # near-square: do not distort frond geometry
                    interpolation=interp,
                )
            )
        else:
            steps.append(T.CenterCrop(spec.image_size))
        if aug.horizontal_flip:
            steps.append(T.RandomHorizontalFlip(p=0.5))
        if aug.vertical_flip:
            steps.append(T.RandomVerticalFlip(p=0.5))
        if any([aug.brightness, aug.contrast, aug.saturation, aug.hue]):
            steps.append(
                T.ColorJitter(
                    brightness=aug.brightness,
                    contrast=aug.contrast,
                    saturation=aug.saturation,
                    hue=aug.hue,
                )
            )
        steps += [T.ToTensor(), T.Normalize(mean=spec.mean, std=spec.std)]
        self.pipeline = T.Compose(steps)

    def __call__(self, img: Image.Image):
        return self.pipeline(img)


class PalmDataset(Dataset):
    """Rows come from the split manifest, so a dataset can never contain an
    image that was not assigned to this split."""

    def __init__(
        self,
        frame: pd.DataFrame,
        spec: PreprocessSpec,
        transform: Optional[Callable] = None,
        exif_fix: bool = True,
    ):
        required = {"image_path", "label_binary"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {missing}")
        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.transform = transform or EvalTransform(spec)
        self.exif_fix = exif_fix

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        from PIL import ImageOps

        row = self.frame.iloc[idx]
        img = load_image(row["image_path"])
        if self.exif_fix and not isinstance(self.transform, EvalTransform):
            # EvalTransform does EXIF itself; the augmented path needs it applied
            # before any geometric transform.
            img = ImageOps.exif_transpose(img)
            if img.mode != self.spec.color_mode:
                img = img.convert(self.spec.color_mode)
        tensor = self.transform(img)
        return tensor, float(row["label_binary"]), idx


def make_loader(
    frame: pd.DataFrame,
    spec: PreprocessSpec,
    batch_size: int,
    train: bool,
    aug: Optional[AugmentationConfig] = None,
    num_workers: int = 2,
    seed: int = 1337,
) -> DataLoader:
    import torch

    transform = TrainTransform(spec, aug) if train and aug is not None else EvalTransform(spec)
    dataset = PalmDataset(frame, spec, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,  # no effect on MPS/CPU
        drop_last=False,
        generator=generator if train else None,
        persistent_workers=bool(num_workers) and train,
    )
