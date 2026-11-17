"""Device selection, determinism, and provenance capture.

Apple Silicon first: MPS when available, CPU otherwise. CUDA is *never*
assumed; if someone runs this on an NVIDIA box it is used only when explicitly
requested, so results stay comparable to the M4 Air runs.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    device: str  # "mps" | "cpu" | "cuda"
    reason: str
    mps_available: bool
    mps_built: bool

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "reason": self.reason,
            "mps_available": self.mps_available,
            "mps_built": self.mps_built,
        }


def select_device(prefer: Optional[str] = None) -> DeviceInfo:
    """Return the torch device to use, and say plainly why."""
    import torch

    prefer = prefer or os.environ.get("PALMSCREEN_DEVICE")
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())

    if prefer:
        return DeviceInfo(prefer, f"forced by request/env: {prefer}", mps_available, mps_built)
    if mps_available:
        return DeviceInfo("mps", "Apple Silicon GPU (MPS) available", mps_available, mps_built)
    if mps_built and not mps_available:
        reason = "MPS is built into this torch but unavailable (macOS < 12.3 or no Apple GPU); using CPU"
    else:
        reason = "no MPS backend in this build; using CPU"
    return DeviceInfo("cpu", reason, mps_available, mps_built)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Note: full bit-exact determinism is not
    guaranteed on MPS, so re-runs may differ slightly. The report records this."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "numpy", "sklearn", "PIL", "scipy", "pandas"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def provenance(prefer_device: Optional[str] = None) -> dict:
    """Everything a reader needs to know about *where* a number came from."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": _git_commit(),
        "packages": package_versions(),
    }
    try:
        info["device"] = select_device(prefer_device).as_dict()
    except ImportError:
        info["device"] = {"device": "unknown", "reason": "torch not installed"}
    return info
