"""Environment-driven settings. No secrets, no defaults that surprise you.

    PALMSCREEN_BUNDLE_DIR      pin one bundle (default: artifacts/bundles/latest)
    PALMSCREEN_BUNDLES_ROOT    where bundles live
    PALMSCREEN_REPORTS_DIR     read-only report directory served by the API
    PALMSCREEN_CORS_ORIGINS    comma-separated allowed origins for the frontend
    PALMSCREEN_MAX_UPLOAD_MB   reject larger uploads (default 12)
    PALMSCREEN_MAX_PIXELS      reject decompression bombs (default 40e6)
    PALMSCREEN_STORE_UPLOADS   "true" to persist uploads (default false: nothing is kept)
    PALMSCREEN_UPLOAD_DIR      only used when storing is explicitly enabled
    PALMSCREEN_REQUIRE_BUNDLE  "true" to fail startup when no model is present
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> List[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    bundle_dir: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_BUNDLE_DIR", ""))
    bundles_root: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_BUNDLES_ROOT", "artifacts/bundles"))
    reports_dir: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_REPORTS_DIR", "reports"))
    cors_origins: List[str] = field(
        default_factory=lambda: _list(
            "PALMSCREEN_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:5500",
        )
    )
    cors_allow_credentials: bool = field(default_factory=lambda: _bool("PALMSCREEN_CORS_CREDENTIALS", False))
    max_upload_bytes: int = field(
        default_factory=lambda: int(float(os.environ.get("PALMSCREEN_MAX_UPLOAD_MB", "12")) * 1024 * 1024)
    )
    max_pixels: int = field(default_factory=lambda: int(float(os.environ.get("PALMSCREEN_MAX_PIXELS", "40000000"))))
    min_pixels: int = field(default_factory=lambda: int(os.environ.get("PALMSCREEN_MIN_PIXELS", "1024")))
    store_uploads: bool = field(default_factory=lambda: _bool("PALMSCREEN_STORE_UPLOADS", False))
    upload_dir: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_UPLOAD_DIR", "artifacts/uploads"))
    require_bundle: bool = field(default_factory=lambda: _bool("PALMSCREEN_REQUIRE_BUNDLE", False))
    enable_gradcam: bool = field(default_factory=lambda: _bool("PALMSCREEN_GRADCAM", True))
    feedback_log: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_FEEDBACK_LOG", "artifacts/feedback.jsonl"))
    api_prefix: str = field(default_factory=lambda: os.environ.get("PALMSCREEN_API_PREFIX", "/api/v1"))

    allowed_content_types: tuple = (
        "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff",
    )
    allowed_formats: tuple = ("JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO")

    def public(self) -> dict:
        """Non-sensitive settings, exposed on /healthz for debugging CORS issues."""
        return {
            "bundles_root": self.bundles_root,
            "cors_origins": self.cors_origins,
            "max_upload_mb": round(self.max_upload_bytes / 1024 / 1024, 1),
            "max_pixels": self.max_pixels,
            "store_uploads": self.store_uploads,
            "gradcam_enabled": self.enable_gradcam,
            "api_prefix": self.api_prefix,
        }


settings = Settings()
