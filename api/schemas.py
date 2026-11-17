"""Response models. These define the contract the frontend codes against."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImageMeta(BaseModel):
    width: int
    height: int
    format: str
    filename: str


class PredictionResponse(BaseModel):
    label: str = Field(..., description="'infected' or 'healthy'")
    probability_infected: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated probability that the palm is infected. Not a raw network score.",
    )
    threshold: float = Field(..., description="Decision threshold frozen at training time.")
    needs_review: bool = Field(
        ..., description="True when the probability sits close to the threshold; show it to a human."
    )
    model_version: str
    disclaimer: str
    image: Optional[ImageMeta] = None
    inference_ms: Optional[float] = None

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "label": "infected",
                "probability_infected": 0.83,
                "threshold": 0.42,
                "needs_review": False,
                "model_version": "0.1.0+efficientnet_b0.20250907120000",
                "disclaimer": "Screening aid only. This is not an agricultural diagnosis.",
                "image": {"width": 3024, "height": 4032, "format": "JPEG", "filename": "palm.jpg"},
                "inference_ms": 118.4,
            }
        },
    }


class LegacyPredictionResponse(BaseModel):
    """Response for POST /predict, the endpoint the bundled frontend calls.

    A superset of the original contract. The first three fields keep any
    existing client working; the rest are what the UI should actually show.
    """

    # Original contract
    class_name: str = Field(..., description="'Infected' or 'Healthy'")
    confidence: float = Field(
        ...,
        description=(
            "DEPRECATED. Calibrated probability of the predicted class, as a percentage. "
            "Kept for backward compatibility only. Prefer probability_infected: this field "
            "reports a reassuring-looking number for borderline cases."
        ),
    )
    uncertain: bool = Field(..., description="Same value as needs_review.")

    # What the UI should render
    probability_infected: float = Field(..., ge=0.0, le=1.0)
    probability_infected_percent: float
    threshold: float
    threshold_percent: float
    needs_review: bool
    label: str
    model_version: str
    disclaimer: str
    inference_ms: Optional[float] = None

    # Fields the bundled UI reads. FastAPI validates the response against this
    # model and drops anything not declared here, so every key the frontend
    # depends on must be listed.
    class_idx: int = Field(..., description="0 = healthy, 1 = infected")
    confidence_pct: str = Field(..., description="Calibrated probability of infection, formatted.")
    probability_healthy: float = Field(..., ge=0.0, le=1.0)
    inference_time_ms: Optional[float] = None
    warning: Optional[str] = Field(
        None, description="Set when the score is too close to the threshold to call."
    )
    gradcam_base64: Optional[str] = Field(
        None, description="PNG overlay showing where the model looked. A debugging aid, not proof."
    )

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    bundle_error: Optional[str] = None
    settings: Dict[str, Any]
    server_time: str

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    model_version: Optional[str] = None
    arch: Optional[str] = None
    created_at: Optional[str] = None
    image_size: Optional[int] = None
    preprocess: Optional[Dict[str, Any]] = None
    class_map: Optional[Dict[str, str]] = None
    positive_class: Optional[str] = None
    calibration_method: Optional[str] = None
    calibration_fitted_on: Optional[str] = None
    threshold: Optional[Dict[str, Any]] = None
    weights_sha256: Optional[str] = None
    config_fingerprint: Optional[str] = None
    n_parameters: Optional[int] = None
    training_summary: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    error: str
    message: str
    hint: Optional[str] = None
    accepted: Optional[List[str]] = None
    disclaimer: Optional[str] = None
