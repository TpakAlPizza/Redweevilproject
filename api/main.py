"""FastAPI service for palm infection screening.

Serves two things from one process:

  1. The existing Arabic frontend (web/templates/index.html + web/static/**),
     so the browser calls /predict same-origin and CORS never enters the picture.
  2. The versioned JSON API under /api/v1 for any other client.

Serving guarantees:
  * the image is decoded, EXIF-rotated, converted, resized and normalized by
    `palmscreen.preprocess` -- the same module training used;
  * the probability is calibrated and thresholded with the values frozen in the
    bundle, not with 0.5;
  * uploads are handled in memory and discarded unless storage is explicitly
    switched on;
  * every response repeats that this is a screening aid.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palmscreen.bundle import DISCLAIMER, InferenceBundle, resolve_latest  # noqa: E402
from palmscreen.report import latest_report  # noqa: E402

from .schemas import (  # noqa: E402
    ErrorResponse, HealthResponse, LegacyPredictionResponse, ModelInfoResponse,
    PredictionResponse,
)
from .settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("palmscreen.api")

WEB_DIR = ROOT / "web"
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"
STATIC_DIR = WEB_DIR / "static"

_bundle: Optional[InferenceBundle] = None
_bundle_error: Optional[str] = None


def load_bundle(force: bool = False) -> Optional[InferenceBundle]:
    """Load the inference bundle once, at startup."""
    global _bundle, _bundle_error
    if _bundle is not None and not force:
        return _bundle
    try:
        target = settings.bundle_dir or resolve_latest(settings.bundles_root)
        if not target:
            raise FileNotFoundError(
                f"No inference bundle found under {settings.bundles_root}. "
                "Train a model first (see README) or set PALMSCREEN_BUNDLE_DIR."
            )
        _bundle = InferenceBundle.load(target)
        _bundle_error = None
        logger.info("Loaded bundle %s (model %s)", target, _bundle.meta.get("model_version"))
    except Exception as exc:
        _bundle, _bundle_error = None, f"{type(exc).__name__}: {exc}"
        logger.warning("No model loaded: %s", _bundle_error)
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()
    if settings.require_bundle and _bundle is None:
        raise RuntimeError(f"PALMSCREEN_REQUIRE_BUNDLE is set but no bundle loaded: {_bundle_error}")
    yield


app = FastAPI(
    title="Palm infection screening API",
    version="1.0.0",
    description=(
        "Screening aid for palm-tree infection from a photograph. "
        "Not an agricultural diagnosis. Confirm every result with a qualified agronomist."
    ),
    lifespan=lifespan,
)

# CORS matters only when the frontend is served from a different origin. When the
# page comes from this app (the default), requests are same-origin already.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _require_bundle() -> InferenceBundle:
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": "No trained model is loaded, so no prediction can be made.",
                "message_ar": "لم يتم تحميل نموذج مدرَّب بعد، لذا لا يمكن إجراء الفحص.",
                "hint": _bundle_error,
            },
        )
    return bundle


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

async def _read_upload(file: UploadFile) -> bytes:
    """Read with a hard byte cap. Never trusts the declared content-length."""
    if file.content_type and file.content_type.split(";")[0].strip().lower() not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_media_type",
                "message": f"Content-Type '{file.content_type}' is not an accepted image type.",
                "message_ar": "نوع الملف غير مدعوم. يرجى رفع صورة بصيغة JPG أو PNG أو WebP.",
                "accepted": list(settings.allowed_content_types),
            },
        )
    chunks, total, limit = [], 0, settings.max_upload_bytes
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "message": f"Image exceeds the {limit // (1024 * 1024)} MB limit.",
                    "message_ar": f"حجم الملف كبير جداً. الحد الأقصى هو {limit // (1024 * 1024)} ميجابايت.",
                },
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "The uploaded file is empty.",
                    "message_ar": "الملف المرفوع فارغ."},
        )
    return b"".join(chunks)


def _validate_image(data: bytes) -> Tuple[int, int, str]:
    """Decode the bytes and check them. The declared MIME type is only a hint;
    what matters is whether PIL can actually decode an image of a sane size."""
    from PIL import Image, UnidentifiedImageError

    try:
        probe = Image.open(io.BytesIO(data))
        image_format = (probe.format or "").upper()
        width, height = probe.size
        probe.verify()  # structural check; consumes the probe object
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": "The uploaded file is not a readable image.",
                "message_ar": "الملف غير صالح أو تالف. يرجى رفع صورة سليمة.",
                "hint": f"{type(exc).__name__}",
            },
        )

    if image_format not in settings.allowed_formats:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_image_format",
                    "message": f"Image format '{image_format}' is not supported.",
                    "message_ar": "صيغة الصورة غير مدعومة.",
                    "accepted": list(settings.allowed_formats)},
        )
    pixels = width * height
    if pixels > settings.max_pixels:
        raise HTTPException(
            status_code=413,
            detail={"error": "image_too_large",
                    "message": f"{width}x{height} exceeds the {settings.max_pixels:,}-pixel limit.",
                    "message_ar": "أبعاد الصورة كبيرة جداً."},
        )
    if pixels < settings.min_pixels:
        raise HTTPException(
            status_code=400,
            detail={"error": "image_too_small",
                    "message": f"{width}x{height} is too small to screen. Send a clear photo of the palm.",
                    "message_ar": "الصورة صغيرة جداً للفحص. يرجى رفع صورة أوضح للنخلة."},
        )
    return width, height, image_format


async def _screen(file: UploadFile, keep_bytes: bool = False) -> dict:
    """Shared prediction path for both the JSON API and the frontend endpoint.

    `keep_bytes` returns the decoded upload under `_bytes` so an explanation can
    be rendered from the same image. It is still never written to disk.
    """
    bundle = _require_bundle()
    started = time.time()
    data = await _read_upload(file)
    width, height, image_format = _validate_image(data)

    try:
        result = bundle.predict(data)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": "The image could not be processed.",
                    "message_ar": "تعذّر تحليل الصورة.", "hint": type(exc).__name__},
        )
    if keep_bytes:
        result["_bytes"] = data

    result["image"] = {"width": width, "height": height, "format": image_format,
                       "filename": Path(file.filename or "upload").name}
    result["inference_ms"] = round((time.time() - started) * 1000, 1)
    logger.info("prediction label=%s p=%.3f review=%s %.0fms",
                result["label"], result["probability_infected"], result["needs_review"],
                result["inference_ms"])
    return result


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    """Serve the existing Arabic UI. Falls back to a JSON index if web/ is absent
    (for example in an API-only container)."""
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "service": "palm-screen",
            "docs": "/docs",
            "health": "/healthz",
            "predict": settings.api_prefix + "/predict",
            "disclaimer": DISCLAIMER,
        }
    )


@app.post("/predict", response_model=LegacyPredictionResponse, tags=["prediction"])
async def predict_for_frontend(file: UploadFile = File(...)):
    """Endpoint the bundled frontend calls.

    Returns a superset of the original contract: `class_name`, `confidence` and
    `uncertain` keep the old shape working, while the fields the UI should
    actually display (calibrated probability, threshold, model version,
    disclaimer) are added alongside.

    `confidence` is retained only for backward compatibility and is deliberately
    NOT what the updated UI shows. It is the calibrated probability of the
    *predicted* class, which reads as reassuring precisely when the model is
    least useful: a borderline 0.51 becomes "51% confident" for whichever side
    it happened to land on.
    """
    result = await _screen(file, keep_bytes=True)
    probability_infected = float(result["probability_infected"])
    predicted_class_probability = (
        probability_infected if result["label"] == "infected" else 1.0 - probability_infected
    )

    warning = None
    if result["needs_review"]:
        warning = (
            "This score sits close to the decision threshold and cannot be called either way. "
            "Have the palm inspected."
        )

    gradcam_b64 = None
    raw_bytes = result.pop("_bytes", None)
    if settings.enable_gradcam and raw_bytes:
        try:
            from palmscreen.gradcam import gradcam_overlay_base64

            gradcam_b64 = gradcam_overlay_base64(_require_bundle(), raw_bytes)
        except Exception as exc:  # an explanation is optional; a prediction is not
            logger.warning("Grad-CAM failed: %s", exc)

    return {
        # original contract, preserved
        "class_name": "Infected" if result["label"] == "infected" else "Healthy",
        "confidence": round(predicted_class_probability * 100, 1),
        "uncertain": bool(result["needs_review"]),
        # what the UI should actually render
        "probability_infected": round(probability_infected, 6),
        "probability_infected_percent": round(probability_infected * 100, 1),
        "threshold": float(result["threshold"]),
        "threshold_percent": round(float(result["threshold"]) * 100, 1),
        "needs_review": bool(result["needs_review"]),
        "label": result["label"],
        "model_version": result["model_version"],
        "disclaimer": result["disclaimer"],
        "inference_ms": result.get("inference_ms"),
        # Fields the bundled UI reads
        "class_idx": 1 if result["label"] == "infected" else 0,
        "confidence_pct": f"{round(probability_infected * 100, 1)}%",
        "probability_healthy": round(1.0 - probability_infected, 6),
        "inference_time_ms": result.get("inference_ms"),
        "warning": warning,
        "gradcam_base64": gradcam_b64,
    }


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz():
    bundle = load_bundle()
    return {
        "status": "ok" if bundle else "degraded",
        "model_loaded": bundle is not None,
        "model_version": bundle.meta.get("model_version") if bundle else None,
        "bundle_error": None if bundle else _bundle_error,
        "settings": settings.public(),
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_alias():
    """Alias kept for the original web/app.py health path."""
    return healthz()


@app.get(settings.api_prefix + "/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    return _require_bundle().info()


@app.post(settings.api_prefix + "/predict", response_model=PredictionResponse,
          responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                     415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
          tags=["prediction"])
async def predict(file: UploadFile = File(..., description="Palm photograph")):
    return await _screen(file)


@app.post(settings.api_prefix + "/feedback", tags=["feedback"])
async def submit_feedback(request: Request):
    """Record whether a user judged a prediction correct.

    This APPENDS TO A LOG. It does not update the model, and the UI must not
    imply that it does. Retraining on user-reported labels without review would
    be a fast way to corrupt a model, especially one whose users cannot verify
    the ground truth from a photograph.
    """
    payload = await request.json()
    verdict = str(payload.get("verdict", "")).lower()
    if verdict not in {"correct", "incorrect"}:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_verdict", "message": "verdict must be 'correct' or 'incorrect'."},
        )

    entry = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "predicted_label": payload.get("predicted_label"),
        "probability_infected": payload.get("probability_infected"),
        "model_version": payload.get("model_version"),
    }
    path = Path(settings.feedback_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("feedback %s for %s", verdict, entry.get("predicted_label"))

    return {
        "status": "recorded",
        "note": (
            "Feedback is stored for human review. The model is not retrained "
            "automatically and this prediction is unchanged."
        ),
    }


@app.get(settings.api_prefix + "/stats", tags=["reports"])
def stats():
    """Headline numbers for the UI, read from the latest evaluation report.

    Served from the report rather than hard-coded into the page, so the figures
    on the site cannot drift away from the measured ones.
    """
    found = latest_report(settings.reports_dir)
    if not found:
        return JSONResponse({"available": False, "reason": "no evaluation report yet"})
    payload = json.loads((Path(found["dir"]) / "report.json").read_text())
    test = payload.get("locked_test_evaluation")
    if not test:
        return JSONResponse({"available": False, "reason": "model has not been evaluated"})

    m = test["metrics"]
    wilson = (test.get("wilson_intervals") or {}).get("accuracy") or {}
    return JSONResponse({
        "available": True,
        "n_test_images": test["n_test_images"],
        "n_test_groups": test["n_test_groups"],
        "accuracy": m["accuracy"],
        "accuracy_ci": [wilson.get("lower"), wilson.get("upper")],
        "balanced_accuracy": m["balanced_accuracy"],
        "roc_auc": m["roc_auc"],
        "sensitivity_infected": m["sensitivity_infected"],
        "specificity_healthy": m["specificity_healthy"],
        "model_version": payload.get("model_version"),
        "report_url": f"{settings.api_prefix}/reports/latest.html",
        "caveat": (
            f"Measured once on {test['n_test_images']} images. The confidence interval "
            "is the result; the point estimate alone overstates what is known."
        ),
    })


@app.get(settings.api_prefix + "/reports/latest", tags=["reports"])
def reports_latest():
    """Read-only JSON of the most recent evaluation report."""
    found = latest_report(settings.reports_dir)
    if not found:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_report",
                    "message": "No evaluation report has been generated yet.",
                    "message_ar": "لم يتم إنشاء تقرير تقييم بعد.",
                    "hint": "Run: palmscreen report --config configs/effnet_b0.yaml"},
        )
    payload = json.loads((Path(found["dir"]) / "report.json").read_text())
    return JSONResponse(
        {
            "report_version": payload.get("report_version"),
            "generated_at": payload.get("generated_at"),
            "model_version": payload.get("model_version"),
            "grouping": payload.get("grouping"),
            "limitations": payload.get("limitations"),
            "development_cross_validation": (payload.get("development_cross_validation") or {}).get(
                "oof_ranking_metrics"
            ),
            "locked_test_evaluation": _test_summary(payload.get("locked_test_evaluation")),
            "html_url": f"{settings.api_prefix}/reports/latest.html",
            "pdf_url": f"{settings.api_prefix}/reports/latest.pdf",
            "disclaimer": DISCLAIMER,
        }
    )


def _test_summary(test: Optional[dict]) -> Optional[dict]:
    if not test:
        return None
    m = test["metrics"]
    return {
        "n_test_images": test["n_test_images"],
        "n_test_groups": test["n_test_groups"],
        "threshold": test["threshold"],
        "confusion_matrix": {k: m[k] for k in ("tp", "fp", "tn", "fn")},
        "sensitivity_infected": m["sensitivity_infected"],
        "specificity_healthy": m["specificity_healthy"],
        "precision_infected": m["precision_infected"],
        "balanced_accuracy": m["balanced_accuracy"],
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "ece": test.get("ece"),
        "wilson_intervals": test.get("wilson_intervals"),
        "caveat": (
            f"Measured once on {test['n_test_images']} images. Wide uncertainty; "
            "not a demonstration of field performance."
        ),
    }


def _report_file(filename: str, media_type: str):
    found = latest_report(settings.reports_dir)
    if not found:
        raise HTTPException(status_code=404, detail={"error": "no_report", "message": "No report generated yet."})
    # Resolve inside the reports directory only; never accept a client-supplied path.
    path = (Path(found["dir"]) / filename).resolve()
    if not str(path).startswith(str(Path(settings.reports_dir).resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": f"{filename} is unavailable."})
    return FileResponse(path, media_type=media_type)


@app.get(settings.api_prefix + "/reports/latest.html", tags=["reports"])
def reports_latest_html():
    return _report_file("report.html", "text/html")


@app.get(settings.api_prefix + "/reports/latest.pdf", tags=["reports"])
def reports_latest_pdf():
    return _report_file("report.pdf", "application/pdf")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": "error", "message": str(exc.detail)}
    detail = dict(detail)
    # The bundled frontend reads `detail`; the JSON API reads `error`/`message`.
    # Both are populated so neither client needs a special case.
    detail.setdefault("detail", detail.get("message_ar") or detail.get("message"))
    detail.setdefault("disclaimer", DISCLAIMER)
    return JSONResponse(status_code=exc.status_code, content=detail)
