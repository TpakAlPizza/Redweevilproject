# CPU-only inference image.
#
# MPS is a local-Mac training accelerator and does not exist in a Linux
# container, so the CPU wheels of torch are installed deliberately here. That
# also keeps the image ~800 MB instead of several GB of unusable CUDA payload.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PALMSCREEN_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 libwebp7 libtiff6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch from the PyTorch CPU index; everything else from PyPI.
COPY requirements-serve.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 \
 && pip install -r requirements-serve.txt

COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
# The frontend is served by the same process, so the browser calls /predict
# same-origin and CORS is not involved in the default deployment.
COPY web/ ./web/

# Bundles and reports are mounted at runtime rather than baked in: the image is
# the code, the bundle is the model, and they version independently.
RUN mkdir -p /app/artifacts/bundles /app/reports \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src:/app \
    PALMSCREEN_BUNDLES_ROOT=/app/artifacts/bundles \
    PALMSCREEN_REPORTS_DIR=/app/reports \
    PORT=8000

EXPOSE 8000

# Fails when the process is dead, passes while it is up. Deliberately does NOT
# require a loaded model: a container with no bundle should report "degraded"
# and stay alive so the frontend can show a useful message.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
