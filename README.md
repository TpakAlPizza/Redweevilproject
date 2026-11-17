# Palm infection screening

Binary image screening (`infected` vs `healthy`) for palm trees: a reproducible training
pipeline, an honest evaluation protocol, and a FastAPI backend that serves the trained model
to a web frontend.

**This is a screening aid, not an agricultural diagnosis.** Nothing it produces should be the
sole basis for treating, removing or quarantining a tree.

---

## Status: no model has been trained

The repository contains the pipeline. It does not contain results, because no real images have
been trained on yet.

Every performance figure in this project comes from running the commands below on your data.
Until you do, `reports/` will contain a report that says, in as many words, that it contains no
measured performance — that is deliberate, not a gap to be filled in with a plausible number.

### What was verified here, and what was not

| Verified by running it | Not executed |
|---|---|
| `audit` end to end on 258 synthetic images | Training loop (`train.py`) |
| `split` end to end: 54 test / 204 dev, 0 group overlap | Model builders (`models.py`) |
| Full report generation: HTML, JSON, CSV, PDF | Grad-CAM |
| 83 tests passing (leakage, metrics, calibration, API, reports) | Bundle inference path |
| API: 19 tests including invalid uploads, CORS, 503-with-no-model | |
| Frontend served end to end: page, CSS, JS, `/predict` with a stub model | |

PyTorch could not be installed in the environment where this was built, so the torch-dependent
modules are written but unrun. Three preprocessing-parity tests skip without torch and will
execute on your Mac. Treat the first `crossval` run as the real smoke test.

---

## Setup

Requires Python 3.11 or 3.12 on macOS (Apple Silicon).

```bash
git clone <this repo> && cd palm-screen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src:.
```

Confirm the GPU is visible:

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

`True` means training uses the M4's GPU. `False` falls back to CPU automatically — slower, but
correct. There are no CUDA assumptions anywhere; every device selection goes through
`palmscreen.runtime.select_device`, which logs which device it chose and why.

---

## 1. Place the data

```
data/raw/infected/    ~130 images
data/raw/healthy/     ~130 images
```

Any of `.jpg .jpeg .png .webp .bmp .tif .tiff`. Subfolders are fine.

### Grouping metadata — the single most important input

Create `data/metadata.csv`:

```csv
image_path,label,tree_id,site_id,session_id,capture_date,camera_id
data/raw/infected/IMG_0421.jpg,infected,tree_017,farm_north,2025-03-11_am,2025-03-11,iphone15
data/raw/infected/IMG_0422.jpg,infected,tree_017,farm_north,2025-03-11_am,2025-03-11,iphone15
data/raw/healthy/IMG_0510.jpg,healthy,tree_044,farm_south,2025-03-14_pm,2025-03-14,iphone15
```

`tree_id` is what matters. **Two photos of the same palm must never sit on opposite sides of a
train/test split** — if they do, the model can memorise that individual tree and the resulting
accuracy is measuring recall of the training set, not disease detection. With ~130 images per
class, a handful of trees photographed repeatedly can inflate a score by tens of points.

Only `image_path` and one grouping column are required, but fill in what you have. `site_id`,
`capture_date` and `camera_id` also drive the shortcut detection in the audit.

If you have no metadata, run the audit anyway: it writes `data/metadata.csv.template`
pre-filled with every image path, so you only fill in the group columns.

**Without grouping metadata** the pipeline falls back to perceptual-hash pseudo-groups (catches
near-duplicates, does not catch two different photos of one tree), and failing that to an
image-level split. Both fallbacks are loudly labelled in every report as potentially optimistic.
Set `data.allow_image_level_split: false` in the config to make the run fail instead.

---

## 2. Audit the data before training anything

```bash
make audit          # or: python3 -m palmscreen.cli audit --config configs/effnet_b0.yaml
```

Writes to `reports/audit/<timestamp>/`:

- `audit.json` — full findings
- `catalog.csv` — every image with SHA-256, perceptual hash, dimensions
- `montage_infected.png`, `montage_healthy.png` — **open these and check the labels by eye**
- `montage_cross_class_near_dupes.png` — near-identical images that carry different labels;
  these are usually mislabels

It checks for unreadable files, exact duplicates, near-duplicates, class imbalance, conflicting
labels, and shortcut signals: metadata fields that predict the class on their own (one camera
per class, one site per class), resolutions unique to one class, and large average colour
differences between classes.

That last one matters. If infected photos were taken at midday and healthy ones at dusk, a model
reaches high accuracy on brightness alone and collapses in the field. The audit reports Cohen's d
per channel and warns above 1.5.

**A question the audit cannot answer for you:** is the disease actually visible in these frames?
At 224×224, symptoms occupying a few dozen pixels are effectively invisible. If your photos are
mostly whole trees at distance, whole-image classification is the wrong tool — you need crop/ROI
annotations around symptomatic tissue, or a palm detector feeding a classifier on each crown.
The audit prints this guidance; deciding it is a human job.

---

## 3. Split

```bash
make split
```

- Locks a **~20% test set, grouped and class-stratified**. Whole groups move together.
- Splits the remaining development data into **5 grouped, stratified folds**.
- Asserts no group, no SHA-256 and no near-duplicate pair crosses a split boundary.
  A violation raises `LeakageError` and stops the run — it is not a warning.
- Writes a versioned manifest to `artifacts/splits/` recording every image, its group, its
  split, its hashes, the seed, and the config fingerprint.

For a genuinely stronger test set, hold out an entire farm or collection session:

```yaml
data:
  holdout_column: site_id
  holdout_values: ["farm_B"]
```

A random grouped split answers "does this work on new trees from the same collection?"
A held-out site answers "does this work somewhere else?" — the question you actually care about.

---

## 3b. Probe for shortcuts before you trust any score

```bash
make probe
```

Trains four deliberately stupid baselines on your exact grouped folds: average RGB, a colour
histogram, an 8x8 greyscale thumbnail, and file dimensions alone. Fine detail cannot survive an
8x8 downsample, so anything that still separates the classes afterwards is not the disease.

```
probe                               ROC-AUC   PR-AUC   verdict
mean_rgb                              1.000    1.000   SHORTCUT: this alone nearly solves the task
colour_histogram                      1.000    1.000   SHORTCUT: this alone nearly solves the task
tiny_greyscale_8x8                    0.993    0.994   SHORTCUT: this alone nearly solves the task
acquisition_metadata                  0.537    0.536   little signal (expected)
colour_histogram (labels shuffled)    0.517    0.508   ok
```

A probe above ~0.90 means a high CV score proves nothing about disease detection. Anything above
~0.75 means part of the network's score is acquisition, not infection.

The last row is a negative control: labels shuffled between groups. It must land near 0.50. If it
does not, the folds themselves are broken and no other number in the project can be trusted.

**Run this before reading any cross-validation result.** It costs seconds and it decides whether
the rest of the numbers mean anything.

## 4. Train

```bash
make crossval             # EfficientNet-B0, grouped 5-fold
make crossval-baseline    # MobileNetV3-Large, same protocol
```

Both use ImageNet-pretrained weights and a two-phase schedule: train the head with the backbone
frozen, then unfreeze the top blocks at a 10× lower learning rate. Nothing is trained from
scratch — 260 images cannot support it.

Frozen BatchNorm layers are kept in eval mode. At batch size 8–16, letting them keep updating
their running statistics quietly corrupts the pretrained features and produces a train/eval gap
that looks like overfitting but is not.

Interrupted runs resume: `python3 -m palmscreen.cli crossval --config ... --resume`.

Expect roughly 10–25 minutes per fold on an M4 Air with MPS. If memory pressure appears, drop
`train.batch_size` to 8.

### Choosing between the two models

```bash
make select
```

Compares the candidates on **development out-of-fold predictions only** using a paired,
group-aware bootstrap. If the 95% interval for the difference contains zero, the models are
statistically indistinguishable on this data and the smaller/faster one wins automatically.
On ~260 images, that is the usual outcome, and it is the right answer.

### Freeze the model

```bash
make finalize
```

Refits on all development data for the median best-epoch count from CV, then fits the
probability calibrator and selects the decision threshold — both from out-of-fold predictions
only — and writes a versioned inference bundle to `artifacts/bundles/`.

The bundle holds the weights, the exact preprocessing spec, the class map, the calibrator, the
threshold, the config and the provenance in one directory, with a SHA-256 over the weights. The
API refuses to load a bundle whose weights do not match that hash, so a model can never be served
with someone else's threshold.

**The threshold is not 0.5.** It is chosen as the most specific threshold that still reaches the
target infected-recall (default 90%), because missing an infected palm costs a tree while a false
alarm costs an inspection. Change the trade-off in the config:

```yaml
threshold:
  policy: max_specificity_at_recall   # or youden, max_f1, fixed
  target_recall: 0.90
  review_band: 0.10    # predictions this close to the threshold are flagged needs_review
```

---

## 5. Evaluate — once

```bash
make evaluate
```

This is the only command that touches the locked test set, and it runs **after** the model,
threshold and calibration are fixed.

Every evaluation is recorded in `artifacts/test_evaluation_ledger.jsonl` keyed by the weights
hash. Running it twice on the same weights is refused:

```
REFUSED: These weights were already evaluated on the locked test set on 20250907-1420 (1 time(s)).
Re-running and reporting the better result turns the test set into a validation set.
```

`--allow-repeat` exists for genuine cases (a data bug, say), and the repeat is stamped into the
ledger and printed as a red banner in the report. That is the point: the escape hatch is visible.

---

## 6. Report

```bash
make report
```

Writes `reports/report_<version>/` containing `report.html` (self-contained, charts embedded),
`report.json`, several CSVs, and `report.pdf`.

Structure:

- **Limitations at the top**, before any number, generated from the actual counts.
- **Development cross-validation**, tagged *NOT PERFORMANCE* — these numbers chose the model,
  the threshold and the calibrator, so quoting them as performance means grading the exam with
  the answer key.
- **Locked test results**, in a separate section, with confusion matrix in raw counts,
  sensitivity/specificity/precision/NPV/macro-F1/balanced accuracy/MCC/ROC-AUC/PR-AUC/log loss/
  Brier/ECE, ROC, PR, calibration and learning-curve charts, and grids of the most confident
  false positives and false negatives.
- **Confidence intervals on everything**: Wilson for sensitivity and specificity (correct on
  small counts), group-aware bootstrap for the rest (resampling trees, not photos, so eight
  images of one palm are not counted as eight independent observations).
- **A prevalence table.** Sensitivity and specificity do not change with prevalence; precision
  and NPV do. Your dataset is ~50% infected. A real grove is not. At 5% prevalence, a model with
  90% sensitivity and 90% specificity produces roughly two false alarms for every true detection.

With ~52 test images, the intervals will be wide — roughly ±15 percentage points on sensitivity.
That is not a flaw in the report, it is the actual state of the evidence, and the report says so
in plain language rather than burying it.

### Optional: Grad-CAM

```bash
python3 -m palmscreen.cli gradcam --images data/raw/infected/IMG_0421.jpg
```

Useful for spotting a model that has locked onto a watermark, a fence, or the sky. It is a
debugging tool. A plausible heat map is not evidence that a prediction is correct, and the
generated images carry that disclaimer.

---

## 7. Run the backend

```bash
make serve      # uvicorn api.main:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/predict` | multipart upload, field name `file` |
| `GET /healthz` | liveness + whether a model is loaded |
| `GET /api/v1/model-info` | version, threshold, calibration, preprocessing |
| `GET /api/v1/reports/latest` | read-only JSON report summary |
| `GET /api/v1/reports/latest.html` | full report |
| `GET /docs` | interactive OpenAPI docs |

Response:

```json
{
  "label": "infected",
  "probability_infected": 0.83,
  "threshold": 0.42,
  "needs_review": false,
  "model_version": "0.1.0+efficientnet_b0.20250907120000",
  "disclaimer": "Screening aid only. This is not an agricultural diagnosis. ..."
}
```

The server applies the identical EXIF handling, RGB conversion, resize, crop, normalization,
calibration and threshold used in training — literally the same module, which is what
`tests/test_preprocess_parity.py` pins down.

Uploads are validated on content-type, actual decodability, byte size and pixel count, held in
memory, and discarded. **No user image is written to disk** unless `PALMSCREEN_STORE_UPLOADS=true`.

Test it:

```bash
./scripts/sample_request.sh path/to/palm.jpg
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PALMSCREEN_CORS_ORIGINS` | localhost dev ports | comma-separated frontend origins |
| `PALMSCREEN_BUNDLE_DIR` | latest bundle | pin a specific model version |
| `PALMSCREEN_MAX_UPLOAD_MB` | `12` | upload size cap |
| `PALMSCREEN_MAX_PIXELS` | `40000000` | decompression-bomb guard |
| `PALMSCREEN_STORE_UPLOADS` | `false` | persist user images |
| `PALMSCREEN_REQUIRE_BUNDLE` | `false` | fail startup when no model is present |
| `PALMSCREEN_DEVICE` | auto | force `cpu` or `mps` |

---

## 8. The frontend

The Arabic RTL frontend (`web/`) is served by the same FastAPI process, so the browser calls
`/predict` same-origin and CORS is not involved at all in the default deployment.

```bash
make serve      # http://localhost:8000
```

| Path | Serves |
|---|---|
| `/` | `web/templates/index.html` |
| `/static/**` | `web/static/css/style.css`, `web/static/js/app.js` |
| `POST /predict` | what the page calls; superset of the original response shape |
| `/health`, `/healthz` | health check (the first path is kept from the original `web/app.py`) |
| `/api/v1/**` | versioned JSON API for any other client |

### What changed in the frontend, and why

The layout, stylesheet, colour system, drag-and-drop, RTL direction and every DOM id were kept.
The changes are confined to claims the model cannot support:

| Before | After | Reason |
|---|---|---|
| "نظام تشخيص سوسة النخيل الحمراء" | "نظام الفرز المبدئي لإصابات النخيل" | A binary infected/healthy model cannot diagnose a specific pest, and "diagnosis" contradicts the screening disclaimer. |
| "دقة عالية" in the header | removed | No model has been trained. There is no accuracy figure to claim. |
| "نسبة الثقة" (confidence) | "الاحتمال المُقدَّر لوجود إصابة" | The displayed number is a calibrated probability of infection, not confidence. |
| implicit 0.5 boundary | threshold drawn on the bar + stated in text | The operating threshold is chosen during training and is usually not 0.5. |
| no disclaimer until a result appeared | standing banner above the upload card | A disclaimer that only appears after a result is a disclaimer most users never read. |
| "TensorFlow 2.x" in the footer | PyTorch, plus a link to the evaluation report | The served model is PyTorch. |

The original file is preserved at `web/app.py.original` for comparison.

`tests/test_frontend_integration.py` checks that every id `app.js` looks up exists in
`index.html`, that the JS still posts to the endpoint the server exposes, and that the removed
claims do not come back. There is no build step, so a renamed id would otherwise fail silently
in the browser.

### The `/predict` contract

The original three fields still work, so nothing that consumed the old API breaks:

```json
{
  "class_name": "Infected",
  "confidence": 83.0,
  "uncertain": false,

  "probability_infected": 0.83,
  "probability_infected_percent": 83.0,
  "threshold": 0.42,
  "threshold_percent": 42.0,
  "needs_review": false,
  "label": "infected",
  "model_version": "0.1.0+efficientnet_b0.20250907120000",
  "disclaimer": "Screening aid only. This is not an agricultural diagnosis. ..."
}
```

`confidence` is retained for backward compatibility and is **not** what the UI renders. It is the
probability of the *predicted* class, which reads as reassuring exactly when the model is least
useful: a borderline 0.51 becomes "51% confident" for whichever side it happened to land on. The
UI shows `probability_infected` against the drawn threshold instead.

Errors populate both `detail` (which the page reads) and `error`/`message`/`message_ar` (which
the JSON API reads), so neither client needs a special case.

### Hosting the frontend separately

If you later move the frontend to its own origin, set `PALMSCREEN_CORS_ORIGINS` to that origin
and point it at `/api/v1/predict`. `frontend-integration/palmApi.js` is a dependency-free client
for that case, and `frontend-integration/demo-upload.html` is a minimal reference uploader.

## 9. Deploy

```bash
docker compose -f docker/docker-compose.yml up --build
```

The image contains the API and the frontend, so one container serves the whole site. CPU-only by
design: MPS is a local Mac accelerator and does not exist in a Linux container, so the image
installs CPU torch wheels. The bundle is mounted read-only rather than baked in, so
the image versions independently from the model.

Because the frontend is served from the same origin, no CORS configuration is needed for the
default setup. Put TLS in front of it. A single
uvicorn worker handles a screening tool's load comfortably; scale horizontally if needed, since
the container is stateless.

---

## Layout

```
configs/            YAML configs (effnet_b0, mobilenet_v3) — every knob, commented
src/palmscreen/
  preprocess.py     the one preprocessing path, shared by training and serving
  catalog.py        image discovery, metadata join, grouping strategy
  hashing.py        SHA-256, dHash, pHash, near-duplicate clustering
  splits.py         grouped splitting, leakage assertions, versioned manifest
  audit.py          data-quality and shortcut detection
  models.py         EfficientNet-B0 / MobileNetV3, freeze-unfreeze control
  data.py           datasets and transforms (augmentation on the train path only)
  train.py          two-phase training, early stopping, resumable checkpoints
  crossval.py       grouped CV, candidate comparison, final fit
  decision.py       calibration and threshold policy
  metrics.py        metrics, Wilson intervals, group bootstrap, prevalence adjustment
  evaluate.py       one-time locked test evaluation + ledger
  plots.py          report charts
  report.py         HTML/JSON/CSV/PDF generation
  gradcam.py        optional qualitative debugging
  bundle.py         versioned inference bundle
  cli.py            command line
api/                FastAPI service (also serves the frontend)
web/
  templates/        index.html (Arabic RTL UI)
  static/css/       style.css
  static/js/        app.js
  app.py.original   the original TensorFlow backend, kept for reference
tests/              97 tests
docker/             CPU inference image
frontend-integration/
```

## The rules this pipeline enforces in code

1. No metric is ever fabricated. Without a test evaluation the report says so explicitly.
2. Training, validation and cross-validation numbers are never called real-world performance.
3. Photos of the same tree/site/session never cross a split — asserted, and it raises.
4. Splitting happens before augmentation; validation and test images are never augmented.
5. The test set is used once, after everything is frozen, enforced by a ledger.

---

## Model card

See `MODEL_CARD.md`. It ships with performance fields marked *not yet measured* and is filled in
from `report.json` after evaluation.
# Redweevilproject
