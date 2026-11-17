#!/usr/bin/env bash
# Full pipeline in one go, with a stop at the point where the answer is decided.
#
#   ./scripts/run_all.sh /path/to/roboflow-export     # collect, then run
#   ./scripts/run_all.sh                              # data already in data/raw/
#   FAST=1 ./scripts/run_all.sh                       # 3 folds, fewer epochs
#
# Stops after the shortcut probe and asks before spending an hour on training,
# because if the probe fires, the training result will not mean anything.

set -euo pipefail

CONFIG="${CONFIG:-configs/effnet_b0.yaml}"
SOURCE="${1:-}"
export PYTHONPATH="src:${PYTHONPATH:-}"

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$1"; }

if [[ ! -f "$CONFIG" ]]; then
  echo "Run this from the project root (where $CONFIG lives)." >&2
  exit 1
fi

# ---------------------------------------------------------------- 1. collect
if [[ -n "$SOURCE" ]]; then
  step "1/8  Collecting images from $SOURCE"
  python3 scripts/collect_dataset.py --source "$SOURCE"
else
  step "1/8  Using images already in data/raw/"
fi

for class in infected healthy; do
  count=$(find "data/raw/$class" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) 2>/dev/null | wc -l | tr -d ' ')
  echo "    $class: $count images"
  if [[ "$count" -eq 0 ]]; then
    warn "data/raw/$class is empty. Binary classification needs both classes."
    echo "    Find your images:  find ~ -type d -name '*healthy*' 2>/dev/null | head"
    exit 1
  fi
done

# --------------------------------------------------------------- 2. metadata
step "2/8  Removing duplicates and recovering grouping metadata"
# dedupe_and_group.py supersedes build_metadata_from_filenames.py: it does the
# same filename grouping, plus removes exact duplicates and merges images that
# are visually identical under different filenames.
python3 scripts/dedupe_and_group.py

# ------------------------------------------------------------------ 3. audit
step "3/8  Auditing data quality and shortcut risk"
python3 -m palmscreen.cli audit --config "$CONFIG"

# ------------------------------------------------------------------ 4. split
step "4/8  Building the locked test set and grouped folds"
python3 -m palmscreen.cli split --config "$CONFIG"

# ------------------------------------------------------------------ 5. probe
step "5/8  Probing for shortcuts  <-- this decides whether the rest means anything"
python3 -m palmscreen.cli probe --config "$CONFIG"

cat <<'EOF'

--------------------------------------------------------------------------
How to read the probe table above:

  any probe PR-AUC >= 0.90   The classes are separable without disease detail.
                             A high model score proves nothing. Training is
                             still worth running -- your report documents the
                             finding -- but do not claim the model detects
                             infection.

  0.75 - 0.90                Part of the score is acquisition, not disease.

  below 0.70                 Good. Trivial baselines do not solve it.

  negative control ~0.50     Required. Anything else means the folds are broken.
--------------------------------------------------------------------------
EOF

read -r -p "Continue to training? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || { echo "Stopped. Artifacts so far are in reports/ and artifacts/splits/."; exit 0; }

# --------------------------------------------------------------- 6. training
if [[ "${FAST:-0}" == "1" ]]; then
  warn "FAST mode: 3 folds, 12 fine-tune epochs. Fine for a deadline, note it in the report."
  python3 - <<'PY'
import yaml, pathlib, os
path = pathlib.Path(os.environ.get("CONFIG", "configs/effnet_b0.yaml"))
cfg = yaml.safe_load(path.read_text())
cfg["split"]["n_folds"] = 3
cfg["train"]["finetune_epochs"] = 12
cfg["train"]["head_epochs"] = 4
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"{path} adjusted for a fast run")
PY
  python3 -m palmscreen.cli split --config "$CONFIG"
fi

step "6/8  Cross-validation (the long step -- roughly 20-60 minutes)"
python3 -m palmscreen.cli crossval --config "$CONFIG"

CV_SUMMARY=$(ls -t artifacts/checkpoints/cv_*/cv_summary.json 2>/dev/null | head -1)
[[ -n "$CV_SUMMARY" ]] || { echo "No cv_summary.json produced." >&2; exit 1; }

# --------------------------------------------------------------- 7. finalize
step "7/8  Final fit, calibration and threshold"
python3 -m palmscreen.cli finalize --config "$CONFIG" --cv "$CV_SUMMARY"

step "7/8  Locked test evaluation (runs once)"
python3 -m palmscreen.cli evaluate --config "$CONFIG"

# ----------------------------------------------------------------- 8. report
step "8/8  Building the report"
python3 -m palmscreen.cli report --config "$CONFIG"

REPORT=$(ls -td reports/report_*/ 2>/dev/null | head -1)
cat <<EOF

Done.

  Report      ${REPORT}report.html
  PDF         ${REPORT}report.pdf
  Model card  MODEL_CARD.md   <-- fill in the performance table from report.json

  Start the website:   uvicorn api.main:app --port 8000
                       open http://localhost:8000

Read the limitations section at the top of the report before quoting any number.
EOF
