"""Versioned reporting in HTML, JSON, CSV and PDF.

Design rules:
  * development (cross-validation) results and locked-test results never share
    a section, a table, or a colour;
  * no metric is shown without an interval or an explicit sample size;
  * if a stage has not been run, the report says so instead of leaving a gap;
  * the limitations section is at the top as well as the bottom, because
    limitations placed only at the end do not get read.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .runtime import provenance

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
line-height:1.55;margin:0;background:#f6f7f9;color:#1c1e21}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;
border-bottom:2px solid #d8dbe0} h3{font-size:15px;margin:20px 0 8px}
.sub{color:#5c6470;font-size:13px;margin-bottom:18px}
.banner{border-radius:8px;padding:14px 16px;margin:14px 0;font-size:14px}
.banner.red{background:#fdeaea;border:1px solid #e9a5a5}
.banner.amber{background:#fff6e5;border:1px solid #eccb8a}
.banner.blue{background:#eaf2fd;border:1px solid #a9c6ee}
.banner.grey{background:#eef0f2;border:1px solid #cfd4da}
.card{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:16px 18px;margin:12px 0}
.card.test{border-left:5px solid #2f6f4f} .card.dev{border-left:5px solid #7a6ea8}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #e1e4e8;padding:6px 9px;text-align:left}
th{background:#f0f2f5;font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.metric-grid{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
.metric{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:10px 14px;min-width:150px}
.metric .v{font-size:22px;font-weight:600} .metric .k{font-size:11px;color:#5c6470;text-transform:uppercase}
.metric .ci{font-size:11px;color:#5c6470}
img.chart{max-width:100%;border:1px solid #e1e4e8;border-radius:6px;background:#fff;margin:6px 0}
.charts{display:flex;flex-wrap:wrap;gap:12px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
pre{background:#1e2228;color:#e6e6e6;padding:12px;border-radius:6px;overflow:auto;max-height:340px}
details{margin:8px 0} summary{cursor:pointer;font-weight:600;font-size:13px}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;background:#e6e9ed;margin-right:6px}
.tag.dev{background:#e7e3f5} .tag.test{background:#dff0e6} .tag.warn{background:#fbe3e3}
ul.plain li{margin:5px 0}
"""


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return "n/a" if v != v else f"{v:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{100 * v:.1f}%"


def _metric_card(key: str, value: Any, ci: Optional[dict] = None, as_pct: bool = True) -> str:
    shown = _pct(value) if as_pct else _f(value)
    ci_text = ""
    if ci and ci.get("lower") == ci.get("lower"):
        lo = _pct(ci["lower"]) if as_pct else _f(ci["lower"])
        hi = _pct(ci["upper"]) if as_pct else _f(ci["upper"])
        ci_text = f"<div class='ci'>95% CI {lo} - {hi}<br><span style='font-size:10px'>{ci.get('method','')}</span></div>"
    return f"<div class='metric'><div class='k'>{html.escape(key)}</div><div class='v'>{shown}</div>{ci_text}</div>"


def _table(rows: List[dict], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "<p><em>none</em></p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = ""
    for row in rows:
        cells = ""
        for c in columns:
            v = row.get(c, "")
            numeric = isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
            cells += f"<td class='{'num' if numeric else ''}'>{html.escape(_f(v) if numeric else str(v))}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _img(b64: Optional[str], alt: str) -> str:
    return f"<img class='chart' alt='{html.escape(alt)}' src='data:image/png;base64,{b64}'>" if b64 else ""


# --------------------------------------------------------------------------
# Limitations text (generated from the actual numbers, never hard-coded)
# --------------------------------------------------------------------------

def limitations_text(
    n_test: Optional[int],
    n_test_groups: Optional[int],
    grouping: dict,
    sensitivity_ci: Optional[dict],
    test_strategy: str,
) -> List[str]:
    items: List[str] = []
    if n_test:
        width = ""
        if sensitivity_ci and sensitivity_ci.get("lower") == sensitivity_ci.get("lower"):
            width = (
                f" On this test set the 95% interval for infected sensitivity runs from "
                f"{_pct(sensitivity_ci['lower'])} to {_pct(sensitivity_ci['upper'])} - "
                "that range is the honest answer, not the single number above it."
            )
        items.append(
            f"The final test set contains {n_test} images"
            + (f" from {n_test_groups} independent groups." if n_test_groups else ".")
            + f" A test set this small gives very wide uncertainty.{width} "
            "It cannot prove real field performance; it can only rule out a model that is obviously broken."
        )
    else:
        items.append(
            "No locked test evaluation has been run yet, so this report contains "
            "NO measured performance figures. Any number quoted from it as "
            "'model accuracy' would be fabricated."
        )

    trust = grouping.get("trust", "unknown")
    if trust == "strong":
        items.append(
            f"Splits were grouped by '{grouping.get('column')}', so no palm appears in both "
            "training and test data. This removes the most common source of inflated results, "
            "but does not remove the others below."
        )
    elif trust == "weak":
        items.append(
            "NO real grouping metadata was available. Near-duplicate clustering was used as a "
            "substitute. Two different photographs of the same palm can still have landed on "
            "opposite sides of the split, which inflates every figure in this report by an "
            "unknown amount. Treat all results as POTENTIALLY OPTIMISTIC INTERNAL VALIDATION."
        )
    else:
        items.append(
            "An IMAGE-LEVEL split was used because no grouping metadata exists. Photos of the same "
            "palm are almost certainly split across train and test. These numbers are an upper "
            "bound on performance and must not be quoted as model performance."
        )

    items += [
        "Training accuracy, validation accuracy, and cross-validation scores in this report were "
        "used to build the model. They are optimistic by construction and are not performance estimates.",
        f"Test-set construction: {test_strategy}. Unless the test images come from a farm, season, "
        "camera and photographer never seen in training, this measures internal consistency, not "
        "generalisation to a new grove.",
        "Precision (PPV) and negative predictive value depend on how common infection actually is. "
        "This dataset is roughly balanced; a real grove is not. See the prevalence table: at 5% "
        "field prevalence the same model produces far more false alarms per true detection.",
        "The model sees a whole image resized to a small square. If symptoms occupy a few dozen "
        "pixels, the model may be responding to the background, the lighting, or the camera rather "
        "than the disease.",
        "This is a screening aid. It is not an agricultural diagnosis and must not be the sole basis "
        "for removing, treating, or quarantining a tree.",
        "Performance will drift as cameras, seasons, cultivars and disease stages change. Re-audit "
        "and re-evaluate on freshly collected images before trusting it in a new context.",
    ]
    return items


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def build_report(
    cfg: Config,
    out_dir: Optional[str] = None,
    manifest: Optional[dict] = None,
    audit: Optional[dict] = None,
    cv_summary: Optional[dict] = None,
    comparison: Optional[dict] = None,
    test_result: Optional[dict] = None,
    bundle_info: Optional[dict] = None,
    report_version: Optional[str] = None,
) -> Dict[str, str]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    version = report_version or f"{cfg.model_version}_{stamp}"
    out = Path(out_dir or Path(cfg.report.output_dir) / f"report_{version}")
    (out / "charts").mkdir(parents=True, exist_ok=True)

    grouping = (manifest or {}).get("grouping", (audit or {}).get("grouping", {"trust": "unknown"}))
    test_strategy = str((manifest or {}).get("split_info", {}).get("strategy", "unknown"))
    charts: Dict[str, str] = {}

    # ---- development artefacts ----------------------------------------
    oof_frame = None
    if cv_summary and Path(str(cv_summary.get("oof_csv", ""))).exists():
        oof_frame = pd.read_csv(cv_summary["oof_csv"])
        probs = oof_frame.get("oof_prob_calibrated", oof_frame["oof_prob_raw"]).values
        y = oof_frame["label_binary"].values
        charts["dev_roc"] = plots.roc_plot(y, probs, out / "charts/dev_roc.png", "ROC - development OOF")
        charts["dev_pr"] = plots.pr_plot(y, probs, out / "charts/dev_pr.png", "PR - development OOF")
        threshold = float((bundle_info or {}).get("threshold", {}).get("threshold", 0.5))
        charts["dev_threshold"] = plots.threshold_sweep(y, probs, threshold, out / "charts/dev_threshold.png")
        charts["dev_hist"] = plots.score_histogram(y, probs, threshold, out / "charts/dev_hist.png")
        histories = {f"fold{f['fold']}": f["history"] for f in cv_summary.get("folds", [])}
        if histories:
            charts["learning"] = plots.learning_curves(histories, out / "charts/learning_curves.png")

    # ---- locked test artefacts ----------------------------------------
    if test_result:
        preds = pd.read_csv(test_result["predictions_csv"])
        y, p = preds["label_binary"].values, preds["probability_infected"].values
        charts["test_roc"] = plots.roc_plot(y, p, out / "charts/test_roc.png", "ROC - locked test")
        charts["test_pr"] = plots.pr_plot(y, p, out / "charts/test_pr.png", "PR - locked test")
        charts["test_cal"] = plots.calibration_plot(
            test_result["calibration_bins"], out / "charts/test_calibration.png", "Reliability - locked test"
        )
        charts["test_cm"] = plots.confusion_plot(
            test_result["metrics"], out / "charts/test_confusion.png", "Locked test"
        )
        charts["test_hist"] = plots.score_histogram(
            y, p, test_result["threshold"], out / "charts/test_hist.png"
        )
        fn_chart = plots.error_grid(
            test_result.get("false_negatives", []), out / "charts/false_negatives.png",
            "Missed infected palms (false negatives)",
        )
        fp_chart = plots.error_grid(
            test_result.get("false_positives", []), out / "charts/false_positives.png",
            "Healthy palms flagged as infected (false positives)",
        )
        if fn_chart:
            charts["false_negatives"] = fn_chart
        if fp_chart:
            charts["false_positives"] = fp_chart

    payload = {
        "report_version": version,
        "generated_at": stamp,
        "project": cfg.project_name,
        "model_version": (bundle_info or {}).get("model_version", cfg.model_version),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "provenance": provenance(),
        "grouping": grouping,
        "split_manifest": manifest,
        "audit": audit,
        "development_cross_validation": cv_summary,
        "model_comparison": comparison,
        "locked_test_evaluation": test_result,
        "bundle": bundle_info,
        "limitations": limitations_text(
            (test_result or {}).get("n_test_images"),
            (test_result or {}).get("n_test_groups"),
            grouping,
            (test_result or {}).get("bootstrap_intervals", {}).get("sensitivity_infected"),
            test_strategy,
        ),
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    _write_csvs(out, cv_summary, test_result)
    html_path = out / "report.html"
    html_path.write_text(_render_html(payload, charts, cfg))
    pdf_path = _write_pdf(out / "report.pdf", payload, out / "charts")

    latest = Path(cfg.report.output_dir) / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps({"report_dir": out.name, "report_version": version,
                                  "generated_at": stamp}, indent=2))
    print(f"Report written: {out}")
    return {
        "dir": str(out), "html": str(html_path), "json": str(out / "report.json"),
        "pdf": str(pdf_path) if pdf_path else "", "version": version,
    }


def _write_csvs(out: Path, cv_summary: Optional[dict], test_result: Optional[dict]) -> None:
    if cv_summary:
        rows = []
        for fold in cv_summary.get("folds", []):
            row = {"fold": fold["fold"], "n_val": fold["n_val"], "n_val_groups": fold["n_val_groups"],
                   "best_epoch": fold["best_epoch"]}
            row.update({k: v for k, v in fold["metrics_at_0.5"].items() if isinstance(v, (int, float))})
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out / "cv_per_fold_metrics.csv", index=False)
        if Path(str(cv_summary.get("oof_csv", ""))).exists():
            pd.read_csv(cv_summary["oof_csv"]).to_csv(out / "dev_out_of_fold_predictions.csv", index=False)
    if test_result:
        metrics = {k: v for k, v in test_result["metrics"].items() if isinstance(v, (int, float))}
        pd.DataFrame([metrics]).to_csv(out / "test_metrics.csv", index=False)
        pd.DataFrame(test_result["prevalence_sensitivity"]).to_csv(out / "prevalence_sensitivity.csv", index=False)
        if Path(test_result["predictions_csv"]).exists():
            pd.read_csv(test_result["predictions_csv"]).to_csv(out / "test_predictions.csv", index=False)


def _render_html(payload: dict, charts: Dict[str, str], cfg: Config) -> str:
    test = payload.get("locked_test_evaluation")
    cv = payload.get("development_cross_validation")
    audit = payload.get("audit") or {}
    manifest = payload.get("split_manifest") or {}
    grouping = payload.get("grouping", {})
    parts: List[str] = []

    parts.append(
        f"<!doctype html><meta charset='utf-8'><title>Palm screening report {html.escape(payload['report_version'])}</title>"
        f"<style>{CSS}</style><div class='wrap'>"
        f"<h1>Palm infection screening - evaluation report</h1>"
        f"<div class='sub'>report {html.escape(payload['report_version'])} &middot; model "
        f"{html.escape(str(payload['model_version']))} &middot; generated {html.escape(payload['generated_at'])} "
        f"&middot; config {html.escape(payload['config_fingerprint'])}</div>"
    )

    # Banners first: the reader should not have to scroll to find the caveats.
    parts.append(
        "<div class='banner blue'><b>How to read this report.</b> Sections marked "
        "<span class='tag dev'>DEVELOPMENT</span> were used to build the model and are optimistic by "
        "construction. Only the section marked <span class='tag test'>LOCKED TEST</span> is an "
        "evaluation, and it is small. This tool is a screening aid, not a diagnosis.</div>"
    )
    if not test:
        parts.append(
            "<div class='banner red'><b>No locked-test evaluation exists in this report.</b> "
            "The pipeline has not produced measured performance. Do not quote any number here as "
            "model accuracy.</div>"
        )
    trust = grouping.get("trust")
    if trust == "none":
        parts.append(
            "<div class='banner red'><b>IMAGE-LEVEL SPLIT.</b> No tree/site/session metadata was "
            "available, so photographs of the same palm are very likely in both training and test "
            "data. Every figure below is an upper bound, not an estimate.</div>"
        )
    elif trust == "weak":
        parts.append(
            "<div class='banner amber'><b>PSEUDO-GROUPED SPLIT.</b> Grouping was inferred from "
            "near-duplicate clustering, not from real metadata. Different photos of the same palm "
            "may still straddle the split. Results are potentially optimistic internal validation.</div>"
        )
    if test and test.get("is_repeat_evaluation"):
        parts.append(
            f"<div class='banner red'><b>REPEATED TEST EVALUATION.</b> These weights were evaluated on "
            f"the locked test set {test['n_previous_evaluations']} time(s) before. The test set no longer "
            "provides an unbiased estimate.</div>"
        )
    for warning in (audit.get("warnings") or [])[:8]:
        parts.append(f"<div class='banner amber'>{html.escape(str(warning))}</div>")

    # ---- limitations up front ----
    parts.append("<h2>Limitations - read before the numbers</h2><div class='card'><ul class='plain'>")
    parts += [f"<li>{html.escape(item)}</li>" for item in payload["limitations"]]
    parts.append("</ul></div>")

    # ---- dataset & splits ----
    parts.append("<h2>1. Dataset and splits</h2>")
    counts = manifest.get("counts", {})
    rows = []
    for split in ("dev", "test"):
        entry = counts.get(split)
        if not entry:
            continue
        by_class = entry.get("by_class", {})
        rows.append(
            {
                "split": split, "images": entry.get("n_images"),
                "independent groups": entry.get("n_groups"),
                "infected": by_class.get("infected", 0), "healthy": by_class.get("healthy", 0),
                "groups (infected)": entry.get("groups_by_class", {}).get("infected", 0),
                "groups (healthy)": entry.get("groups_by_class", {}).get("healthy", 0),
            }
        )
    parts.append(f"<div class='card'>{_table(rows)}")
    parts.append(
        f"<p><b>Grouping strategy:</b> {html.escape(str(grouping.get('strategy')))} "
        f"(column: {html.escape(str(grouping.get('column')))}, trust: <span class='tag "
        f"{'warn' if grouping.get('trust') != 'strong' else ''}'>{html.escape(str(grouping.get('trust')))}</span>, "
        f"{grouping.get('n_groups', 'n/a')} groups)</p>"
        f"<p><b>Test-set construction:</b> {html.escape(str(manifest.get('split_info', {}).get('strategy', 'n/a')))}. "
        f"{html.escape(str(manifest.get('split_info', {}).get('note', '')))}</p>"
    )
    folds = counts.get("dev", {}).get("folds")
    if folds:
        parts.append("<h3>Development folds</h3>" + _table(
            [{"fold": k, **v} for k, v in folds.items()]
        ))
    parts.append("</div>")

    # ---- leakage audit ----
    parts.append("<h2>2. Duplicate and leakage audit</h2><div class='card'>")
    leak = manifest.get("leakage_audit", {})
    if leak:
        status = "clean" if leak.get("clean") else "VIOLATIONS FOUND"
        colour = "blue" if leak.get("clean") else "red"
        parts.append(
            f"<div class='banner {colour}'>Split independence check: <b>{status}</b>. "
            f"group overlap: {len(leak.get('group_overlap', []))}, identical files across splits: "
            f"{len(leak.get('sha256_overlap', []))}, near-duplicate pairs across splits: "
            f"{len(leak.get('near_duplicate_pairs_across_splits', []))}, fold group overlap: "
            f"{len(leak.get('fold_group_overlap', []))}.</div>"
        )
    dup = (audit.get("duplicates") or {})
    if dup:
        parts.append(_table([{
            "exact duplicate groups": dup.get("n_exact_groups", 0),
            "near-duplicate clusters": dup.get("n_near_duplicate_clusters", 0),
            "cross-class near duplicates": len(dup.get("cross_class_near_duplicates", [])),
            "conflicting labels on identical files": len(dup.get("conflicting_labels", [])),
        }]))
    shortcuts = (audit.get("shortcut_signals") or {}).get("class_correlated_metadata") or []
    if shortcuts:
        parts.append("<h3>Possible shortcuts</h3>" + _table(
            [{"field": s["field"], "single-class values": s["n_single_class_values"], "message": s["message"]}
             for s in shortcuts]
        ))
    if audit.get("manual_review_checklist"):
        parts.append("<details><summary>Manual review checklist (a human must answer these)</summary><ul class='plain'>"
                     + "".join(f"<li>{html.escape(q)}</li>" for q in audit["manual_review_checklist"])
                     + "</ul></details>")
    montages = audit.get("montages") or {}
    if montages:
        listing = ", ".join(f"<code>{html.escape(v)}</code>" for v in montages.values())
        parts.append(
            f"<h3>Label inspection montages</h3><p>Contact sheets were written to "
            f"<code>{html.escape(str(audit.get('output_dir', 'reports/audit/&lt;timestamp&gt;')))}</code>: "
            f"{listing}. Open them and check the labels by eye: no metric in this report can "
            "detect a systematically mislabelled class.</p>"
        )
    if audit.get("roi_guidance"):
        parts.append(f"<div class='banner grey'>{html.escape(audit['roi_guidance'])}</div>")
    parts.append("</div>")

    # ---- development section ----
    parts.append("<h2>3. Development cross-validation <span class='tag dev'>DEVELOPMENT - NOT PERFORMANCE</span></h2>")
    if cv:
        stats = cv.get("per_fold_summary", {})
        parts.append(
            "<div class='card dev'><div class='banner amber'>These numbers selected the model, the "
            "threshold and the calibrator. Reporting them as performance would mean grading the "
            "exam with the answer key.</div><div class='metric-grid'>"
            + _metric_card("OOF ROC-AUC", cv["oof_ranking_metrics"]["roc_auc"], None, False)
            + _metric_card("OOF PR-AUC", cv["oof_ranking_metrics"]["pr_auc"], cv.get("oof_pr_auc_ci"), False)
            + _metric_card("dev images", cv["n_dev_images"], None, False)
            + _metric_card("dev groups", cv["n_dev_groups"], None, False)
            + "</div>"
        )
        fold_rows = [{
            "fold": f["fold"], "n_val": f["n_val"], "n_val_groups": f["n_val_groups"],
            "best_epoch": f["best_epoch"], "roc_auc": f["metrics_at_0.5"].get("roc_auc"),
            "pr_auc": f["metrics_at_0.5"].get("pr_auc"),
            "sensitivity@0.5": f["metrics_at_0.5"].get("sensitivity_infected"),
            "specificity@0.5": f["metrics_at_0.5"].get("specificity_healthy"),
            "device": f.get("device"), "seconds": f.get("seconds"),
        } for f in cv.get("folds", [])]
        parts.append("<h3>Per-fold results</h3>" + _table(fold_rows))
        if stats:
            parts.append("<h3>Fold-to-fold spread</h3>" + _table([
                {"metric": k, "mean": v["mean"], "std": v["std"], "min": v["min"], "max": v["max"]}
                for k, v in stats.items() if isinstance(v, dict)
            ]))
        parts.append("<div class='charts'>"
                     + _img(charts.get("dev_roc"), "development ROC")
                     + _img(charts.get("dev_pr"), "development PR")
                     + _img(charts.get("dev_threshold"), "threshold sweep")
                     + _img(charts.get("dev_hist"), "score distribution")
                     + "</div>")
        if charts.get("learning"):
            parts.append("<h3>Learning curves</h3>" + _img(charts["learning"], "learning curves"))
        parts.append("</div>")
    else:
        parts.append("<div class='card dev'><p><em>Cross-validation has not been run.</em></p></div>")

    # ---- model selection ----
    comparison = payload.get("model_comparison")
    if comparison:
        parts.append("<h2>4. Model selection <span class='tag dev'>DEVELOPMENT ONLY</span></h2><div class='card dev'>")
        parts.append(f"<p><b>Chosen:</b> {html.escape(str(comparison.get('decision')))}</p>")
        parts.append(f"<p>{html.escape(str(comparison.get('reason')))}</p>")
        parts.append(_table([{"candidate": k, comparison.get("metric", "metric"): v}
                             for k, v in (comparison.get("point_estimates") or {}).items()]))
        parts.append("</div>")

    # ---- calibration + threshold ----
    bundle = payload.get("bundle") or {}
    if bundle:
        parts.append("<h2>5. Calibration and decision threshold</h2><div class='card dev'>")
        cal, thr = bundle.get("calibration", {}), bundle.get("threshold", {})
        parts.append(_table([{
            "calibration method": cal.get("method"), "fitted on": cal.get("fitted_on"),
            "n calibration points": cal.get("n_fit"), "threshold": thr.get("threshold"),
            "policy": thr.get("policy"), "target recall": thr.get("target_recall"),
            "selected on": thr.get("selected_on"), "review band": thr.get("review_band"),
        }]))
        for note in thr.get("notes", []):
            parts.append(f"<div class='banner grey'>{html.escape(str(note))}</div>")
        parts.append(
            "<div class='banner blue'>Raw network output is not confidence. The reported "
            "probability is the calibrated value produced by the mapping above, fitted on "
            "development out-of-fold predictions only.</div></div>"
        )

    # ---- locked test ----
    parts.append("<h2>6. Locked test set <span class='tag test'>THE ONLY EVALUATION</span></h2>")
    if test:
        m, boot, wil = test["metrics"], test.get("bootstrap_intervals", {}), test.get("wilson_intervals", {})
        parts.append(
            f"<div class='card test'><p>Evaluated once on {test['n_test_images']} images from "
            f"{test['n_test_groups']} independent groups at threshold {_f(test['threshold'])} "
            f"(fixed before this evaluation). Class counts: {html.escape(str(test['test_class_counts']))}.</p>"
            "<div class='metric-grid'>"
            + _metric_card("sensitivity (infected)", m["sensitivity_infected"], wil.get("sensitivity_infected"))
            + _metric_card("false negative rate", m["false_negative_rate"])
            + _metric_card("specificity (healthy)", m["specificity_healthy"], wil.get("specificity_healthy"))
            + _metric_card("precision (infected)", m["precision_infected"], wil.get("precision_infected"))
            + _metric_card("NPV (healthy)", m["npv_healthy"], wil.get("npv_healthy"))
            + _metric_card("balanced accuracy", m["balanced_accuracy"], boot.get("balanced_accuracy"))
            + _metric_card("macro F1", m["macro_f1"], boot.get("macro_f1"))
            + _metric_card("MCC", m["mcc"], boot.get("mcc"), False)
            + _metric_card("ROC-AUC", m["roc_auc"], boot.get("roc_auc"), False)
            + _metric_card("PR-AUC", m["pr_auc"], boot.get("pr_auc"), False)
            + _metric_card("log loss", m["log_loss"], None, False)
            + _metric_card("Brier score", m["brier"], boot.get("brier"), False)
            + _metric_card("ECE", test.get("ece"), None, False)
            + "</div>"
        )
        parts.append(
            "<h3>Confusion matrix (raw counts)</h3>"
            + _table([{"": "actual healthy", "predicted healthy": m["tn"], "predicted infected": m["fp"]},
                      {"": "actual infected", "predicted healthy": m["fn"], "predicted infected": m["tp"]}])
            + f"<p>{m['fn']} infected palm(s) were missed and {m['fp']} healthy palm(s) were flagged. "
            f"{test.get('n_needs_review', 0)} prediction(s) fell inside the review band and would be "
            "routed to a human.</p>"
        )
        parts.append("<div class='charts'>"
                     + _img(charts.get("test_cm"), "confusion matrix")
                     + _img(charts.get("test_roc"), "test ROC")
                     + _img(charts.get("test_pr"), "test PR")
                     + _img(charts.get("test_cal"), "calibration")
                     + _img(charts.get("test_hist"), "score distribution")
                     + "</div>")
        parts.append(
            "<div class='banner amber'>Wilson intervals are used for sensitivity, specificity, "
            "precision and NPV because they behave correctly on small counts. Bootstrap intervals "
            "resample <i>groups</i>, not images, so repeated photos of one palm are not counted as "
            "independent evidence.</div></div>"
        )
        parts.append("<h2>7. What this means at real field prevalence</h2><div class='card test'>"
                     "<p>Sensitivity and specificity do not change with prevalence; precision and NPV do. "
                     "This dataset is roughly balanced, a real grove is not.</p>"
                     + _table([{
                         "field prevalence": _pct(r["prevalence"]),
                         "precision (PPV)": r["ppv_precision_infected"],
                         "NPV": r["npv_healthy"],
                         "false alarms per 100 healthy": r["expected_false_alarms_per_100_healthy"],
                         "missed per 100 infected": r["expected_missed_per_100_infected"],
                     } for r in test["prevalence_sensitivity"]])
                     + "</div>")
        if charts.get("false_negatives") or charts.get("false_positives"):
            parts.append("<h2>8. Errors for human review</h2><div class='card test'>"
                         + _img(charts.get("false_negatives"), "false negatives")
                         + _img(charts.get("false_positives"), "false positives")
                         + "<p>Look for a pattern: if the misses share a background, a camera, a "
                         "distance, or a disease stage, the fix is data collection, not more epochs.</p></div>")
    else:
        parts.append(
            "<div class='card test'><div class='banner red'>Not evaluated. The locked test set has not "
            "been used, so this report contains no performance measurement. Run "
            "<code>palmscreen evaluate</code> once the model, threshold and calibration are final.</div></div>"
        )

    # ---- reproducibility ----
    parts.append("<h2>9. Reproducibility</h2><div class='card'>")
    prov = payload.get("provenance", {})
    parts.append(_table([{
        "seed": manifest.get("seed", cfg.split.seed),
        "split manifest": manifest.get("manifest_version", "n/a"),
        "config fingerprint": payload["config_fingerprint"],
        "model version": payload["model_version"],
        "weights sha256": str((payload.get("bundle") or {}).get("weights_sha256", "see bundle.json"))[:16],
        "device": prov.get("device", {}).get("device"),
        "platform": prov.get("platform"),
        "git commit": prov.get("git_commit") or "not a git checkout",
    }]))
    parts.append("<details><summary>Package versions</summary>"
                 + _table([{"package": k, "version": v} for k, v in (prov.get("packages") or {}).items()])
                 + "</details>")
    parts.append("<details><summary>Full configuration</summary><pre>"
                 + html.escape(json.dumps(payload["config"], indent=2)) + "</pre></details>")
    parts.append("<details><summary>Split manifest (counts and leakage audit)</summary><pre>"
                 + html.escape(json.dumps({k: v for k, v in (manifest or {}).items() if k != "config"},
                                          indent=2, default=str)[:20000]) + "</pre></details>")
    parts.append("</div>")

    parts.append(
        "<h2>10. Grad-CAM</h2><div class='card'><div class='banner amber'>Grad-CAM heat maps, if "
        "generated, are a qualitative debugging aid only. A plausible-looking heat map is not "
        "evidence that the model is right, and a model can attend to the correct region for the "
        "wrong reason. Use them to spot obvious failures (attention on a watermark, a fence, the "
        "sky), never as proof of correctness.</div></div>"
    )

    parts.append(
        "<div class='banner red' style='margin-top:30px'><b>Screening aid, not a diagnosis.</b> "
        "Results must be confirmed by a qualified agronomist or plant-health inspection before any "
        "treatment, removal or quarantine decision.</div></div>"
    )
    return "".join(parts)


def _write_pdf(path: Path, payload: dict, charts_dir: Path) -> Optional[Path]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print("reportlab not installed; skipping PDF")
        return None

    styles = getSampleStyleSheet()
    warn = ParagraphStyle("warn", parent=styles["Normal"], textColor=colors.HexColor("#8a1c1c"),
                          backColor=colors.HexColor("#fdeaea"), borderPadding=6, spaceAfter=8)
    story = [
        Paragraph("Palm infection screening - evaluation report", styles["Title"]),
        Paragraph(
            f"report {payload['report_version']} &middot; model {payload['model_version']} &middot; "
            f"generated {payload['generated_at']} &middot; config {payload['config_fingerprint']}",
            styles["Normal"],
        ),
        Spacer(1, 10),
        Paragraph("Screening aid only - not an agricultural diagnosis.", warn),
        Paragraph("Limitations", styles["Heading2"]),
    ]
    for item in payload["limitations"]:
        story.append(Paragraph("&bull; " + item, styles["Normal"]))
        story.append(Spacer(1, 3))

    test = payload.get("locked_test_evaluation")
    cv = payload.get("development_cross_validation")
    story.append(PageBreak())
    story.append(Paragraph("Development cross-validation (used for selection - optimistic)", styles["Heading2"]))
    if cv:
        rank = cv["oof_ranking_metrics"]
        story.append(Paragraph(
            f"{cv['n_dev_images']} images, {cv['n_dev_groups']} groups, {cv['n_folds']} grouped folds. "
            f"Out-of-fold ROC-AUC {rank['roc_auc']:.3f}, PR-AUC {rank['pr_auc']:.3f}. "
            "These numbers chose the model, threshold and calibrator; they are not performance.",
            styles["Normal"]))
    else:
        story.append(Paragraph("Not run.", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Locked test set (the only evaluation)", styles["Heading2"]))
    if test:
        m = test["metrics"]
        wil = test.get("wilson_intervals", {})

        def ci(name: str) -> str:
            c = wil.get(name)
            return f"{c['lower']:.3f} - {c['upper']:.3f}" if c and c["lower"] == c["lower"] else "n/a"

        rows = [
            ["Metric", "Value", "95% CI"],
            ["Sensitivity (infected)", f"{m['sensitivity_infected']:.3f}", ci("sensitivity_infected")],
            ["False negative rate", f"{m['false_negative_rate']:.3f}", ""],
            ["Specificity (healthy)", f"{m['specificity_healthy']:.3f}", ci("specificity_healthy")],
            ["Precision (infected)", f"{m['precision_infected']:.3f}", ci("precision_infected")],
            ["NPV (healthy)", f"{m['npv_healthy']:.3f}", ci("npv_healthy")],
            ["Balanced accuracy", f"{m['balanced_accuracy']:.3f}", ""],
            ["Macro F1", f"{m['macro_f1']:.3f}", ""],
            ["MCC", f"{m['mcc']:.3f}", ""],
            ["ROC-AUC", f"{m['roc_auc']:.3f}", ""],
            ["PR-AUC", f"{m['pr_auc']:.3f}", ""],
            ["Log loss", f"{m['log_loss']:.3f}", ""],
            ["Brier score", f"{m['brier']:.3f}", ""],
            ["ECE", f"{test.get('ece', float('nan')):.3f}", ""],
        ]
        table = Table(rows, colWidths=[7 * cm, 4 * cm, 5 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f2")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story += [
            Paragraph(
                f"Evaluated once on {test['n_test_images']} images from {test['n_test_groups']} "
                f"independent groups at a threshold of {test['threshold']:.3f} fixed beforehand. "
                f"Confusion matrix: TP {m['tp']}, FP {m['fp']}, TN {m['tn']}, FN {m['fn']}.",
                styles["Normal"]),
            Spacer(1, 6), table, Spacer(1, 10),
            Paragraph(
                f"A test set of {test['n_test_images']} images cannot establish field performance. "
                "The confidence intervals above are the result; the point estimates are not.", warn),
        ]
        for name in ("test_confusion.png", "test_roc.png", "test_pr.png", "test_calibration.png"):
            chart = charts_dir / name
            if chart.exists():
                story += [Spacer(1, 8), RLImage(str(chart), width=11 * cm, height=9 * cm, kind="proportional")]
    else:
        story.append(Paragraph(
            "NOT EVALUATED. This report contains no measured performance. Any figure quoted from it "
            "as model accuracy would be fabricated.", warn))

    SimpleDocTemplate(str(path), pagesize=A4, title=f"Palm screening report {payload['report_version']}").build(story)
    return path


def latest_report(reports_dir: str | Path) -> Optional[dict]:
    """Resolve the newest report for the API's read-only report endpoint."""
    root = Path(reports_dir)
    pointer = root / "latest.json"
    if pointer.exists():
        meta = json.loads(pointer.read_text())
        report_dir = root / meta["report_dir"]
        if (report_dir / "report.json").exists():
            return {"dir": str(report_dir), "meta": meta}
    candidates = sorted(p.parent for p in root.glob("report_*/report.json"))
    return {"dir": str(candidates[-1]), "meta": {}} if candidates else None
