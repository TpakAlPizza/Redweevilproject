"""palmscreen command line.

    palmscreen audit      --config configs/effnet_b0.yaml
    palmscreen split      --config configs/effnet_b0.yaml
    palmscreen probe      --config configs/effnet_b0.yaml
    palmscreen crossval   --config configs/effnet_b0.yaml
    palmscreen crossval   --config configs/mobilenet_v3.yaml
    palmscreen select     --cv artifacts/checkpoints/cv_a/cv_summary.json --cv .../cv_b/cv_summary.json
    palmscreen finalize   --config configs/effnet_b0.yaml --cv .../cv_summary.json
    palmscreen evaluate   --config configs/effnet_b0.yaml --bundle artifacts/bundles/<v>
    palmscreen report     --config configs/effnet_b0.yaml
    palmscreen gradcam    --bundle artifacts/bundles/<v> --images a.jpg b.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config


def _load_json(path: Optional[str]) -> Optional[dict]:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None


def cmd_audit(args) -> int:
    from .audit import run_audit

    run_audit(Config.load(args.config), args.out)
    return 0


def cmd_split(args) -> int:
    from .catalog import build_catalog
    from .splits import assert_no_leakage, make_folds, make_test_split, write_manifest

    cfg = Config.load(args.config)
    df, resolution = build_catalog(cfg)
    df = df[df["readable"] == True].reset_index(drop=True)  # noqa: E712
    print(f"{len(df)} readable images | grouping: {resolution.strategy} "
          f"({resolution.n_groups} groups, trust={resolution.trust})")
    for warning in resolution.warnings:
        print("  ! " + warning)

    df, split_info = make_test_split(df, cfg)
    dev = make_folds(df[df["split"] == "dev"], cfg)
    test = df[df["split"] == "test"].copy()
    test["fold"] = -1
    import pandas as pd

    combined = pd.concat([dev, test], ignore_index=True)
    leakage = assert_no_leakage(combined, cfg.data.phash_group_distance)
    path = write_manifest(combined, cfg, resolution.as_dict(), split_info, leakage)

    print(f"\nTest set: {split_info['n_test']} images / {split_info['n_test_groups']} groups "
          f"({split_info['n_test_infected']} infected)")
    print(f"Dev set:  {split_info['n_dev']} images / {split_info['n_dev_groups']} groups "
          f"({split_info['n_dev_infected']} infected)")
    print(f"Leakage audit: {'clean' if leakage.clean else 'VIOLATIONS'}")
    print(f"Manifest: {path}")
    if resolution.trust != "strong":
        print("\n*** Results from this split must be labelled POTENTIALLY OPTIMISTIC "
              "INTERNAL VALIDATION. Provide tree/site/session metadata to fix this. ***")
    return 0


def cmd_probe(args) -> int:
    from .probe import run_probes

    run_probes(Config.load(args.config), args.manifest_dir, args.out)
    return 0


def cmd_crossval(args) -> int:
    from .crossval import run_cross_validation

    cfg = Config.load(args.config)
    if args.arch:
        cfg.train.arch = args.arch
    run_cross_validation(cfg, args.manifest_dir, args.out, resume=args.resume)
    return 0


def cmd_select(args) -> int:
    from .crossval import compare_candidates

    summaries = [json.loads(Path(p).read_text()) for p in args.cv]
    result = compare_candidates(summaries, args.metric, args.bootstrap, args.seed)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


def cmd_finalize(args) -> int:
    from .crossval import finalize

    cfg = Config.load(args.config)
    cv = json.loads(Path(args.cv).read_text())
    if cv.get("arch") and cv["arch"] != cfg.train.arch:
        print(f"Note: using arch '{cv['arch']}' from the CV summary.")
        cfg.train.arch = cv["arch"]
    finalize(cfg, cv, args.bundles_root, args.manifest_dir)
    return 0


def cmd_evaluate(args) -> int:
    from .bundle import resolve_latest
    from .evaluate import TestSetLockError, evaluate_locked_test

    cfg = Config.load(args.config)
    bundle_dir = args.bundle or resolve_latest(args.bundles_root)
    if not bundle_dir:
        print("No bundle found. Run 'palmscreen finalize' first.", file=sys.stderr)
        return 2
    try:
        evaluate_locked_test(
            cfg, bundle_dir, args.manifest_dir, allow_repeat=args.allow_repeat, out_dir=args.out
        )
    except TestSetLockError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 3
    return 0


def cmd_report(args) -> int:
    from .bundle import InferenceBundle, resolve_latest
    from .report import build_report
    from .splits import load_manifest

    cfg = Config.load(args.config)
    manifest = None
    try:
        manifest, _ = load_manifest(args.manifest_dir or cfg.split.manifest_dir)
    except FileNotFoundError:
        print("No split manifest found; the report will say so.")

    audit = _load_json(args.audit) or _latest_json(Path(cfg.report.output_dir) / "audit", "audit.json")
    cv = _load_json(args.cv) or _latest_json(Path(cfg.train.checkpoint_dir), "cv_summary.json")
    test = _load_json(args.test) or _latest_json(Path(cfg.report.output_dir) / "test", "test_metrics.json")

    bundle_info = None
    bundle_dir = args.bundle or resolve_latest(args.bundles_root)
    if bundle_dir and Path(bundle_dir, "bundle.json").exists():
        bundle_info = InferenceBundle.load(bundle_dir, verify_hash=False).info()

    paths = build_report(cfg, args.out, manifest, audit, cv, _load_json(args.comparison), test, bundle_info)
    print(json.dumps(paths, indent=2))
    return 0


def _latest_json(root: Path, filename: str) -> Optional[dict]:
    if not root.exists():
        return None
    matches = sorted(root.rglob(filename))
    return json.loads(matches[-1].read_text()) if matches else None


def cmd_gradcam(args) -> int:
    from .bundle import resolve_latest
    from .gradcam import explain_images

    bundle_dir = args.bundle or resolve_latest(args.bundles_root)
    if not bundle_dir:
        print("No bundle found.", file=sys.stderr)
        return 2
    written = explain_images(bundle_dir, args.images, args.out)
    print("\n".join(written))
    print("\nReminder: Grad-CAM is a debugging aid, not evidence of correctness.")
    return 0


def cmd_predict(args) -> int:
    from .bundle import InferenceBundle, resolve_latest

    bundle_dir = args.bundle or resolve_latest(args.bundles_root)
    if not bundle_dir:
        print("No bundle found.", file=sys.stderr)
        return 2
    bundle = InferenceBundle.load(bundle_dir)
    for image in args.images:
        print(json.dumps({"image": image, **bundle.predict(image)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="palmscreen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", default="configs/effnet_b0.yaml")
        return p

    p = with_config(sub.add_parser("audit", help="inspect dataset quality and leakage risk"))
    p.add_argument("--out")
    p.set_defaults(func=cmd_audit)

    p = with_config(sub.add_parser("split", help="build the locked test split and dev folds"))
    p.set_defaults(func=cmd_split)

    p = with_config(sub.add_parser("probe", help="check whether trivial baselines already solve the task"))
    p.add_argument("--manifest-dir")
    p.add_argument("--out")
    p.set_defaults(func=cmd_probe)

    p = with_config(sub.add_parser("crossval", help="grouped K-fold on development data"))
    p.add_argument("--arch", choices=["efficientnet_b0", "mobilenet_v3_large"])
    p.add_argument("--manifest-dir")
    p.add_argument("--out")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_crossval)

    p = sub.add_parser("select", help="compare candidates on development OOF predictions")
    p.add_argument("--cv", action="append", required=True, help="path to a cv_summary.json (repeatable)")
    p.add_argument("--metric", default="pr_auc", choices=["pr_auc", "roc_auc"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out")
    p.set_defaults(func=cmd_select)

    p = with_config(sub.add_parser("finalize", help="refit on dev, fix calibration+threshold, write bundle"))
    p.add_argument("--cv", required=True)
    p.add_argument("--bundles-root", default="artifacts/bundles")
    p.add_argument("--manifest-dir")
    p.set_defaults(func=cmd_finalize)

    p = with_config(sub.add_parser("evaluate", help="ONE-TIME locked test evaluation"))
    p.add_argument("--bundle")
    p.add_argument("--bundles-root", default="artifacts/bundles")
    p.add_argument("--manifest-dir")
    p.add_argument("--out")
    p.add_argument("--allow-repeat", action="store_true",
                   help="re-evaluate weights already scored on the test set (recorded in the report)")
    p.set_defaults(func=cmd_evaluate)

    p = with_config(sub.add_parser("report", help="build the HTML/JSON/CSV/PDF report"))
    p.add_argument("--audit"); p.add_argument("--cv"); p.add_argument("--test")
    p.add_argument("--comparison"); p.add_argument("--bundle")
    p.add_argument("--bundles-root", default="artifacts/bundles")
    p.add_argument("--manifest-dir"); p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("gradcam", help="optional qualitative debugging heat maps")
    p.add_argument("--bundle"); p.add_argument("--bundles-root", default="artifacts/bundles")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", default="reports/gradcam")
    p.set_defaults(func=cmd_gradcam)

    p = sub.add_parser("predict", help="predict on local files using the bundle")
    p.add_argument("--bundle"); p.add_argument("--bundles-root", default="artifacts/bundles")
    p.add_argument("--images", nargs="+", required=True)
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
