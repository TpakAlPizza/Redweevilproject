"""Two-phase transfer learning with early stopping and resumable checkpoints.

Phase 1  frozen backbone, train the head only  (lr_head)
Phase 2  unfreeze the top N backbone blocks    (lr_backbone, much lower)

Early stopping watches a *validation* metric from the current fold. The locked
test set is never loaded by this module -- it has no code path to it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data import make_loader
from .metrics import ranking_metrics
from .models import (
    build_model,
    count_params,
    freeze_backbone,
    model_state_hash,
    parameter_groups,
    set_frozen_bn_eval,
    unfreeze_top_blocks,
)
from .preprocess import PreprocessSpec
from .runtime import select_device, set_seed

HIGHER_IS_BETTER = {"val_pr_auc": True, "val_roc_auc": True, "val_loss": False}


@dataclass
class TrainResult:
    best_epoch: int
    best_metric: float
    metric_name: str
    history: List[dict]
    checkpoint_path: str
    device: str
    device_reason: str
    n_train: int
    n_val: int
    model_hash: str
    epochs_run: int
    stopped_early: bool
    seconds: float
    val_probs: Optional[np.ndarray] = field(default=None, repr=False)
    val_labels: Optional[np.ndarray] = field(default=None, repr=False)
    val_index: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "epochs_run": self.epochs_run,
            "stopped_early": self.stopped_early,
            "device": self.device,
            "device_reason": self.device_reason,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "model_hash": self.model_hash,
            "seconds": round(self.seconds, 1),
            "history": self.history,
        }


def _forward_epoch(model, loader, device, criterion, optimizer=None, grad_clip=0.0, freeze_bn=True):
    import torch

    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        set_frozen_bn_eval(model)

    losses, probs, labels, indices = [], [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_x, batch_y, batch_idx in loader:
            batch_x = batch_x.to(device, non_blocking=False)
            batch_y = batch_y.to(device, dtype=torch.float32)
            logits = model(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(batch_y))
            probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
            labels.append(batch_y.detach().cpu().numpy())
            indices.append(np.asarray(batch_idx))

    n = sum(len(x) for x in labels)
    return (
        float(sum(losses) / max(n, 1)),
        np.concatenate(probs) if probs else np.array([]),
        np.concatenate(labels) if labels else np.array([]),
        np.concatenate(indices) if indices else np.array([]),
    )


def _is_better(current: float, best: float, metric: str, min_delta: float) -> bool:
    if best != best:  # NaN
        return True
    if HIGHER_IS_BETTER.get(metric, True):
        return current > best + min_delta
    return current < best - min_delta


def _class_weight(train_df: pd.DataFrame, enabled: bool) -> Optional[float]:
    if not enabled:
        return None
    pos = int(train_df["label_binary"].sum())
    neg = int(len(train_df) - pos)
    return float(neg / pos) if pos else None


def train_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    cfg: Config,
    spec: PreprocessSpec,
    run_dir: str | Path,
    tag: str = "fold0",
    max_epochs_override: Optional[int] = None,
    resume: bool = False,
) -> Tuple[object, TrainResult]:
    """Train one model. `val_df=None` disables early stopping (final fit)."""
    import torch
    import torch.nn as nn

    started = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / f"{tag}_best.pt"
    ckpt_last = run_dir / f"{tag}_last.pt"

    set_seed(cfg.train.seed)
    dev_info = select_device()
    device = torch.device(dev_info.device)
    print(f"[{tag}] device={dev_info.device} ({dev_info.reason})")

    model = build_model(cfg.train.arch, pretrained=True).to(device)
    pos_weight = _class_weight(train_df, cfg.train.use_class_weights)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    train_loader = make_loader(
        train_df, spec, cfg.train.batch_size, True, cfg.train.augmentation,
        cfg.train.num_workers, cfg.train.seed,
    )
    val_loader = (
        make_loader(val_df, spec, cfg.train.batch_size, False, None, cfg.train.num_workers)
        if val_df is not None and len(val_df)
        else None
    )

    metric_name = cfg.train.early_stopping_metric if val_loader is not None else "train_loss"
    history: List[dict] = []
    best_metric = float("nan")
    best_epoch = -1
    start_epoch = 0
    stopped_early = False
    patience_left = cfg.train.early_stopping_patience

    total_head = cfg.train.head_epochs
    total_ft = max_epochs_override if max_epochs_override is not None else cfg.train.finetune_epochs
    total_epochs = total_head + total_ft

    freeze_backbone(model, cfg.train.arch)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, cfg.train.lr_head, cfg.train.lr_backbone),
        weight_decay=cfg.train.weight_decay,
    )

    if resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        if state["epoch"] + 1 >= total_head:  # rebuild phase-2 optimizer shape
            unfreeze_top_blocks(model, cfg.train.arch, cfg.train.unfreeze_last_n_blocks)
            optimizer = torch.optim.AdamW(
                parameter_groups(model, cfg.train.lr_head, cfg.train.lr_backbone),
                weight_decay=cfg.train.weight_decay,
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_metric = state.get("best_metric", float("nan"))
        best_epoch = state.get("best_epoch", -1)
        history = state.get("history", [])
        print(f"[{tag}] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, total_epochs):
        if epoch == total_head:
            info = unfreeze_top_blocks(model, cfg.train.arch, cfg.train.unfreeze_last_n_blocks)
            optimizer = torch.optim.AdamW(
                parameter_groups(model, cfg.train.lr_head, cfg.train.lr_backbone),
                weight_decay=cfg.train.weight_decay,
            )
            print(
                f"[{tag}] phase 2: unfroze {info['unfrozen_blocks']}/{info['total_blocks']} "
                f"blocks, {info['trainable_params']:,} trainable params"
            )

        phase = "head" if epoch < total_head else "finetune"
        train_loss, *_ = _forward_epoch(
            model, train_loader, device, criterion, optimizer,
            cfg.train.grad_clip_norm, cfg.train.freeze_batchnorm,
        )
        row = {"epoch": epoch, "phase": phase, "train_loss": train_loss}

        if val_loader is not None:
            val_loss, val_probs, val_labels, val_idx = _forward_epoch(
                model, val_loader, device, criterion, None, 0.0, cfg.train.freeze_batchnorm
            )
            rank = ranking_metrics(val_labels, val_probs)
            row.update(
                val_loss=val_loss,
                val_roc_auc=rank["roc_auc"],
                val_pr_auc=rank["pr_auc"],
            )
            current = row.get(metric_name, float("nan"))
        else:
            current = -train_loss  # lower loss is better; keep "higher is better"
            row["val_loss"] = None

        history.append(row)
        print(
            f"[{tag}] epoch {epoch:02d} ({phase}) train_loss={train_loss:.4f} "
            + (
                f"val_loss={row.get('val_loss'):.4f} val_pr_auc={row.get('val_pr_auc'):.4f}"
                if val_loader is not None
                else ""
            )
        )

        improved = _is_better(
            current,
            best_metric,
            metric_name if val_loader is not None else "val_roc_auc",
            cfg.train.early_stopping_min_delta,
        )
        if improved:
            best_metric, best_epoch, patience_left = current, epoch, cfg.train.early_stopping_patience
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metric": current,
                 "metric_name": metric_name, "config": cfg.to_dict()},
                ckpt_best,
            )
        elif val_loader is not None and phase == "finetune":
            patience_left -= 1

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "best_metric": best_metric, "best_epoch": best_epoch,
             "history": history},
            ckpt_last,
        )

        if val_loader is not None and patience_left <= 0:
            stopped_early = True
            print(f"[{tag}] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if ckpt_best.exists():
        model.load_state_dict(torch.load(ckpt_best, map_location=device, weights_only=False)["model"])

    val_probs = val_labels = val_idx = None
    if val_loader is not None:
        _, val_probs, val_labels, val_idx = _forward_epoch(
            model, val_loader, device, criterion, None, 0.0, cfg.train.freeze_batchnorm
        )

    result = TrainResult(
        best_epoch=best_epoch,
        best_metric=float(best_metric),
        metric_name=metric_name,
        history=history,
        checkpoint_path=str(ckpt_best),
        device=dev_info.device,
        device_reason=dev_info.reason,
        n_train=len(train_df),
        n_val=len(val_df) if val_df is not None else 0,
        model_hash=model_state_hash(model),
        epochs_run=len(history),
        stopped_early=stopped_early,
        seconds=time.time() - started,
        val_probs=val_probs,
        val_labels=val_labels,
        val_index=val_idx,
    )
    (run_dir / f"{tag}_history.json").write_text(json.dumps(result.summary(), indent=2))
    return model, result


def predict_frame(model, frame: pd.DataFrame, cfg: Config, spec: PreprocessSpec) -> np.ndarray:
    """Raw (uncalibrated) sigmoid outputs for a frame, in row order."""
    import torch

    device = torch.device(select_device().device)
    model = model.to(device).eval()
    loader = make_loader(frame, spec, cfg.train.batch_size, False, None, cfg.train.num_workers)
    out = np.zeros(len(frame), dtype=float)
    with torch.no_grad():
        for batch_x, _, batch_idx in loader:
            logits = model(batch_x.to(device)).squeeze(1)
            out[np.asarray(batch_idx)] = torch.sigmoid(logits).float().cpu().numpy()
    return out
