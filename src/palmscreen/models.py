"""ImageNet-pretrained backbones with a single-logit head.

Nothing is trained from scratch: ~260 images cannot support it. Two compact
candidates only -- EfficientNet-B0 (default) and MobileNetV3-Large (baseline).
No architecture search, no ViTs.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

SUPPORTED = ("efficientnet_b0", "mobilenet_v3_large")


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
    import torch.nn as nn
    import torchvision.models as tvm

    if arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif arch == "mobilenet_v3_large":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED}.")
    return model


def _backbone_blocks(model, arch: str) -> List:
    """Ordered list of backbone stages, shallow first."""
    if arch == "efficientnet_b0":
        return list(model.features)
    if arch == "mobilenet_v3_large":
        return list(model.features)
    raise ValueError(arch)


def freeze_backbone(model, arch: str) -> None:
    """Phase 1: only the classification head learns."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top_blocks(model, arch: str, n_blocks: int) -> Dict[str, int]:
    """Phase 2: reopen the last N backbone stages at a low learning rate.

    Early layers hold generic edge/texture filters that ~200 training images
    cannot improve; late layers hold the ImageNet-specific semantics worth
    adapting to palm fronds.
    """
    blocks = _backbone_blocks(model, arch)
    n_blocks = max(0, min(n_blocks, len(blocks)))
    for block in blocks[len(blocks) - n_blocks :]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return {
        "total_blocks": len(blocks),
        "unfrozen_blocks": n_blocks,
        "trainable_params": count_trainable(model),
    }


def set_frozen_bn_eval(model) -> int:
    """Keep BatchNorm layers whose weights are frozen in eval mode.

    At batch size 8-16, letting frozen BN layers keep updating their running
    statistics quietly corrupts the pretrained features and produces
    train/eval gaps that look like overfitting but are not.
    """
    import torch.nn as nn

    frozen = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()
                frozen += 1
    return frozen


def count_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_state_hash(model) -> str:
    """SHA-256 over the sorted state dict: identifies exact trained weights."""
    import torch

    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu", torch.float32).contiguous()
        h.update(key.encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def parameter_groups(model, lr_head: float, lr_backbone: float) -> List[dict]:
    """Discriminative learning rates: head fast, reopened backbone slow."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith("classifier") else backbone_params).append(param)
    groups = [{"params": head_params, "lr": lr_head, "name": "head"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    return groups
