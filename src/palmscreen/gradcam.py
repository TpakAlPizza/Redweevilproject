"""Grad-CAM: an optional debugging aid.

What it is good for: catching a model that has locked onto a watermark, a fence
post, the sky, or the photographer's shadow.

What it is NOT: evidence that a prediction is correct. Heat maps look convincing
whether or not the model is right, and a model can attend to the right region for
entirely the wrong reason. Nothing produced here belongs in a claim about accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

DISCLAIMER = (
    "Grad-CAM is a qualitative debugging aid. A plausible heat map is not proof "
    "that the model is correct."
)


def _target_layer(model, arch: str):
    if arch in ("efficientnet_b0", "mobilenet_v3_large"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer configured for {arch}")


def gradcam_heatmap(model, tensor, target_layer) -> np.ndarray:
    """Grad-CAM for a single-logit binary head, returned as a 0-1 map."""
    import torch

    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(tensor).squeeze()
        logit.backward()
        acts = activations["value"][0]           # C,H,W
        grads = gradients["value"][0]            # C,H,W
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def explain_images(
    bundle_dir: str | Path,
    image_paths: List[str],
    out_dir: str | Path,
    alpha: float = 0.45,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .bundle import InferenceBundle
    from .preprocess import load_image, prepare_pil, to_tensor

    bundle = InferenceBundle.load(bundle_dir)
    model = bundle.model()
    layer = _target_layer(model, bundle.meta["arch"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for path in image_paths:
        pil = prepare_pil(load_image(path), bundle.spec)
        tensor = to_tensor(pil, bundle.spec).unsqueeze(0)
        cam = gradcam_heatmap(model, tensor, layer)
        result = bundle.predict(path)

        from PIL import Image

        heat = np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(pil.size, Image.Resampling.BILINEAR)
        ) / 255.0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))
        ax1.imshow(pil); ax1.set_title("input (as the model sees it)", fontsize=8); ax1.axis("off")
        ax2.imshow(pil); ax2.imshow(heat, cmap="jet", alpha=alpha)
        ax2.set_title(
            f"Grad-CAM - p(infected)={result['probability_infected']:.2f} -> {result['label']}",
            fontsize=8,
        )
        ax2.axis("off")
        fig.suptitle(DISCLAIMER, fontsize=7, color="#8a1c1c")
        target = out / (Path(path).stem + "_gradcam.png")
        fig.tight_layout()
        fig.savefig(target, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(target))

    (out / "README.txt").write_text(DISCLAIMER + "\n")
    return written


def gradcam_overlay_base64(bundle, image_source, alpha: float = 0.45) -> str:
    """Grad-CAM overlay for one image, returned as a base64 PNG.

    Used by the API so the web UI can show an explanation without writing files
    to disk. Same caveat as everywhere else: this shows where the network looked,
    which is not the same as showing that it looked for the right reason.
    """
    import base64
    import io

    import matplotlib
    from PIL import Image

    from .preprocess import load_image, prepare_pil, to_tensor

    model = bundle.model()
    layer = _target_layer(model, bundle.meta["arch"])
    pil = prepare_pil(load_image(image_source), bundle.spec)
    tensor = to_tensor(pil, bundle.spec).unsqueeze(0)

    cam = gradcam_heatmap(model, tensor, layer)
    heat = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(pil.size, Image.Resampling.BILINEAR)
    ) / 255.0

    colormap = matplotlib.colormaps["jet"]
    coloured = (colormap(heat)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(pil.convert("RGB"), dtype=np.float32)
    blended = (1 - alpha) * base + alpha * coloured.astype(np.float32)
    overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
