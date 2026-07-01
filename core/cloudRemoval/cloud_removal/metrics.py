from __future__ import annotations

import torch

from .losses import masked_mean, ssim2d


@torch.no_grad()
def reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    mask_mode: str = "all",
) -> dict[str, float]:
    if observed_mask is None or mask_mode == "all":
        mask = None
    elif mask_mode == "missing":
        mask = 1.0 - observed_mask
    elif mask_mode == "observed":
        mask = observed_mask
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    error = pred - target
    mse = masked_mean(error.pow(2), mask)
    mae = masked_mean(error.abs(), mask)
    ssim = ssim2d(pred.flatten(0, 1), target.flatten(0, 1))
    return {
        "mse": float(mse.detach().cpu()),
        "rmse": float(torch.sqrt(mse).detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "ssim": float(ssim.detach().cpu()),
    }
