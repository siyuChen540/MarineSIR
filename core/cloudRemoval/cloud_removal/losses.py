from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    one_d = torch.exp(-(coords**2) / (2 * sigma**2))
    one_d = one_d / one_d.sum()
    two_d = one_d[:, None] @ one_d[None, :]
    return two_d.expand(channel, 1, window_size, window_size).contiguous()


def ssim2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    val_range: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("ssim2d expects [N, C, H, W] tensors")

    common_dtype = torch.promote_types(pred.dtype, target.dtype)
    pred = pred.to(dtype=common_dtype)
    target = target.to(device=pred.device, dtype=common_dtype)

    _, channel, height, width = pred.shape
    real_window = min(window_size, height, width)
    if real_window % 2 == 0:
        real_window -= 1
    real_window = max(real_window, 3)
    window = _gaussian_window(real_window, 1.5, channel, pred.device, pred.dtype)

    padding = real_window // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target, window, padding=padding, groups=channel)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu_xy

    if val_range is None:
        value_range = (target.max() - target.min()).detach().clamp_min(eps)
    else:
        value_range = torch.tensor(val_range, device=pred.device, dtype=pred.dtype)
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2

    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    )
    return score.mean()


def _elementwise_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return (pred - target).pow(2)
    if kind == "l1":
        return (pred - target).abs()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unknown pixel loss: {kind}")


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    weighted = values * mask
    return weighted.sum() / mask.expand_as(values).sum().clamp_min(eps)


class HybridReconstructionLoss(nn.Module):
    def __init__(
        self,
        pixel_loss: str = "mse",
        pixel_weight: float = 0.16,
        ssim_weight: float = 0.84,
        mask_mode: str = "all",
        temporal_weight: float = 0.0,
        ssim_window: int = 11,
        ssim_val_range: float | None = None,
        normalize_ssim_loss: bool = True,
    ) -> None:
        super().__init__()
        self.pixel_loss = pixel_loss
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.mask_mode = mask_mode
        self.temporal_weight = temporal_weight
        self.ssim_window = ssim_window
        self.ssim_val_range = ssim_val_range
        self.normalize_ssim_loss = normalize_ssim_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_mask = self._select_mask(observed_mask)
        pixel = masked_mean(_elementwise_loss(pred, target, self.pixel_loss), loss_mask)

        pred_2d = pred.flatten(0, 1)
        target_2d = target.flatten(0, 1)
        ssim_score = ssim2d(pred_2d, target_2d, window_size=self.ssim_window, val_range=self.ssim_val_range)
        ssim_loss = 1.0 - ssim_score
        if self.normalize_ssim_loss:
            ssim_loss = ssim_loss * 0.5

        temporal = pred.new_tensor(0.0)
        if self.temporal_weight > 0 and pred.shape[1] > 1:
            pred_dt = pred[:, 1:] - pred[:, :-1]
            target_dt = target[:, 1:] - target[:, :-1]
            temporal = (pred_dt - target_dt).abs().mean()

        total = self.pixel_weight * pixel + self.ssim_weight * ssim_loss + self.temporal_weight * temporal
        return total, {
            "loss": total.detach(),
            "pixel_loss": pixel.detach(),
            "ssim_loss": ssim_loss.detach(),
            "ssim": ssim_score.detach(),
            "temporal_loss": temporal.detach(),
        }

    def _select_mask(self, observed_mask: torch.Tensor | None) -> torch.Tensor | None:
        if self.mask_mode == "all" or observed_mask is None:
            return None
        if self.mask_mode == "observed":
            return observed_mask
        if self.mask_mode == "missing":
            return 1.0 - observed_mask
        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")


def build_loss(config: dict[str, Any]) -> HybridReconstructionLoss:
    cfg = config.get("loss", {})
    return HybridReconstructionLoss(
        pixel_loss=cfg.get("pixel_loss", "mse"),
        pixel_weight=float(cfg.get("pixel_weight", 0.16)),
        ssim_weight=float(cfg.get("ssim_weight", 0.84)),
        mask_mode=cfg.get("mask_mode", "all"),
        temporal_weight=float(cfg.get("temporal_weight", 0.0)),
        ssim_window=int(cfg.get("ssim_window", 11)),
        ssim_val_range=cfg.get("ssim_val_range"),
        normalize_ssim_loss=bool(cfg.get("normalize_ssim_loss", True)),
    )
