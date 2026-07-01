from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cloud_removal.config import apply_overrides, load_config
from cloud_removal.data import build_dataset
from cloud_removal.models import build_model
from cloud_removal.utils import ensure_dir, get_device, load_checkpoint, move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or export predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-format", default="npz", choices=["npz", "netcdf"])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def _inverse_transform(array: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    transform_cfg = config.get("data", {}).get("transform", {})
    if bool(transform_cfg.get("log10", False)):
        return np.power(10.0, array, dtype=np.float32)
    return array


def _save_npz(output_dir: Path, index: int, batch: dict[str, Any], pred: torch.Tensor) -> None:
    np.savez_compressed(
        output_dir / f"sample_{index:06d}.npz",
        inputs=batch["input"].detach().cpu().numpy(),
        targets=batch["target"].detach().cpu().numpy(),
        preds=pred.detach().cpu().numpy(),
        observed_masks=batch["observed_mask"].detach().cpu().numpy(),
    )


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    return [str(value)]


def _save_netcdf(output_dir: Path, index: int, batch: dict[str, Any], pred: torch.Tensor, config: dict[str, Any]) -> None:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("xarray and netCDF4 are required for NetCDF export") from exc

    pred_np = _inverse_transform(pred.detach().cpu().numpy()[0, :, 0], config).astype(np.float32)
    input_np = _inverse_transform(batch["input"].detach().cpu().numpy()[0, :, 0], config).astype(np.float32)
    target_np = _inverse_transform(batch["target"].detach().cpu().numpy()[0, :, 0], config).astype(np.float32)
    mask_np = batch["observed_mask"].detach().cpu().numpy()[0, :, 0].astype(np.float32)

    time_size, height, width = pred_np.shape
    ds = xr.Dataset(
        data_vars={
            "reconstruction": (("time", "y", "x"), pred_np),
            "input": (("time", "y", "x"), input_np),
            "target": (("time", "y", "x"), target_np),
            "observed_mask": (("time", "y", "x"), mask_np),
        },
        coords={
            "time": np.arange(time_size, dtype=np.int32),
            "y": np.arange(height, dtype=np.int32),
            "x": np.arange(width, dtype=np.int32),
        },
        attrs={
            "title": "MarineSIR cloud-removal reconstruction",
            "source_sample_id": (_flatten_text(batch.get("sample_id")) or [f"sample_{index:06d}"])[0],
            "source_paths": " | ".join(_flatten_text(batch.get("paths"))),
            "fourier_mode": str(config.get("model", {}).get("fourier_mode", "unknown")),
            "log10_transform_inverted": str(bool(config.get("data", {}).get("transform", {}).get("log10", False))),
        },
    )
    target_path = output_dir / f"sample_{index:06d}.nc"
    if target_path.exists():
        target_path.unlink()
    # The scipy backend is more reliable than netCDF4 for Unicode Windows paths.
    ds.to_netcdf(target_path, mode="w", engine="scipy")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    device = get_device(config.get("training", {}).get("device", "auto"))

    dataset = build_dataset(config, args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    output_dir = ensure_dir(args.output_dir or Path(args.checkpoint).parent.parent / f"predictions_{args.split}")
    for batch in tqdm(loader, desc=f"predict {args.split}"):
        batch = move_to_device(batch, device)
        pred = model(batch["input"])
        index = int(batch["index"].item())
        if args.output_format == "npz":
            _save_npz(output_dir, index, batch, pred)
        else:
            _save_netcdf(output_dir, index, batch, pred, config)

    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
