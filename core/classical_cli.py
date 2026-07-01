from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def list_files(root: Path, suffix: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == suffix.lower())
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {root}")
    return files


def first_numeric_variable(dataset) -> str:
    for name, data_array in dataset.data_vars.items():
        if np.issubdtype(data_array.dtype, np.number) and data_array.ndim >= 2:
            return name
    raise ValueError("No numeric 2D variable found in NetCDF file")


def coerce_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).squeeze()
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D image, got {array.shape}")
    return array


def load_array(path: Path, variable: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return coerce_2d(np.load(path))
    if suffix in {".nc", ".nc4", ".cdf"}:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            variable = variable or first_numeric_variable(ds)
            return coerce_2d(ds[variable].values)
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return coerce_2d(tifffile.imread(path))
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def load_sequence(root: Path, suffix: str, variable: str | None) -> tuple[np.ndarray, list[Path]]:
    files = list_files(root, suffix)
    arrays = [load_array(path, variable) for path in files]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All frames must share one shape, got: {sorted(shapes)}")
    return np.stack(arrays, axis=0).astype(np.float32), files


def dineof_reconstruct(data: np.ndarray, rank: int = 8, max_iter: int = 50, tol: float = 1e-4) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.isfinite(data) & (data != 0)
    if not observed.any():
        raise ValueError("No observed pixels found for DINEOF")

    matrix = data.reshape(data.shape[0], -1).astype(np.float64)
    observed_matrix = observed.reshape(data.shape[0], -1)
    global_mean = float(np.nanmean(np.where(observed, data, np.nan)))
    filled = matrix.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(np.where(observed_matrix, matrix, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    missing = ~observed_matrix
    filled[missing] = np.take(col_means, np.where(missing)[1])

    last_delta = float("inf")
    rank = max(1, min(rank, min(filled.shape)))
    for iteration in range(1, max_iter + 1):
        previous_missing = filled[missing].copy()
        mean = filled.mean(axis=0, keepdims=True)
        centered = filled - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        reconstructed = (u[:, :rank] * s[:rank]) @ vt[:rank] + mean
        filled[missing] = reconstructed[missing]
        denom = np.linalg.norm(previous_missing) + 1e-8
        last_delta = float(np.linalg.norm(filled[missing] - previous_missing) / denom)
        print(f"DINEOF iteration {iteration}: delta={last_delta:.6g}", flush=True)
        if last_delta < tol:
            break

    result = filled.reshape(data.shape).astype(np.float32)
    info = {
        "rank": rank,
        "iterations": iteration,
        "delta": last_delta,
        "observed_ratio": float(observed.mean()),
    }
    return result, info


def save_outputs(output_dir: Path, data: np.ndarray, reconstruction: np.ndarray, files: list[Path], info: dict[str, Any], fmt: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_mask = (np.isfinite(data) & (data != 0)).astype(np.float32)
    clean_input = np.nan_to_num(data, nan=np.nan).astype(np.float32)
    if fmt == "npz":
        target = output_dir / "dineof_reconstruction.npz"
        np.savez_compressed(target, input=clean_input, reconstruction=reconstruction, observed_mask=observed_mask, files=[str(p) for p in files], **info)
        return target

    import xarray as xr

    time_size, height, width = reconstruction.shape
    ds = xr.Dataset(
        data_vars={
            "reconstruction": (("time", "y", "x"), reconstruction.astype(np.float32)),
            "input": (("time", "y", "x"), clean_input.astype(np.float32)),
            "observed_mask": (("time", "y", "x"), observed_mask.astype(np.float32)),
        },
        coords={"time": np.arange(time_size, dtype=np.int32), "y": np.arange(height, dtype=np.int32), "x": np.arange(width, dtype=np.int32)},
        attrs={"algorithm": "DINEOF", "source_files": " | ".join(str(p) for p in files), **{k: str(v) for k, v in info.items()}},
    )
    target = output_dir / "dineof_reconstruction.nc"
    if target.exists():
        target.unlink()
    ds.to_netcdf(target, mode="w", engine="scipy")
    return target


def run_dineof(args: argparse.Namespace) -> None:
    data, files = load_sequence(Path(args.data_root), args.suffix, args.variable)
    print(f"DINEOF input: frames={data.shape[0]}, height={data.shape[1]}, width={data.shape[2]}", flush=True)
    reconstruction, info = dineof_reconstruct(data, rank=args.rank, max_iter=args.max_iter, tol=args.tol)
    target = save_outputs(Path(args.output_dir), data, reconstruction, files, info, args.output_format)
    payload = {"output": str(target), **info}
    print("MARINESIR_CLASSICAL_RESULT=" + json.dumps(payload, ensure_ascii=False), flush=True)
    print(f"DINEOF result saved to: {target}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarineSIR classical reconstruction algorithms")
    sub = parser.add_subparsers(dest="command", required=True)
    dineof = sub.add_parser("dineof", help="Run a lightweight DINEOF/SVD gap-filling baseline")
    dineof.add_argument("--data-root", required=True)
    dineof.add_argument("--suffix", default=".npy")
    dineof.add_argument("--variable", default=None)
    dineof.add_argument("--output-dir", required=True)
    dineof.add_argument("--output-format", default="netcdf", choices=["netcdf", "npz"])
    dineof.add_argument("--rank", type=int, default=8)
    dineof.add_argument("--max-iter", type=int, default=50)
    dineof.add_argument("--tol", type=float, default=1e-4)
    dineof.set_defaults(func=run_dineof)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
