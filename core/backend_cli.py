from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _list_files(root: Path, suffix: str) -> list[Path]:
    suffix = suffix.lower()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == suffix)
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in: {root}")
    return files


def _first_numeric_variable(dataset) -> str:
    for name, data_array in dataset.data_vars.items():
        if np.issubdtype(data_array.dtype, np.number) and data_array.ndim >= 2:
            return name
    raise ValueError("No numeric 2D variable found in NetCDF file")


def _coerce_2d(array: np.ndarray) -> np.ndarray:
    array = np.array(array, dtype=np.float32, copy=True).squeeze()
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got shape={array.shape}")
    return array


def _load_array(path: Path, variable: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif suffix in {".nc", ".nc4", ".cdf"}:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = variable or _first_numeric_variable(ds)
            array = np.asarray(ds[var_name].values)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        array = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported suffix: {path.suffix}")
    return _coerce_2d(array)


def inspect_data(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    files = _list_files(root, args.suffix)
    samples: list[dict[str, Any]] = []
    finite_total = 0
    pixel_total = 0
    mins: list[float] = []
    maxs: list[float] = []
    means: list[float] = []
    shapes: list[tuple[int, int]] = []

    for path in files[: args.limit]:
        array = _load_array(path, args.variable)
        finite = np.isfinite(array)
        finite_count = int(finite.sum())
        finite_total += finite_count
        pixel_total += int(array.size)
        shapes.append(tuple(int(v) for v in array.shape))
        if finite_count:
            mins.append(float(np.nanmin(array)))
            maxs.append(float(np.nanmax(array)))
            means.append(float(np.nanmean(array)))
        samples.append(
            {
                "name": path.name,
                "shape": list(array.shape),
                "finite_pixels": finite_count,
                "pixels": int(array.size),
                "missing_ratio": float(1.0 - finite_count / max(1, array.size)),
                "min": float(np.nanmin(array)) if finite_count else None,
                "max": float(np.nanmax(array)) if finite_count else None,
                "mean": float(np.nanmean(array)) if finite_count else None,
            }
        )

    profile = {
        "data_root": str(root),
        "suffix": args.suffix,
        "file_count": len(files),
        "inspected_count": len(samples),
        "frames": args.frames,
        "window_count": max(0, len(files) - args.frames + 1),
        "shapes": sorted({str(shape) for shape in shapes}),
        "missing_ratio": float(1.0 - finite_total / max(1, pixel_total)),
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "mean": float(np.mean(means)) if means else None,
        "samples": samples,
    }

    print(f"MarineSIR data profile: {profile['file_count']} files, {profile['window_count']} windows")
    print(f"Missing ratio over inspected files: {profile['missing_ratio']:.4f}")
    print("MARINESIR_DATA_PROFILE=" + json.dumps(profile, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarineSIR backend helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect an input image sequence folder")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--suffix", default=".npy")
    inspect.add_argument("--variable", default=None)
    inspect.add_argument("--frames", type=int, default=4)
    inspect.add_argument("--limit", type=int, default=12)
    inspect.set_defaults(func=inspect_data)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
