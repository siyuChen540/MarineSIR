from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _as_path_or_none(value: str | None) -> Path | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return Path(value)


def _list_files(root: Path, suffix: str = ".npy", sort_files: bool = True) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    suffix = suffix.lower()
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == suffix]
    if sort_files:
        files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in: {root}")
    return files


def _build_windows(files: list[Path], frames: int, stride: int) -> list[list[Path]]:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if len(files) < frames:
        raise ValueError(f"Need at least {frames} files, got {len(files)}")
    return [files[i : i + frames] for i in range(0, len(files) - frames + 1, stride)]


def _split_indices(
    length: int,
    split: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    mode: str,
) -> list[int]:
    indices = np.arange(length)
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    elif mode != "chronological":
        raise ValueError(f"Unknown split mode: {mode}")

    train_end = int(round(length * train_ratio))
    val_end = train_end + int(round(length * val_ratio))
    if split == "train":
        selected = indices[:train_end]
    elif split in {"val", "valid", "validation"}:
        selected = indices[train_end:val_end]
    elif split == "test":
        selected = indices[val_end:]
    elif split == "all":
        selected = indices
    else:
        raise ValueError(f"Unknown split: {split}")

    return selected.tolist()


class SequenceNpyDataset(Dataset):
    """Load consecutive image windows for cloud-removal reconstruction.

    Returned tensors use [T, C, H, W]. ``observed_mask`` is 1 for visible pixels
    and 0 for cloud/missing pixels.
    """

    def __init__(
        self,
        root_dir: str,
        frames: int,
        image_size: list[int] | tuple[int, int] | None = None,
        mask_dir: str | None = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        split_mode: str = "chronological",
        stride: int = 1,
        suffix: str = ".npy",
        variable: str | None = None,
        sort_files: bool = True,
        seed: int = 42,
        use_mmap: bool = True,
        mask_sampling: str = "aligned",
        synthetic_keep_prob: float = 0.8,
        log10: bool = True,
        nan_value: float = 1.0,
        zero_value: float | None = 1.0,
        normalize: str = "none",
        eps: float = 1e-6,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mask_dir = _as_path_or_none(mask_dir)
        self.frames = frames
        self.image_size = tuple(image_size) if image_size else None
        self.split = split
        self.seed = seed
        self.use_mmap = use_mmap
        self.mask_sampling = mask_sampling
        self.variable = variable
        self.synthetic_keep_prob = synthetic_keep_prob
        self.log10 = log10
        self.nan_value = nan_value
        self.zero_value = zero_value
        self.normalize = normalize
        self.eps = eps

        files = _list_files(self.root_dir, suffix=suffix, sort_files=sort_files)
        all_windows = _build_windows(files, frames=frames, stride=stride)
        indices = _split_indices(
            len(all_windows),
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            mode=split_mode,
        )
        self.windows = [all_windows[i] for i in indices]

        self.mask_windows: list[list[Path]] = []
        if self.mask_dir is not None:
            mask_files = _list_files(self.mask_dir, suffix=suffix, sort_files=sort_files)
            all_mask_windows = _build_windows(mask_files, frames=frames, stride=stride)
            mask_indices = _split_indices(
                len(all_mask_windows),
                split=split,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                mode=split_mode,
            )
            self.mask_windows = [all_mask_windows[i] for i in mask_indices]

        if not self.windows:
            raise ValueError(f"No samples for split={split}. Check split ratios and data length.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_paths = self.windows[index]
        mask_paths = self._select_mask_window(index)

        targets: list[torch.Tensor] = []
        inputs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for step, frame_path in enumerate(frame_paths):
            target = self._load_image_tensor(frame_path)
            if mask_paths is None:
                mask = self._synthetic_mask(target, sample_index=index, step=step)
            else:
                mask = self._load_mask_tensor(mask_paths[step], target.shape[-2:])

            targets.append(target)
            masks.append(mask)
            inputs.append(target * mask)

        target_tensor = torch.stack(targets, dim=0)
        input_tensor = torch.stack(inputs, dim=0)
        mask_tensor = torch.stack(masks, dim=0)

        return {
            "index": torch.tensor(index, dtype=torch.long),
            "input": input_tensor,
            "target": target_tensor,
            "observed_mask": mask_tensor,
            "missing_mask": 1.0 - mask_tensor,
            "sample_id": frame_paths[0].stem,
            "paths": [str(p) for p in frame_paths],
        }

    def _select_mask_window(self, index: int) -> list[Path] | None:
        if not self.mask_windows:
            return None
        if self.mask_sampling == "aligned":
            return self.mask_windows[min(index, len(self.mask_windows) - 1)]
        if self.mask_sampling == "random":
            rng = np.random.default_rng(self.seed + index)
            return self.mask_windows[int(rng.integers(0, len(self.mask_windows)))]
        raise ValueError(f"Unknown mask_sampling: {self.mask_sampling}")

    def _load_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            mmap_mode = "r" if self.use_mmap else None
            array = np.load(path, mmap_mode=mmap_mode)
        elif suffix in {".nc", ".nc4", ".cdf"}:
            array = self._load_netcdf_array(path)
        elif suffix in {".tif", ".tiff"}:
            array = self._load_tiff_array(path)
        else:
            raise ValueError(f"Unsupported data suffix: {path.suffix}")
        return self._coerce_2d(array)

    def _load_netcdf_array(self, path: Path) -> np.ndarray:
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError("xarray is required to read NetCDF inputs") from exc

        with xr.open_dataset(path) as ds:
            variable = self.variable or self._first_numeric_variable(ds)
            if variable not in ds:
                raise KeyError(f"Variable '{variable}' not found in {path}")
            return np.asarray(ds[variable].values)

    @staticmethod
    def _first_numeric_variable(dataset) -> str:
        for name, data_array in dataset.data_vars.items():
            if np.issubdtype(data_array.dtype, np.number) and data_array.ndim >= 2:
                return name
        raise ValueError("No numeric 2D variable found in NetCDF file")

    @staticmethod
    def _load_tiff_array(path: Path) -> np.ndarray:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError("tifffile is required to read TIFF inputs") from exc
        return np.asarray(tifffile.imread(path))

    @staticmethod
    def _coerce_2d(array: np.ndarray) -> np.ndarray:
        array = np.array(array, dtype=np.float32, copy=True).squeeze()
        while array.ndim > 2:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D image array, got shape={array.shape}")
        return array

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        array = self._load_array(path)
        array = np.nan_to_num(array, nan=self.nan_value, posinf=self.nan_value, neginf=self.nan_value)
        if self.zero_value is not None:
            array[array == 0.0] = self.zero_value
        if self.log10:
            array = np.log10(np.clip(array, self.eps, None))
        tensor = torch.from_numpy(array).float().unsqueeze(0)
        tensor = self._resize(tensor, mode="bilinear")
        if self.normalize == "per_sample":
            std = tensor.std().clamp_min(self.eps)
            tensor = (tensor - tensor.mean()) / std
        elif self.normalize != "none":
            raise ValueError(f"Unknown normalize mode: {self.normalize}")
        return tensor

    def _load_mask_tensor(self, path: Path, target_size: tuple[int, int]) -> torch.Tensor:
        array = self._load_array(path)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)
        tensor = self._resize(tensor, size=target_size, mode="nearest")
        return (tensor > 0.5).float()

    def _resize(
        self,
        tensor: torch.Tensor,
        size: tuple[int, int] | None = None,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        target_size = size or self.image_size
        if target_size is None or tuple(tensor.shape[-2:]) == tuple(target_size):
            return tensor
        batch = tensor.unsqueeze(0)
        if mode == "nearest":
            resized = F.interpolate(batch, size=target_size, mode=mode)
        else:
            resized = F.interpolate(batch, size=target_size, mode=mode, align_corners=False)
        return resized.squeeze(0)

    def _synthetic_mask(self, target: torch.Tensor, sample_index: int, step: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1000003 + sample_index * 101 + step)
        mask = rng.random(target.shape, dtype=np.float32) < self.synthetic_keep_prob
        return torch.from_numpy(mask.astype(np.float32))


def build_dataset(config: dict[str, Any], split: str) -> SequenceNpyDataset:
    data_cfg = config.get("data", {})
    transform_cfg = data_cfg.get("transform", {})
    mask_cfg = data_cfg.get("synthetic_mask", {})
    split_cfg = data_cfg.get("split", {})

    return SequenceNpyDataset(
        root_dir=data_cfg["root_dir"],
        frames=int(data_cfg.get("frames", config.get("model", {}).get("frames", 4))),
        image_size=data_cfg.get("image_size"),
        mask_dir=data_cfg.get("mask_dir"),
        split=split,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        split_mode=split_cfg.get("mode", "chronological"),
        stride=int(data_cfg.get("stride", 1)),
        suffix=data_cfg.get("suffix", ".npy"),
        variable=data_cfg.get("variable"),
        sort_files=bool(data_cfg.get("sort_files", True)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
        use_mmap=bool(data_cfg.get("use_mmap", True)),
        mask_sampling=data_cfg.get("mask_sampling", "aligned"),
        synthetic_keep_prob=float(mask_cfg.get("keep_prob", 0.8)),
        log10=bool(transform_cfg.get("log10", True)),
        nan_value=float(transform_cfg.get("nan_value", 1.0)),
        zero_value=transform_cfg.get("zero_value", 1.0),
        normalize=transform_cfg.get("normalize", "none"),
        eps=float(transform_cfg.get("eps", 1e-6)),
    )


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    training_cfg = config.get("training", {})
    loader_cfg = config.get("data", {}).get("loader", {})

    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0

    train_dataset = build_dataset(config, "train")
    val_dataset = None
    try:
        val_dataset = build_dataset(config, "val")
    except ValueError:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(loader_cfg.get("shuffle_train", True)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(loader_cfg.get("drop_last", False)),
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
        )
    return train_loader, val_loader

