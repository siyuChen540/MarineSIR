from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import save_config
from .data import build_dataloaders
from .losses import build_loss
from .metrics import reconstruction_metrics
from .models import build_model
from .utils import (
    CSVLogger,
    EarlyStopping,
    MetricTracker,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    timestamp,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None


def create_run_dir(config: dict[str, Any]) -> Path:
    exp_cfg = config.get("experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", "record/refactor_runs"))
    name = exp_cfg.get("name", "experiment")
    run_name = exp_cfg.get("run_name") or f"{name}-{timestamp()}"
    return ensure_dir(output_dir / run_name)


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    cfg = config.get("optimizer", {})
    lr = float(cfg.get("lr", config.get("training", {}).get("lr", 2e-3)))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    cfg = config.get("scheduler", {})
    if cfg.get("name", "plateau") in {None, "none"}:
        return None
    if cfg.get("name", "plateau") != "plateau":
        raise ValueError("Only ReduceLROnPlateau scheduler is configured in this refactor")
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("factor", 0.5)),
        patience=int(cfg.get("patience", 14)),
        min_lr=float(cfg.get("min_lr", 0.0)),
    )


def train(config: dict[str, Any]) -> Path:
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(exp_cfg.get("deterministic", False)))

    run_dir = create_run_dir(config)
    save_config(config, run_dir / "config.resolved.yaml")
    logger = CSVLogger(run_dir / "metrics.csv")
    writer = SummaryWriter(str(run_dir / "tensorboard")) if SummaryWriter is not None else None

    device = get_device(train_cfg.get("device", "auto"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    start_epoch = 0
    best_metric = float("inf")
    resume_path = train_cfg.get("resume")
    if resume_path not in {None, "", "None"}:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.3f} M")

    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    early_stopping = EarlyStopping(
        patience=int(train_cfg.get("early_stopping_patience", 20)),
        min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 100))
    validate_every = int(train_cfg.get("validate_every", 1))
    sample_every = int(train_cfg.get("save_sample_every", validate_every))
    show_progress = bool(train_cfg.get("progress_bar", not bool(os.environ.get("MARINESIR_GUI"))))

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=True,
            config=config,
            epoch=epoch,
            show_progress=show_progress,
        )

        row: dict[str, Any] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})

        val_metrics: dict[str, float] | None = None
        if val_loader is not None and (epoch + 1) % validate_every == 0:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=device,
                train=False,
                config=config,
                epoch=epoch,
                show_progress=show_progress,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})

            if scheduler is not None:
                scheduler.step(val_metrics["loss"])

            improved = early_stopping.step(val_metrics["loss"])
            if improved:
                best_metric = val_metrics["loss"]
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, best_metric, config)
            if (epoch + 1) % sample_every == 0:
                save_prediction_sample(model, val_loader, device, run_dir, epoch, split="val")
            if early_stopping.should_stop:
                row["epoch_time_sec"] = time.perf_counter() - epoch_started
                print(f"Early stopping at epoch {epoch}")
                logger.write(row)
                print("MARINESIR_EPOCH_METRICS=" + json.dumps(row, ensure_ascii=False), flush=True)
                break
        elif scheduler is not None and val_loader is None:
            scheduler.step(train_metrics["loss"])

        row["epoch_time_sec"] = time.perf_counter() - epoch_started
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, best_metric, config)
        logger.write(row)
        print("MARINESIR_EPOCH_METRICS=" + json.dumps(row, ensure_ascii=False), flush=True)
        if writer is not None:
            write_tensorboard(writer, row, epoch)

    if writer is not None:
        writer.close()
    return run_dir


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    train: bool,
    config: dict[str, Any],
    epoch: int,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train(train)
    tracker = MetricTracker()
    train_cfg = config.get("training", {})
    metric_mask_mode = config.get("metrics", {}).get("mask_mode", config.get("loss", {}).get("mask_mode", "all"))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    desc = f"{'train' if train else 'valid'} {epoch:04d}"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    sample_count = 0
    batch_count = 0

    iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for batch in iterator:
        batch = move_to_device(batch, device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(batch["input"])
                loss, loss_items = criterion(pred, batch["target"], batch.get("observed_mask"))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, train_cfg)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradients(model, train_cfg)
                optimizer.step()

        batch_size = int(batch["input"].shape[0])
        sample_count += batch_size
        batch_count += 1
        values = {key: float(value.detach().cpu()) for key, value in loss_items.items()}
        values.update(reconstruction_metrics(pred.detach(), batch["target"], batch.get("observed_mask"), metric_mask_mode))
        tracker.update(values, n=batch_size)
        if show_progress:
            iterator.set_postfix(loss=f"{tracker.averages()['loss']:.5f}", ssim=f"{tracker.averages()['ssim']:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    averages = tracker.averages()
    averages["duration_sec"] = duration
    averages["samples_per_sec"] = sample_count / max(duration, 1e-8)
    averages["batches"] = float(batch_count)
    if device.type == "cuda":
        averages["gpu_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        averages["gpu_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        averages["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        averages["gpu_allocated_mb"] = 0.0
        averages["gpu_reserved_mb"] = 0.0
        averages["gpu_peak_allocated_mb"] = 0.0
    return averages


def clip_gradients(model: nn.Module, train_cfg: dict[str, Any]) -> None:
    clip_value = train_cfg.get("clip_grad_value")
    clip_norm = train_cfg.get("clip_grad_norm")
    if clip_value not in {None, "", "None"}:
        torch.nn.utils.clip_grad_value_(model.parameters(), float(clip_value))
    if clip_norm not in {None, "", "None"}:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))


@torch.no_grad()
def save_prediction_sample(model: nn.Module, loader, device: torch.device, run_dir: Path, epoch: int, split: str) -> None:
    model.eval()
    batch = next(iter(loader))
    batch = move_to_device(batch, device)
    pred = model(batch["input"])
    out_dir = ensure_dir(run_dir / "samples")
    np.savez_compressed(
        out_dir / f"{split}_epoch_{epoch:04d}.npz",
        inputs=batch["input"].detach().cpu().numpy(),
        targets=batch["target"].detach().cpu().numpy(),
        preds=pred.detach().cpu().numpy(),
        observed_masks=batch["observed_mask"].detach().cpu().numpy(),
        indices=batch["index"].detach().cpu().numpy(),
    )


def write_tensorboard(writer, row: dict[str, Any], epoch: int) -> None:
    for key, value in row.items():
        if key == "epoch":
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, epoch)
