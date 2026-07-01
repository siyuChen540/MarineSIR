from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

try:
    from loguru import logger
except Exception:  # pragma: no cover - optional GUI dependency fallback
    import logging

    logger = logging.getLogger("marinesir")

from ui.log_panel import LogPanel
from ui.widgets import MplCanvas, RunSetupWidget


APP_ROOT = Path(__file__).resolve().parents[1]
CLOUD_ROOT = APP_ROOT / "core" / "cloudRemoval"
DEFAULT_BACKEND_PYTHON = Path(r"E:\Users\chens\anaconda3\envs\torch_311\python.exe")
DEFAULT_CONFIG = CLOUD_ROOT / "configs" / "fast_debug.yaml"
DEFAULT_DATA = APP_ROOT / "exampleDS"
DEFAULT_OUTPUT = APP_ROOT / "record" / "gui_runs"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MarineSIR - FTC-LSTM Reconstruction Workbench")
        self.resize(1500, 900)
        self.process: QProcess | None = None
        self.process_kind = ""
        self.run_dir: Path | None = None
        self.data_profile: dict | None = None
        self.last_metrics_row: dict[str, str] | None = None

        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(2000)
        self.metrics_timer.timeout.connect(self.refresh_training_views)

        self.setup_ui()
        self.apply_stylesheet()

    def setup_ui(self) -> None:
        defaults = {
            "backend_python": str(DEFAULT_BACKEND_PYTHON),
            "config_path": str(DEFAULT_CONFIG),
            "data_root": str(DEFAULT_DATA),
            "output_dir": str(DEFAULT_OUTPUT),
        }
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        self.setCentralWidget(central)

        self.setup_panel = RunSetupWidget(defaults)
        self.setup_panel.inspect_requested.connect(self.inspect_data)
        self.setup_panel.train_requested.connect(self.start_training)
        self.setup_panel.stop_requested.connect(self.stop_process)
        self.setup_panel.predict_requested.connect(self.export_predictions)

        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        setup_scroll.setObjectName("SetupScroll")
        setup_scroll.setWidget(self.setup_panel)
        setup_scroll.setMinimumWidth(360)
        setup_scroll.setMaximumWidth(500)
        layout.addWidget(setup_scroll)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setPlaceholderText("Inspect a dataset to see file count, missing ratio, value range, and sample windows.")
        self.tabs.addTab(self.overview_text, "Data")

        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("StatusLabel")
        self.metrics_canvas = MplCanvas(width=8, height=6.2, nrows=3, ncols=1)
        training_layout.addWidget(self.status_label)
        training_layout.addWidget(self.metrics_canvas, 1)
        self.tabs.addTab(training_tab, "Training")

        self.sample_canvas = MplCanvas(width=8, height=6, nrows=2, ncols=2)
        self.tabs.addTab(self.sample_canvas, "Samples")

        self.parameters_text = QTextEdit()
        self.parameters_text.setReadOnly(True)
        self.tabs.addTab(self.parameters_text, "Parameters")

        self.log_panel = LogPanel()
        self.tabs.addTab(self.log_panel, "Log")
        self.configure_logger()
        self.render_parameter_summary()
        self.statusBar().showMessage("Ready")


    def configure_logger(self) -> None:
        if hasattr(logger, "remove") and hasattr(logger, "add"):
            logger.remove()
            logger.add(self._loguru_sink, level="DEBUG", format="{message}")
        self.log("INFO", "MarineSIR GUI ready", "app")

    def _loguru_sink(self, message) -> None:
        record = getattr(message, "record", None)
        if record is None:
            self.log_panel.append("INFO", str(message), "app")
            return
        self.log_panel.append(
            record["level"].name,
            record["message"],
            record.get("extra", {}).get("source", "app"),
        )

    def log(self, level: str, message: str, source: str = "app") -> None:
        level = level.upper()
        if hasattr(logger, "bind"):
            bound = logger.bind(source=source)
            getattr(bound, level.lower(), bound.info)(message)
        else:
            self.log_panel.append(level, message, source)

    def apply_stylesheet(self) -> None:
        qss = APP_ROOT / "assets" / "style.qss"
        if qss.exists():
            self.setStyleSheet(qss.read_text(encoding="utf-8"))

    def settings(self) -> dict:
        return self.setup_panel.settings()

    def inspect_data(self) -> None:
        settings = self.settings()
        self.render_parameter_summary()
        args = [
            str(APP_ROOT / "core" / "backend_cli.py"),
            "inspect",
            "--data-root",
            str(settings["data_root"]),
            "--suffix",
            str(settings["suffix"]),
            "--frames",
            str(settings["frames"]),
        ]
        if settings["variable"]:
            args.extend(["--variable", str(settings["variable"])])
        self.start_process("inspect", str(settings["backend_python"]), args)

    def start_training(self) -> None:
        settings = self.settings()
        self.render_parameter_summary()
        algorithm = str(settings.get("algorithm", "FTC-LSTM"))
        if algorithm == "DINEOF":
            self.run_dineof(settings)
            return
        if algorithm.startswith("DINCAE"):
            QMessageBox.information(
                self,
                "DINCAE planned",
                "DINCAE needs a separate pretrained or training implementation. The software now exposes the slot, but this backend is not yet bundled.",
            )
            return

        self.run_dir = None
        args = [str(CLOUD_ROOT / "train.py"), "--config", str(settings["config_path"])]
        for override in self.training_overrides(settings):
            args.extend(["--set", override])
        self.start_process("train", str(settings["backend_python"]), args)
        self.metrics_timer.start()

    def run_dineof(self, settings: dict) -> None:
        output_dir = Path(str(settings["output_dir"])) / "dineof"
        args = [
            str(APP_ROOT / "core" / "classical_cli.py"),
            "dineof",
            "--data-root",
            str(settings["data_root"]),
            "--suffix",
            str(settings["suffix"]),
            "--output-dir",
            str(output_dir),
            "--rank",
            "8",
            "--max-iter",
            "50",
        ]
        if settings["variable"]:
            args.extend(["--variable", str(settings["variable"])])
        self.start_process("dineof", str(settings["backend_python"]), args)

    def export_predictions(self) -> None:
        settings = self.settings()
        self.render_parameter_summary()
        if str(settings.get("algorithm", "FTC-LSTM")) == "DINEOF":
            self.run_dineof(settings)
            return
        checkpoint = str(settings["checkpoint_path"]).strip()
        if not checkpoint and self.run_dir is not None:
            best = self.run_dir / "checkpoints" / "best.pt"
            last = self.run_dir / "checkpoints" / "last.pt"
            checkpoint = str(best if best.exists() else last)
            self.setup_panel.checkpoint_path.setText(checkpoint)
        if not checkpoint:
            QMessageBox.warning(self, "Checkpoint required", "Select a checkpoint or train a model first.")
            return

        output_dir = Path(str(settings["output_dir"])) / "predictions"
        args = [
            str(CLOUD_ROOT / "evaluate.py"),
            "--config",
            str(settings["config_path"]),
            "--checkpoint",
            checkpoint,
            "--split",
            "all",
            "--output-dir",
            str(output_dir),
            "--output-format",
            str(settings["output_format"]),
        ]
        for override in self.training_overrides(settings):
            args.extend(["--set", override])
        self.start_process("predict", str(settings["backend_python"]), args)

    def training_overrides(self, settings: dict) -> list[str]:
        data_root = Path(str(settings["data_root"])).as_posix()
        output_dir = Path(str(settings["output_dir"])).as_posix()
        mask_root = str(settings["mask_root"]).strip()
        variable = str(settings["variable"]).strip()
        image_size = f"[{settings['image_h']},{settings['image_w']}]"
        overrides = [
            f"data.root_dir={data_root}",
            f"data.mask_dir={Path(mask_root).as_posix() if mask_root else 'null'}",
            f"data.suffix={settings['suffix']}",
            f"data.variable={variable if variable else 'null'}",
            f"data.frames={settings['frames']}",
            f"data.image_size={image_size}",
            f"training.device={settings['device']}",
            "training.progress_bar=false",
            f"model.frames={settings['frames']}",
            f"model.fourier_mode={settings['fourier_mode']}",
            f"loss.pixel_loss={settings['pixel_loss']}",
            f"loss.mask_mode={settings['mask_mode']}",
            f"metrics.mask_mode={settings['mask_mode']}",
            f"training.epochs={settings['epochs']}",
            f"training.batch_size={settings['batch_size']}",
            f"experiment.output_dir={output_dir}",
            "experiment.name=marinesir-gui",
        ]
        return overrides

    def start_process(self, kind: str, program: str, args: list[str]) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Process running", "Stop the current backend process before starting another one.")
            return
        if not Path(program).exists():
            QMessageBox.warning(self, "Python not found", f"Backend Python does not exist:\n{program}")
            return

        self.process_kind = kind
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(CLOUD_ROOT))
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH")
        pythonpath = str(CLOUD_ROOT) if not existing else str(CLOUD_ROOT) + os.pathsep + existing
        env.insert("PYTHONPATH", pythonpath)
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("MARINESIR_GUI", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)

        self.setup_panel.set_busy(True)
        self.status_label.setText(f"Running {kind}...")
        self.statusBar().showMessage(f"Running {kind}")
        self.log("PROCESS", f"$ {program} {' '.join(args)}", kind)
        self.process.start(program, args)

    def stop_process(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.log("WARNING", "Stopping backend process...", "process")
            self.process.terminate()
            QTimer.singleShot(4000, self.kill_process_if_needed)

    def kill_process_if_needed(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.log("ERROR", "Backend did not exit after terminate; killing it.", "process")
            self.process.kill()

    def read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in self.clean_process_lines(text):
            self.handle_output_line(line)

    def read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in self.clean_process_lines(text):
            level = "WARNING" if "warning" in line.lower() else "ERROR"
            self.log(level, line, "stderr")


    @staticmethod
    def clean_process_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.replace("\r", "\n").splitlines():
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()
            if not line:
                continue
            if "|" in line and "it/s" in line and "%" in line:
                continue
            lines.append(line)
        return lines

    def handle_output_line(self, line: str) -> None:
        if line.startswith("MARINESIR_EPOCH_METRICS="):
            payload = line.split("=", 1)[1]
            row = json.loads(payload)
            self.last_metrics_row = {key: str(value) for key, value in row.items()}
            self.log("INFO", self.format_epoch_summary(row), "metrics")
            return
        if line.startswith("MARINESIR_CLASSICAL_RESULT="):
            payload = json.loads(line.split("=", 1)[1])
            self.log("INFO", f"Classical result saved: {payload.get('output')}", "dineof")
            self.statusBar().showMessage(f"DINEOF completed: {payload.get('output')}")
            return
        self.log("INFO", line, self.process_kind or "stdout")
        if line.startswith("MARINESIR_DATA_PROFILE="):
            payload = line.split("=", 1)[1]
            self.data_profile = json.loads(payload)
            self.render_data_profile()
        elif line.startswith("Run directory:"):
            self.run_dir = Path(line.split(":", 1)[1].strip())
            self.status_label.setText(f"Training run: {self.run_dir}")
        elif line.startswith("Predictions saved to:"):
            self.statusBar().showMessage(line)


    @staticmethod
    def format_epoch_summary(row: dict) -> str:
        epoch = int(float(row.get("epoch", 0))) + 1
        pieces = [f"epoch={epoch}"]
        for key in ["train_loss", "val_loss", "train_rmse", "val_rmse", "train_ssim", "val_ssim", "train_samples_per_sec", "train_gpu_reserved_mb", "epoch_time_sec"]:
            if key in row and row[key] not in {None, ""}:
                value = row[key]
                if isinstance(value, float):
                    pieces.append(f"{key}={value:.4g}")
                else:
                    pieces.append(f"{key}={value}")
        return " | ".join(pieces)

    def process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        ok = exit_code == 0 and exit_status == QProcess.NormalExit
        self.log("INFO" if ok else "ERROR", f"Process finished: kind={self.process_kind}, exit_code={exit_code}", "process")
        self.setup_panel.set_busy(False)
        if self.process_kind == "train":
            self.metrics_timer.stop()
            self.refresh_training_views()
            if self.run_dir is not None:
                best = self.run_dir / "checkpoints" / "best.pt"
                last = self.run_dir / "checkpoints" / "last.pt"
                self.setup_panel.checkpoint_path.setText(str(best if best.exists() else last))
        self.status_label.setText("Completed" if ok else "Failed")
        self.statusBar().showMessage("Completed" if ok else "Failed")

    def refresh_training_views(self) -> None:
        if self.run_dir is None:
            return
        self.plot_metrics(self.run_dir / "metrics.csv")
        self.plot_latest_sample(self.run_dir / "samples")

    def plot_metrics(self, metrics_path: Path) -> None:
        if not metrics_path.exists():
            return
        with metrics_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        axes = np.ravel(self.metrics_canvas.axes)
        for ax in axes:
            ax.clear()
        epochs = [int(float(row.get("epoch", i))) + 1 for i, row in enumerate(rows)]
        self._plot_fields(axes[0], rows, epochs, ["train_loss", "val_loss", "train_pixel_loss", "val_pixel_loss"], "Loss")
        self._plot_fields(axes[1], rows, epochs, ["train_rmse", "val_rmse", "train_ssim", "val_ssim", "train_mae", "val_mae"], "Reconstruction Metrics")
        self._plot_fields(axes[2], rows, epochs, ["train_samples_per_sec", "val_samples_per_sec", "train_gpu_reserved_mb", "train_gpu_peak_allocated_mb", "epoch_time_sec"], "Speed and Device")
        self.last_metrics_row = rows[-1]
        self.update_status_from_metrics(rows[-1])
        self.metrics_canvas.figure.tight_layout(pad=1.4)
        self.metrics_canvas.draw_idle()


    def update_status_from_metrics(self, row: dict[str, str]) -> None:
        parts = []
        if row.get("epoch") not in {None, ""}:
            parts.append(f"Epoch {int(float(row['epoch'])) + 1}")
        for key, label in [("train_loss", "train loss"), ("val_loss", "val loss"), ("train_samples_per_sec", "samples/s"), ("train_gpu_reserved_mb", "GPU MB")]:
            raw = row.get(key)
            if raw not in {None, ""}:
                try:
                    parts.append(f"{label}: {float(raw):.4g}")
                except ValueError:
                    parts.append(f"{label}: {raw}")
        if parts:
            self.status_label.setText(" | ".join(parts))

    def _plot_fields(self, ax, rows: list[dict], epochs: list[int], fields: list[str], title: str) -> None:
        plotted = False
        for field in fields:
            values = []
            xs = []
            for epoch, row in zip(epochs, rows):
                raw = row.get(field)
                if raw not in {None, ""}:
                    xs.append(epoch)
                    values.append(float(raw))
            if values:
                ax.plot(xs, values, marker="o", linewidth=1.5, markersize=3, label=field)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        if plotted:
            ax.legend(loc="best", fontsize=8)

    def plot_latest_sample(self, samples_dir: Path) -> None:
        if not samples_dir.exists():
            return
        samples = sorted(samples_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not samples:
            return
        sample_path = samples[-1]
        data = np.load(sample_path)
        titles = ["Input", "Target", "Prediction", "Observed mask"]
        keys = ["inputs", "targets", "preds", "observed_masks"]
        axes = np.ravel(self.sample_canvas.axes)
        for ax, title, key in zip(axes, titles, keys):
            ax.clear()
            image = self._last_frame(data[key])
            cmap = "gray" if "mask" in key else "viridis"
            ax.imshow(image, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        self.sample_canvas.figure.suptitle(sample_path.name)
        self.sample_canvas.figure.tight_layout(pad=1.3)
        self.sample_canvas.draw_idle()

    @staticmethod
    def _last_frame(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        if value.ndim == 5:
            return value[0, -1, 0]
        if value.ndim == 4:
            return value[-1, 0]
        while value.ndim > 2:
            value = value[0]
        return value

    def render_data_profile(self) -> None:
        if not self.data_profile:
            return
        profile = self.data_profile
        lines = [
            "MarineSIR Data Profile",
            "",
            f"Root: {profile['data_root']}",
            f"Format: {profile['suffix']}",
            f"Files: {profile['file_count']}",
            f"Windows: {profile['window_count']} with {profile['frames']} frames",
            f"Shapes: {', '.join(profile['shapes'])}",
            f"Missing ratio: {profile['missing_ratio']:.4f}",
            f"Value range: {profile['min']} to {profile['max']}",
            f"Mean of inspected frames: {profile['mean']}",
            "",
            "Inspected files:",
        ]
        for sample in profile.get("samples", []):
            lines.append(
                f"- {sample['name']} | shape={sample['shape']} | missing={sample['missing_ratio']:.4f} | mean={sample['mean']}"
            )
        self.overview_text.setPlainText("\n".join(lines))
        self.tabs.setCurrentWidget(self.overview_text)

    def render_parameter_summary(self) -> None:
        settings = self.settings()
        overrides = self.training_overrides(settings) if settings.get("algorithm") == "FTC-LSTM" else []
        lines = [
            "MarineSIR Parameter Summary",
            "",
            "Runtime",
            f"- Backend Python: {settings['backend_python']}",
            f"- Device: {settings['device']} (auto uses CUDA when available, otherwise CPU)",
            f"- Config: {settings['config_path']}",
            f"- Output: {settings['output_dir']}",
            "",
            "Data",
            f"- Input folder: {settings['data_root']}",
            f"- Mask folder: {settings['mask_root'] or 'None'}",
            f"- Format: {settings['suffix']}",
            f"- Variable: {settings['variable'] or 'auto/none'}",
            f"- Frames per sample: {settings['frames']}",
            f"- Resize: {settings['image_h']} x {settings['image_w']}",
            "",
            "Algorithm",
            f"- Selected: {settings['algorithm']}",
            "- FTC-LSTM: trains or evaluates the Fourier ConvLSTM backend in core/cloudRemoval.",
            "- DINEOF: classical EOF/SVD gap-filling baseline for quick comparison.",
            "- DINCAE: reserved integration point; not bundled in this revision.",
            "",
            "Training",
            f"- Fourier mode: {settings['fourier_mode']}",
            f"- Pixel loss: {settings['pixel_loss']}",
            f"- Metric region: {settings['mask_mode']}",
            f"- Epochs: {settings['epochs']}",
            f"- Batch size: {settings['batch_size']}",
            "",
            "Runtime metrics shown during training",
            "- Loss and reconstruction metrics: loss, RMSE, MAE, SSIM.",
            "- Speed: samples per second and epoch duration.",
            "- GPU: allocated, reserved, and peak allocated memory in MB when CUDA is used.",
        ]
        if overrides:
            lines.extend(["", "Resolved GUI overrides:"])
            lines.extend(f"- {item}" for item in overrides)
        self.parameters_text.setPlainText("\n".join(lines))
