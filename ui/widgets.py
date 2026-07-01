from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, nrows: int = 1, ncols: int = 1):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor="#f7f8fa")
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.tight_layout(pad=1.6)


class PathPicker(QWidget):
    def __init__(self, mode: str = "dir", caption: str = "Select", text: str = "") -> None:
        super().__init__()
        self.mode = mode
        self.caption = caption
        self.edit = QLineEdit(text)
        self.button = QPushButton("Browse")
        self.button.setObjectName("SecondaryButton")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text)

    def _browse(self) -> None:
        current = self.text() or str(Path.home())
        if self.mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, self.caption, current)
        elif self.mode == "save_dir":
            path = QFileDialog.getExistingDirectory(self, self.caption, current)
        else:
            path = QFileDialog.getExistingDirectory(self, self.caption, current)
        if path:
            self.edit.setText(path)


class RunSetupWidget(QFrame):
    inspect_requested = pyqtSignal()
    train_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    predict_requested = pyqtSignal()

    def __init__(self, defaults: dict[str, str]) -> None:
        super().__init__()
        self.setObjectName("SetupPanel")
        self.setMinimumWidth(340)
        self.setMaximumWidth(460)
        root = QVBoxLayout(self)
        root.setSpacing(14)

        title = QLabel("MarineSIR")
        title.setObjectName("AppTitle")
        subtitle = QLabel("FTC-LSTM reconstruction workbench")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        runtime = QGroupBox("Runtime")
        runtime_form = QFormLayout(runtime)
        self.backend_python = PathPicker("file", "Select backend Python", defaults["backend_python"])
        self.config_path = PathPicker("file", "Select YAML config", defaults["config_path"])
        self.output_dir = PathPicker("dir", "Select output folder", defaults["output_dir"])
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
        runtime_form.addRow("Backend Python", self.backend_python)
        runtime_form.addRow("Device", self.device)
        runtime_form.addRow("Config", self.config_path)
        runtime_form.addRow("Output", self.output_dir)
        root.addWidget(runtime)

        data = QGroupBox("Data")
        data_form = QFormLayout(data)
        self.data_root = PathPicker("dir", "Select data folder", defaults["data_root"])
        self.mask_root = PathPicker("dir", "Select mask folder", "")
        self.suffix = QComboBox()
        self.suffix.addItems([".npy", ".nc", ".nc4", ".tif", ".tiff"])
        self.variable = QLineEdit()
        self.variable.setPlaceholderText("NetCDF variable, optional")
        self.frames = QSpinBox()
        self.frames.setRange(1, 64)
        self.frames.setValue(4)
        self.image_h = QSpinBox()
        self.image_h.setRange(16, 4096)
        self.image_h.setValue(48)
        self.image_w = QSpinBox()
        self.image_w.setRange(16, 4096)
        self.image_w.setValue(48)
        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(self.image_h)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.image_w)
        data_form.addRow("Input folder", self.data_root)
        data_form.addRow("Mask folder", self.mask_root)
        data_form.addRow("Format", self.suffix)
        data_form.addRow("Variable", self.variable)
        data_form.addRow("Frames", self.frames)
        data_form.addRow("Image size", size_row)
        root.addWidget(data)

        model = QGroupBox("Model and Training")
        model_grid = QGridLayout(model)
        self.algorithm = QComboBox()
        self.algorithm.addItems(["FTC-LSTM", "DINEOF", "DINCAE (planned)"])
        self.fourier_mode = QComboBox()
        self.fourier_mode.addItems(["fft_add", "fft_concat", "none"])
        self.pixel_loss = QComboBox()
        self.pixel_loss.addItems(["mse", "l1", "smooth_l1"])
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["all", "missing", "observed"])
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(2)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(1)
        model_grid.addWidget(QLabel("Algorithm"), 0, 0)
        model_grid.addWidget(self.algorithm, 0, 1)
        model_grid.addWidget(QLabel("Fourier"), 1, 0)
        model_grid.addWidget(self.fourier_mode, 1, 1)
        model_grid.addWidget(QLabel("Pixel loss"), 2, 0)
        model_grid.addWidget(self.pixel_loss, 2, 1)
        model_grid.addWidget(QLabel("Metric region"), 3, 0)
        model_grid.addWidget(self.mask_mode, 3, 1)
        model_grid.addWidget(QLabel("Epochs"), 4, 0)
        model_grid.addWidget(self.epochs, 4, 1)
        model_grid.addWidget(QLabel("Batch"), 5, 0)
        model_grid.addWidget(self.batch_size, 5, 1)
        root.addWidget(model)

        export = QGroupBox("Prediction Export")
        export_form = QFormLayout(export)
        self.checkpoint_path = PathPicker("file", "Select checkpoint", "")
        self.output_format = QComboBox()
        self.output_format.addItems(["netcdf", "npz"])
        export_form.addRow("Checkpoint", self.checkpoint_path)
        export_form.addRow("Format", self.output_format)
        root.addWidget(export)

        buttons = QGridLayout()
        self.inspect_button = QPushButton("Inspect Data")
        self.train_button = QPushButton("Run / Train")
        self.stop_button = QPushButton("Stop")
        self.predict_button = QPushButton("Export Predictions")
        self.stop_button.setEnabled(False)
        self.inspect_button.clicked.connect(self.inspect_requested.emit)
        self.train_button.clicked.connect(self.train_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.predict_button.clicked.connect(self.predict_requested.emit)
        buttons.addWidget(self.inspect_button, 0, 0)
        buttons.addWidget(self.train_button, 0, 1)
        buttons.addWidget(self.stop_button, 1, 0)
        buttons.addWidget(self.predict_button, 1, 1)
        root.addLayout(buttons)
        root.addStretch(1)

    def settings(self) -> dict[str, str | int]:
        return {
            "backend_python": self.backend_python.text(),
            "config_path": self.config_path.text(),
            "output_dir": self.output_dir.text(),
            "device": self.device.currentText(),
            "data_root": self.data_root.text(),
            "mask_root": self.mask_root.text(),
            "suffix": self.suffix.currentText(),
            "variable": self.variable.text().strip(),
            "frames": self.frames.value(),
            "image_h": self.image_h.value(),
            "image_w": self.image_w.value(),
            "algorithm": self.algorithm.currentText(),
            "fourier_mode": self.fourier_mode.currentText(),
            "pixel_loss": self.pixel_loss.currentText(),
            "mask_mode": self.mask_mode.currentText(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "checkpoint_path": self.checkpoint_path.text(),
            "output_format": self.output_format.currentText(),
        }

    def set_busy(self, busy: bool) -> None:
        self.inspect_button.setEnabled(not busy)
        self.train_button.setEnabled(not busy)
        self.predict_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
