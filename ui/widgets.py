import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout, 
    QProgressBar, QStyle, QGroupBox, QComboBox, QCheckBox, QFileDialog
)
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """A custom matplotlib canvas widget to embed in PyQt5."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f2f5')
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

    
class DataSetupWidget(QGroupBox):
    """Step 1: Data and Preprocessing"""
    def __init__(self):
        super().__init__("Step 1: Data and Preprocessing")
        layout = QVBoxLayout(self)

        self.data_path_label = QLabel("No Selection", objectName="PathLabel")
        self.mask_path_label = QLabel("No Selection (Optional)", objectName="PathLabel")

        data_btn = QPushButton("Select Input Data Folder")
        mask_btn = QPushButton("Select Mask Folder")

        self.log_scale_check = QCheckBox("Apply Log10 Transformation to Chl-a")
        self.log_scale_check.setChecked(True)

        layout.addWidget(QLabel("Input Data Path:"))
        layout.addWidget(self.data_path_label)
        layout.addWidget(data_btn)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Cloud/Land Mask Path:"))
        layout.addWidget(self.mask_path_label)
        layout.addWidget(mask_btn)
        layout.addSpacing(10)
        layout.addWidget(self.log_scale_check)
        
        data_btn.clicked.connect(self.select_data_folder)
        mask_btn.clicked.connect(self.select_mask_folder)

    def select_data_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Data Folder")
        if path: self.data_path_label.setText(path)

    def select_mask_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        if path: self.mask_path_label.setText(path)
        
    def get_data_path(self): return self.data_path_label.text()
    def get_mask_path(self): return self.mask_path_label.text()
    def use_log_scale(self): return self.log_scale_check.isChecked()

class TrainingWidget(QGroupBox):
    """Step 2: Model Training"""
    start_training_signal = pyqtSignal()
    stop_training_signal = pyqtSignal()

    def __init__(self):
        super().__init__("Step 2: Model Training")
        layout = QVBoxLayout(self)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Balanced", "High Quality"])
        self.quality_combo.setToolTip("Choosing a training quality will affect training time and model accuracy.\n"
                                     "Fast: Suitable for quick validation.\n"
                                     "Balanced: Recommended option, balancing speed and accuracy.\n"
                                     "High Quality: Requires more time and computational resources.")

        self.start_button = QPushButton("Start Training")
        self.stop_button = QPushButton("Stop Training")
        self.progress_bar = QProgressBar()

        layout.addWidget(QLabel("Training Quality Preset:"))
        layout.addWidget(self.quality_combo)
        layout.addSpacing(20)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Overall Progress:"))
        layout.addWidget(self.progress_bar)
        
        self.start_button.clicked.connect(self.start_training_signal.emit)
        self.stop_button.clicked.connect(self.stop_training_signal.emit)
        self.set_controls_enabled(True)

    def get_training_quality(self): return self.quality_combo.currentText()
    
    def update_progress(self, value): self.progress_bar.setValue(value)
    
    def set_controls_enabled(self, enabled):
        self.start_button.setEnabled(enabled)
        self.quality_combo.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)

class ResultsWidget(QGroupBox):
    """Step 3: Results and Analysis"""
    def __init__(self):
        super().__init__("Step 3: Results and Analysis")
        self.layout = QGridLayout(self)
        self.metrics_labels = {}

        self.layout.addWidget(QLabel("<b>Training Status:</b>"), 0, 0)
        self.status_label = QLabel("Waiting")
        self.layout.addWidget(self.status_label, 0, 1)

        self.layout.addWidget(QLabel("<b>Model Path:</b>"), 1, 0)
        self.path_label = QLabel("N/A")
        self.layout.addWidget(self.path_label, 1, 1)

        # Additional metrics can be added here
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(2, 1)

    def update_metrics(self, metrics: dict):
        self.status_label.setText(metrics.get("Status", "N/A"))
        self.path_label.setText(metrics.get("Model Path", "N/A"))
        # Update more metrics as needed
