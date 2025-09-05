from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout, 
    QProgressBar, QStyle
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

class ControlPanel(QWidget):
    """Widget for the main control panel dock."""
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.clicked.connect(self.run_clicked.emit)

        self.stop_button = QPushButton("Stop Process")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_clicked.emit)

        self.progress_bar = QProgressBar()
        
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        layout.addSpacing(20)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addStretch()

class ValidationPanel(QWidget):
    """Widget for the validation metrics dock."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        self.rmse_label = QLabel("N/A")
        self.ssim_label = QLabel("N/A")
        self.psnr_label = QLabel("N/A")
        self.r2_label = QLabel("N/A")
        
        layout.addWidget(QLabel("<b>RMSE:</b>"), 0, 0)
        layout.addWidget(self.rmse_label, 0, 1)
        layout.addWidget(QLabel("<b>SSIM:</b>"), 1, 0)
        layout.addWidget(self.ssim_label, 1, 1)
        layout.addWidget(QLabel("<b>PSNR:</b>"), 2, 0)
        layout.addWidget(self.psnr_label, 2, 1)
        layout.addWidget(QLabel("<b>R²:</b>"), 3, 0)
        layout.addWidget(self.r2_label, 3, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(4, 1)
