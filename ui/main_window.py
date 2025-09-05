import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QTextEdit, QToolBar, QAction, 
    QStatusBar, QStyle
)
from PyQt5.QtCore import Qt, QThread, QSize

from ui.widgets import MplCanvas, ControlPanel, ValidationPanel
from ui.dialogs import SettingsDialog
from core.inference import InferenceWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarineSIR")
        self.setGeometry(100, 100, 1400, 900)
        self.thread = None
        self.worker = None
        self.settings = {
            "input_path": "Not selected", "output_path": "Not selected",
            "temporal_window": "10", "model_weights": "Default (FTC-LSTM_v1.pt)",
            "use_log_scale": True, "use_gpu": True
        }

        self.map_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.map_canvas.axes.set_xlabel("Longitude")
        self.map_canvas.axes.set_ylabel("Latitude")
        self.setCentralWidget(self.map_canvas)
        self.map_canvas.mpl_connect('button_press_event', self.on_map_click)

        self.setup_ui()
        self.apply_stylesheet()
        
    def setup_ui(self):
        self.create_docks()
        self.create_actions()
        self.create_toolbar()
        self.create_menu_bar()
        self.create_status_bar()
        self.connect_signals()

    def create_docks(self):
        control_dock = QDockWidget("Control Panel", self)
        self.control_panel = ControlPanel()
        control_dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)

        ts_dock = QDockWidget("Time Series Plot", self)
        self.ts_canvas = MplCanvas(self)
        self.ts_canvas.axes.set_title("Time Series at Point")
        self.ts_canvas.axes.grid(True)
        ts_dock.setWidget(self.ts_canvas)
        self.addDockWidget(Qt.RightDockWidgetArea, ts_dock)

        validation_dock = QDockWidget("Validation Metrics", self)
        self.validation_panel = ValidationPanel()
        validation_dock.setWidget(self.validation_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, validation_dock)
        
        log_dock = QDockWidget("Logs", self)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_dock.setWidget(self.log_text_edit)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        
        self.tabifyDockWidget(validation_dock, log_dock)
        validation_dock.raise_()

    def create_actions(self):
        style = self.style()
        self.open_action = QAction(style.standardIcon(QStyle.SP_DirOpenIcon), "&Open Data...", self)
        self.save_action = QAction(style.standardIcon(QStyle.SP_DriveHDIcon), "&Save Map...", self)
        self.settings_action = QAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "&Settings...", self)
        self.exit_action = QAction("&Exit", self)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.settings_action)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        view_menu = menu_bar.addMenu("&View")
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def connect_signals(self):
        self.control_panel.run_clicked.connect(self.start_reconstruction)
        self.control_panel.stop_clicked.connect(self.stop_reconstruction)
        self.settings_action.triggered.connect(self.open_settings_dialog)
        self.exit_action.triggered.connect(self.close)

    def open_settings_dialog(self):
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec_():
            self.settings = dialog.get_settings()
            self.update_log(f"Settings updated: {self.settings}")

    def start_reconstruction(self):
        self.control_panel.run_button.setEnabled(False)
        self.control_panel.stop_button.setEnabled(True)
        self.status_bar.showMessage("Processing...")
        
        self.thread = QThread()
        self.worker = InferenceWorker(self.settings)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_reconstruction_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(self.update_log)
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def stop_reconstruction(self):
        if self.worker: self.worker.stop()
        self.control_panel.stop_button.setEnabled(False)
        self.control_panel.run_button.setEnabled(True)
        self.status_bar.showMessage("Process stopped by user.")

    def on_reconstruction_finished(self, map_data, metrics):
        self.update_log("Process finished successfully.")
        self.control_panel.run_button.setEnabled(True)
        self.control_panel.stop_button.setEnabled(False)
        self.status_bar.showMessage("Ready")
        self.display_map(map_data)
        self.update_validation_metrics(metrics)

    def update_progress(self, value, text):
        self.control_panel.progress_bar.setValue(value)
        self.status_bar.showMessage(text)

    def update_log(self, message):
        self.log_text_edit.append(message)

    def update_validation_metrics(self, metrics):
        self.validation_panel.rmse_label.setText(metrics.get("RMSE", "N/A"))
        self.validation_panel.ssim_label.setText(metrics.get("SSIM", "N/A"))
        self.validation_panel.psnr_label.setText(metrics.get("PSNR", "N/A"))
        self.validation_panel.r2_label.setText(metrics.get("R2", "N/A"))

    def display_map(self, data):
        self.map_canvas.axes.clear()
        im = self.map_canvas.axes.imshow(data, cmap='viridis', interpolation='nearest')
        self.map_canvas.figure.colorbar(im, ax=self.map_canvas.axes, shrink=0.6)
        self.map_canvas.axes.set_title("Reconstructed Chlorophyll-a (mg/m³)")
        self.map_canvas.axes.set_xlabel("Longitude")
        self.map_canvas.axes.set_ylabel("Latitude")
        self.map_canvas.draw()

    def on_map_click(self, event):
        if event.inaxes != self.map_canvas.axes: return
        x, y = int(event.xdata), int(event.ydata)
        self.update_log(f"Map clicked at pixel ({x}, {y}). Generating time series...")
        
        time_steps = np.arange(0, 30)
        ts_data = 5 + np.random.randn(30).cumsum()
        
        self.ts_canvas.axes.clear()
        self.ts_canvas.axes.plot(time_steps, ts_data, marker='o', linestyle='-')
        self.ts_canvas.axes.set_title(f"Time Series at ({x}, {y})")
        self.ts_canvas.axes.set_xlabel("Time Step")
        self.ts_canvas.axes.set_ylabel("Chl-a Value")
        self.ts_canvas.axes.grid(True)
        self.ts_canvas.figure.tight_layout()
        self.ts_canvas.draw()

    def apply_stylesheet(self):
        try:
            with open("assets/style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Stylesheet file not found. Using default styles.")
