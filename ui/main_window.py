import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QTabWidget, QTextEdit, QStatusBar, QFrame
)
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont

# import new workflow widgets and Matplotlib canvas
from ui.widgets import DataSetupWidget, TrainingWidget, ResultsWidget, MplCanvas
from core.training_worker import TrainingWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarineSIR - Intelligent Marine Image Reconstruction Platform")
        self.setGeometry(100, 100, 1600, 900)
        self.thread = None
        self.worker = None

        self.setup_ui()
        self.apply_stylesheet()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Left Control Panel (Workflow Panel) ---
        left_panel = QFrame(self)
        left_panel.setObjectName("SidePanel")
        left_panel.setFixedWidth(350)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(20)

        # Step 1: Data Preparation
        self.data_widget = DataSetupWidget()
        # Step 2: Model Training
        self.training_widget = TrainingWidget()
        # Step 3: Results and Analysis
        self.results_widget = ResultsWidget()

        left_panel_layout.addWidget(self.data_widget)
        left_panel_layout.addWidget(self.training_widget)
        left_panel_layout.addWidget(self.results_widget)
        left_panel_layout.addStretch()

        # --- Right Content Area (Content Area) ---
        right_tabs = QTabWidget()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        self.map_canvas = MplCanvas()
        self.map_canvas.axes.set_title("Reconstruction Result")
        self.map_canvas.axes.set_xlabel("Longitude")
        self.map_canvas.axes.set_ylabel("Latitude")

        self.training_plot_canvas = MplCanvas()
        self.training_plot_canvas.axes.set_title("Real-time Training Curve")
        self.training_plot_canvas.axes.set_xlabel("Epoch")
        self.training_plot_canvas.axes.set_ylabel("Loss")
        self.training_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)

        right_tabs.addTab(self.map_canvas, "Reconstruction")
        right_tabs.addTab(self.training_plot_canvas, "Training Curve")
        right_tabs.addTab(self.log_text_edit, "Log")

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_tabs, 1) # Give more stretch space

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please start from Step 1.")

        # --- Connect signals and slots ---
        self.training_widget.start_training_signal.connect(self.start_training)
        self.training_widget.stop_training_signal.connect(self.stop_training)

    def start_training(self):
        settings = {
            'data_path': self.data_widget.get_data_path(),
            'mask_path': self.data_widget.get_mask_path(),
            'use_log_scale': self.data_widget.use_log_scale(),
            'training_quality': self.training_widget.get_training_quality()
        }
        
        if not os.path.isdir(settings['data_path']):
            self.log_message("Error: Invalid input data path.")
            self.status_bar.showMessage("Error: Invalid input data path.")
            return

        self.training_widget.set_controls_enabled(False)
        self.status_bar.showMessage("Initializing training...")

        # Clear previous training plot
        self.training_plot_canvas.axes.clear()
        self.training_plot_canvas.axes.set_title("Real-time Training Curve")
        self.training_plot_canvas.axes.set_xlabel("Epoch")
        self.training_plot_canvas.axes.set_ylabel("Loss")
        self.training_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.train_loss_line, = self.training_plot_canvas.axes.plot([], [], 'o-', label='Train Loss')
        self.val_loss_line, = self.training_plot_canvas.axes.plot([], [], 'o-', label='Validation Loss')
        self.training_plot_canvas.axes.legend()

        # Start background training thread
        self.thread = QThread()
        self.worker = TrainingWorker(settings)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.log_message)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.epoch_finished.connect(self.update_training_plot)
        self.worker.training_finished.connect(self.on_training_finished)
        
        self.worker.training_finished.connect(self.thread.quit)
        self.worker.training_finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_message("Sending stop signal...")
            self.training_widget.set_controls_enabled(True)
            self.status_bar.showMessage("Training stopped.")

    def log_message(self, msg):
        self.log_text_edit.append(msg)

    def update_progress(self, value, text):
        self.training_widget.update_progress(value)
        self.status_bar.showMessage(text)

    def update_training_plot(self, epoch, train_loss, val_loss):
        # Update training curve plot
        x_data = list(self.train_loss_line.get_xdata()) + [epoch]
        y_train_data = list(self.train_loss_line.get_ydata()) + [train_loss]
        y_val_data = list(self.val_loss_line.get_ydata()) + [val_loss]

        self.train_loss_line.set_data(x_data, y_train_data)
        self.val_loss_line.set_data(x_data, y_val_data)

        self.training_plot_canvas.axes.relim()
        self.training_plot_canvas.axes.autoscale_view()
        self.training_plot_canvas.draw()
        
    def on_training_finished(self, model_path):
        self.training_widget.set_controls_enabled(True)
        if model_path:
            self.status_bar.showMessage(f"Training completed! Model saved.")
            # Automatically load the model and perform inference/display example results here
            self.results_widget.update_metrics({"Status": "Training successful", "Model Path": model_path})
            # Example: Display a randomly generated "result map"
            dummy_map = np.random.rand(48, 48)
            self.map_canvas.axes.clear()
            im = self.map_canvas.axes.imshow(dummy_map, cmap='viridis')
            self.map_canvas.figure.colorbar(im, ax=self.map_canvas.axes)
            self.map_canvas.draw()
        else:
            self.status_bar.showMessage("Training failed, please check the log.")
            self.results_widget.update_metrics({"Status": "Training failed"})


    def apply_stylesheet(self):
        try:
            with open("assets/style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: style.qss not found, using default style.")
