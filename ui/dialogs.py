from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, 
    QLineEdit, QComboBox, QCheckBox, QPushButton, QFileDialog
)

class SettingsDialog(QDialog):
    """A dialog window for configuring the application settings."""
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)
        self.layout = QVBoxLayout(self)

        self.input_path_label = QLabel()
        self.output_path_label = QLabel()
        input_btn = QPushButton("Select Input Folder")
        output_btn = QPushButton("Select Output Folder")
        self.temporal_window_input = QLineEdit()
        self.model_weights_combo = QComboBox()
        self.model_weights_combo.addItems(["Default (FTC-LSTM_v1.pt)", "Experimental (FTC-LSTM_v2.pt)"])
        self.log_scale_checkbox = QCheckBox("Use Log-Scale for Chl-a")
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        
        if current_settings:
            self.input_path_label.setText(current_settings.get("input_path", "Not selected"))
            self.output_path_label.setText(current_settings.get("output_path", "Not selected"))
            self.temporal_window_input.setText(current_settings.get("temporal_window", "10"))
            self.model_weights_combo.setCurrentText(current_settings.get("model_weights", ""))
            self.log_scale_checkbox.setChecked(current_settings.get("use_log_scale", True))
            self.gpu_checkbox.setChecked(current_settings.get("use_gpu", True))

        path_layout = QGridLayout()
        path_layout.addWidget(QLabel("Input Data Path:"), 0, 0)
        path_layout.addWidget(self.input_path_label, 0, 1)
        path_layout.addWidget(input_btn, 0, 2)
        path_layout.addWidget(QLabel("Output Data Path:"), 1, 0)
        path_layout.addWidget(self.output_path_label, 1, 1)
        path_layout.addWidget(output_btn, 1, 2)
        self.layout.addLayout(path_layout)
        
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Temporal Window (T):"), 0, 0)
        param_layout.addWidget(self.temporal_window_input, 0, 1)
        param_layout.addWidget(QLabel("Model Weights:"), 1, 0)
        param_layout.addWidget(self.model_weights_combo, 1, 1)
        param_layout.addWidget(self.log_scale_checkbox, 2, 0, 1, 2)
        param_layout.addWidget(self.gpu_checkbox, 3, 0, 1, 2)
        self.layout.addLayout(param_layout)

        button_layout = QHBoxLayout()
        ok_button, cancel_button = QPushButton("OK"), QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        self.layout.addLayout(button_layout)

        input_btn.clicked.connect(self.select_input_folder)
        output_btn.clicked.connect(self.select_output_folder)
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path: self.input_path_label.setText(path)

    def select_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path: self.output_path_label.setText(path)

    def get_settings(self):
        return {
            "input_path": self.input_path_label.text(),
            "output_path": self.output_path_label.text(),
            "temporal_window": self.temporal_window_input.text(),
            "model_weights": self.model_weights_combo.currentText(),
            "use_log_scale": self.log_scale_checkbox.isChecked(),
            "use_gpu": self.gpu_checkbox.isChecked()
        }
