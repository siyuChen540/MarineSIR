import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import r2_score

class InferenceWorker(QObject):
    """
    Handles the long-running reconstruction task in a separate thread.
    This class is designed to be used by the GUI.
    """
    finished = pyqtSignal(np.ndarray, dict)
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._is_running = True

    def run(self):
        """
        Simulates the FTC-LSTM batch reconstruction process.
        In a real application, this is where you would:
        1. Load the pre-trained model weights.
        2. Create a DataLoader for the input data specified in settings.
        3. Loop through the data, perform model inference.
        4. Stitch the results back together into a final map.
        5. Calculate validation metrics against ground truth if available.
        """
        self.log_message.emit("Starting reconstruction process...")
        self.log_message.emit(f"Settings: {self.settings}")
        
        # --- Placeholder for real model inference ---
        # 1. Load Model
        self.log_message.emit(f"Loading model weights from: {self.settings.get('model_weights')}")
        # device = "cuda" if self.settings.get('use_gpu') and torch.cuda.is_available() else "cpu"
        # model = ED(encoder, decoder).to(device)
        # model.load_state_dict(torch.load(self.settings.get('model_weights')))
        # model.eval()

        # 2. Load Data
        self.log_message.emit(f"Loading data from: {self.settings.get('input_path')}")
        # dataset = ...
        # dataloader = ...

        # 3. Simulation Loop
        total_steps = 100
        for i in range(total_steps + 1):
            if not self._is_running:
                self.log_message.emit("Process was stopped by the user.")
                break
            
            # This simulates one batch of inference
            time.sleep(0.05)
            
            progress_percent = int((i / total_steps) * 100)
            self.progress.emit(progress_percent, f"Reconstructing batch {i+1}/{total_steps}...")

            if i % 10 == 0:
                self.log_message.emit(f"  -> Completed batch {i}")
        
        # --- End of Placeholder ---

        if self._is_running:
            self.log_message.emit("Reconstruction finished. Generating map and metrics...")
            
            # 4. Generate final map and metrics (using simulated data)
            ground_truth = np.random.rand(100, 100) * 15
            reconstructed_map = ground_truth + np.random.normal(0, 0.5, (100,100))
            reconstructed_map = np.clip(reconstructed_map, 0, 15)

            # 5. Calculate Metrics
            rmse = np.sqrt(np.mean((ground_truth - reconstructed_map) ** 2))
            ssim_val = structural_similarity(ground_truth, reconstructed_map, data_range=ground_truth.max() - ground_truth.min())
            psnr_val = peak_signal_noise_ratio(ground_truth, reconstructed_map, data_range=ground_truth.max() - ground_truth.min())
            r2_val = r2_score(ground_truth.flatten(), reconstructed_map.flatten())

            validation_metrics = {
                "RMSE": f"{rmse:.4f}",
                "SSIM": f"{ssim_val:.4f}",
                "PSNR": f"{psnr_val:.2f} dB",
                "R2": f"{r2_val:.4f}"
            }
            self.finished.emit(reconstructed_map, validation_metrics)

    def stop(self):
        self._is_running = False
