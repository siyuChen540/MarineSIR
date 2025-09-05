import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from PyQt5.QtCore import QObject, pyqtSignal

from core.dataset import inpainting_DS
from core.model import ED, Encoder, Decoder, ConvLSTM_cell
from core.utils import SSIM

class TrainingWorker(QObject):
    """
    Handles the long-running training task in a separate thread.
    """
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    epoch_finished = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    training_finished = pyqtSignal(str)             # path to saved model

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._is_running = True

    def run(self):
        try:
            self.log_message.emit("Starting training process...")
            self.log_message.emit(f"Settings: {self.settings}")

            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"Using device: {device.type}")

            # Abstract hyperparameters
            quality = self.settings['training_quality']
            # Map simple user options to complex hyperparameters
            h_params = {
                "Quick": {"epochs": 20, "lr": 0.001, "batch_size": 16},
                "Balanced": {"epochs": 50, "lr": 0.0005, "batch_size": 8},
                "High Quality": {"epochs": 100, "lr": 0.0001, "batch_size": 4}
            }[quality]

            epochs = h_params['epochs']

            # Prepare dataset
            self.log_message.emit("Loading and preprocessing data...")
            # Note: The shape_scale here should be fixed or automatically obtained from the data
            shape_scale = (48, 48)
            train_ds = inpainting_DS(root=self.settings['data_path'], frames=10, shape_scale=shape_scale, is_train=True)
            val_ds = inpainting_DS(root=self.settings['data_path'], frames=10, shape_scale=shape_scale, is_train=False)
            train_loader = DataLoader(train_ds, batch_size=h_params['batch_size'], shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=h_params['batch_size'], shuffle=False, num_workers=2)
            self.log_message.emit(f"Data loading complete. Training set: {len(train_ds)} samples, Validation set: {len(val_ds)} samples.")

            # Build model
            self.log_message.emit("Building FTC-LSTM model...")
            from collections import OrderedDict
            h, w = shape_scale
            is_cuda = (device.type == 'cuda')

            # --- Full model parameters ---
            encoder_params = [
                [
                    OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                    OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
                    OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
                ],
                [   
                    ConvLSTM_cell(shape=(h,w), channels=16, kernel_size=5, features_num=64, is_cuda=is_cuda),
                    ConvLSTM_cell(shape=(h//2,w//2), channels=64, kernel_size=5, features_num=96, is_cuda=is_cuda),
                    ConvLSTM_cell(shape=(h//4,w//4), channels=96, kernel_size=5, features_num=96, is_cuda=is_cuda)
                ]
            ]

            decoder_params = [
                [
                    OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
                    OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
                    OrderedDict({ 'conv3_leaky_1': [64, 16, 3, 1, 1], 'conv4_leaky_1': [16, 1, 1, 1, 0] }),
                ],
                [
                    ConvLSTM_cell(shape=(h//4,w//4), channels=96, kernel_size=5, features_num=96, is_cuda=is_cuda),
                    ConvLSTM_cell(shape=(h//2,w//2), channels=96, kernel_size=5, features_num=96, is_cuda=is_cuda),
                    ConvLSTM_cell(shape=(h,w), channels=96, kernel_size=5, features_num=64, is_cuda=is_cuda),
                ]
            ]
            # ---------------------------------------------

            # --- Use the correct parameters to instantiate the model ---
            try:
                encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
                decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
                model = ED(encoder, decoder).to(device)
                self.log_message.emit("Model construction successful!")
            except Exception as e:
                self.log_message.emit(f"Model construction failed: {e}")
                self.training_finished.emit("") # Send empty signal to indicate failure
                return
            # --------------------------------

            optimizer = optim.Adam(model.parameters(), lr=h_params['lr'])
            ssim_loss_func = SSIM().to(device)
            mse_loss_func = nn.MSELoss().to(device)

            # Training loop
            for epoch in range(epochs):
                if not self._is_running:
                    self.log_message.emit("User manually stopped training.")
                    break

                model.train()
                train_losses = []
                total_steps = len(train_loader)
                for i, (_, inputs, targets, _) in enumerate(train_loader):
                    if not self._is_running: break
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    pred = model(inputs)
                    mse_loss = mse_loss_func(pred, targets)
                    ssim_loss = 1 - ssim_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])
                    loss = 0.5 * mse_loss + 0.5 * ssim_loss
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                    # Update progress bar
                    progress = int(((epoch * total_steps + i + 1) / (epochs * total_steps)) * 100)
                    status_text = f"Epoch {epoch+1}/{epochs}, Step {i+1}/{total_steps}"
                    self.progress_updated.emit(progress, status_text)

                avg_train_loss = np.mean(train_losses)

                # Validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _, inputs, targets, _ in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        pred = model(inputs)
                        mse_loss = mse_loss_func(pred, targets)
                        ssim_loss = 1 - ssim_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])
                        loss = 0.5 * mse_loss + 0.5 * ssim_loss
                        val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)

                self.log_message.emit(f"Epoch {epoch+1} complete | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
                self.epoch_finished.emit(epoch + 1, avg_train_loss, avg_val_loss)

            if self._is_running:
                # Save model and finish
                save_path = f"models/ftclstm_{quality}_{int(time.time())}.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), save_path)
                self.log_message.emit(f"Training complete! Model saved to: {save_path}")
                self.training_finished.emit(save_path)

        except Exception as e:
            self.log_message.emit(f"Error occurred: {e}")
            self.training_finished.emit("") # Send empty path to indicate failure

    def stop(self):
        self._is_running = False
