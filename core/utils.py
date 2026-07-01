import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if val_range is None:
        L = 255 if torch.max(img1) > 128 else 1
    else:
        L = val_range

    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2
    
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model_state, epoch, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model_state, epoch, save_path):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model_state, os.path.join(save_path, f"checkpoint_{epoch}_{val_loss:.6f}.pth.tar"))
        self.val_loss_min = val_loss
