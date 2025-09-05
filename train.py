import os
import yaml
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

from core.dataset import inpainting_DS
from core.model import ED, Encoder, Decoder, ConvLSTM_cell
from core.utils import SSIM, EarlyStopping

def record_dir_setting_create(father_dir: str, mark: str) -> str:
    """Creates a directory for saving checkpoints."""
    dir_temp = os.path.join(father_dir, mark)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    return dir_temp

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Setup ---
    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["random_seed"])
    
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    save_dir = record_dir_setting_create(cfg["ckpts_dir"], cfg["mark"])

    # --- Neptune Integration (Optional) ---
    # To enable, uncomment the following lines and add your API token.
    # run = neptune.init_run(
    #     api_token="YOUR_NEPTUNE_API_TOKEN",
    #     project="your-user/your-project",
    # )
    # npt_logger = NeptuneLogger(run, model=model) # model must be defined first
    # run[npt_logger.base_namespace]["hyperparameters"] = stringify_unsupported(cfg)

    # --- Data Loaders ---
    train_dataset = inpainting_DS(cfg["root_dir"], cfg["frames"], cfg["shape_scale"], 
                                  is_month=cfg["is_month"], mask_dir=cfg.get("mask_dir"), is_train=True)
    valid_dataset = inpainting_DS(cfg["root_dir"], cfg["frames"], cfg["shape_scale"], 
                                  is_month=cfg["is_month"], mask_dir=cfg.get("mask_dir"), is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH"], shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["BATCH"], shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Definition ---
    h, w = cfg["shape_scale"]
    is_cuda = (device.type == 'cuda')
    
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

    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
    model = ED(encoder, decoder).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg["LR"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=14, verbose=True)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # --- Loss Functions ---
    ssmi_loss_func = SSIM().to(device)
    mse_loss_func = nn.MSELoss().to(device)
    beta = cfg["beta"]

    # --- Training Loop ---
    for epoch in range(cfg["EPOCH"]):
        model.train()
        train_losses = []
        t = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['EPOCH']}")
        for _, inputs, targets, _ in t:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            
            mse_loss = mse_loss_func(pred, targets)
            ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])
            loss = beta * mse_loss + (1 - beta) * (1 - ssmi_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            t.set_postfix(loss=np.mean(train_losses))

        # --- Validation Loop ---
        if (epoch + 1) % 2 == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for _, inputs, targets, _ in tqdm(valid_loader, desc="Validation", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    pred = model(inputs)
                    mse_loss = mse_loss_func(pred, targets)
                    ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])
                    loss = beta * mse_loss + (1 - beta) * (1 - ssmi_loss)
                    valid_losses.append(loss.item())

            mean_valid_loss = np.mean(valid_losses)
            print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.6f} | Valid Loss: {mean_valid_loss:.6f}")
            scheduler.step(mean_valid_loss)
            
            model_state = {
                'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()
            }
            early_stopping(mean_valid_loss, model_state, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    # if 'run' in locals(): run.stop()

if __name__ == "__main__":
    main(config_path="config.yml")
