import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from typing import List, Tuple

# --- sequential log transform class ---
class Log10Transform:
    """Applies a safe log10 transformation to a tensor."""
    def __call__(self, tensor):
        # The transformation logic is the same as the lambda
        return torch.log10(torch.clamp(tensor, min=1e-8))

class inpainting_DS(Dataset):
    """PyTorch Dataset for loading sequential frames of satellite data."""
    def __init__(self, root: str, frames: int, shape_scale=(48, 48), is_month:bool=False, 
                 mask_dir:str=None, is_train:bool=True, train_test_ratio:float=0.7, 
                 chunk_list:list=None, mask_chunk_list:list=None, global_mean:float=None,
                 global_std:float=None, apply_log:bool=True, apply_normalize:bool=True):
        super().__init__()
        self.root = root
        self.frames = frames
        self.is_month = is_month
        self.is_train = is_train

        if self.is_month:
            assert mask_dir is not None, "mask_dir must be provided if is_month=True"
            self.mask_dir = mask_dir

        if chunk_list is None:
            full_list = sorted([os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.npy')])
            chunk_list = [full_list[i:i+frames] for i in range(len(full_list)-frames+1)]
            np.random.shuffle(chunk_list)
            split_idx = int(train_test_ratio * len(chunk_list))
            self.chunk_list = chunk_list[:split_idx] if self.is_train else chunk_list[split_idx:]
        else:
            self.chunk_list = chunk_list

        self.mask_chunk = []
        if self.is_month:
            if mask_chunk_list is None:
                full_mask_list = sorted([os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith('.npy')])
                mask_chunk_list = [full_mask_list[i:i+frames] for i in range(len(full_mask_list)-frames+1)]
                # Note: This assumes a 1-to-1 mapping with data chunks after shuffling data.
                # A more robust implementation might map masks by date.
                self.mask_chunk = mask_chunk_list
            else:
                self.mask_chunk = mask_chunk_list

        transform_list = [ToTensor()]
        if apply_log:
            transform_list.append(Log10Transform())
        if apply_normalize and (global_mean is not None and global_std is not None):
            transform_list.append(transforms.Normalize(mean=[global_mean], std=[global_std]))
        self.transform = Compose(transform_list)

    def __len__(self):
        return len(self.chunk_list)

    @staticmethod
    def _apply_mask_logic(data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert mask.shape == data.shape, f"Mask shape {mask.shape} doesn't match data {data.shape}"
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask

    def __getitem__(self, index: int):
        chunk_paths = self.chunk_list[index]
        mask_paths = self.mask_chunk[index] if self.is_month and index < len(self.mask_chunk) else [None] * len(chunk_paths)

        inputs_list, target_list, masks_list = [], [], []
        for data_path, mask_path in zip(chunk_paths, mask_paths):
            array = np.load(data_path)
            array = np.nan_to_num(array, nan=1.0)
            array[array == 0.0] = 1.0
            
            data_tensor = self.transform(array.astype(np.float32))
            
            if self.is_month and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
                in_data, mask_data = self._apply_mask_logic(data_tensor, mask_tensor)
            else:
                in_data, mask_data = self._apply_random_mask(data_tensor)

            inputs_list.append(in_data)
            target_list.append(data_tensor)
            masks_list.append(mask_data)

        return (
            index,
            torch.stack(inputs_list, dim=0).float(),
            torch.stack(target_list, dim=0).float(),
            torch.stack(masks_list, dim=0).float()
        )
