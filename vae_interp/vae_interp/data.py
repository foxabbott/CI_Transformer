
from __future__ import annotations

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vae.vae.data import make_transform

class UnlabeledImageFolder(ImageFolder):
    def __getitem__(self, index):
        x, _ = super().__getitem__(index)
        return x



def make_loader(data_dir: str, image_size: int, batch_size: int, num_workers: int = 0, channels: int = 3) -> DataLoader:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    ds = UnlabeledImageFolder(data_dir, transform=make_transform(image_size, channels))
    if len(ds) == 0:
        raise RuntimeError(
            f"No images found under {data_dir}. ImageFolder expects at least one class subfolder containing images."
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
