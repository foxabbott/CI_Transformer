from __future__ import annotations
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class UnlabeledImageFolder(ImageFolder):
    """ImageFolder that returns only the image tensor (label ignored)."""
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img

def make_transform(image_size: int, channels: int):
    tfms = [transforms.Resize((image_size, image_size))]
    if channels == 1:
        tfms.append(transforms.Grayscale(num_output_channels=1))
    tfms.append(transforms.ToTensor())
    return transforms.Compose(tfms)

def make_loader(
    data_dir: str,
    image_size: int,
    channels: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = UnlabeledImageFolder(data_dir, transform=make_transform(image_size, channels))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
