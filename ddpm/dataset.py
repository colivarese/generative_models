import os
import torch
import torchvision
import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass

import subprocess
import sys
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("wandb")

class BaseConfig:
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.get_default_device()
    DATASET = "Flowers"

    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    log_dir = "version_0"
    checkpoint_dir = "version_0"


class TrainingConfig:
    TIMESTEPS = 1000
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else(3, 32, 32)
    NUM_EPOCHS = 800
    LR = 2e-4
    NUM_WORKERS = 2
    BATCH_SIZE = 32

def get_dataset(dataset_name: str = "MNIST"):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32,32), interpolation=TF.InterpolationMode.BICUBIC, 
            antialias=True),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Lambda(lambda x: (x*2) -1)
        ]
    )
    if dataset_name == "MNIST":
        dataset = dataset.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.Flowers102(root="ddpm/data", split="train", download=True, transform=transforms)
    return dataset

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def inverse_transform(tensors):
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

def get_dataloader(dataset_name = "MNIST", batch_size=32, pin_memory=False,
                    shuffle=True, num_workers=0, device="cpu"):
    dataset = get_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
    #return dataloader
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

# Test Viz 
# loader = get_dataloader(dataset_name="Flowers", batch_size=16, device="cuda")

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 8), facecolor='white')
# for b_image, _ in loader:
#     b_image = inverse_transform(b_image)
#     grid_img = make_grid(b_image / 255, nrow=4)
#     break
# plt.imshow(grid_img.permute(1, 2, 0).cpu())
# plt.savefig('ddpm/grid_img.png')
# plt.show()