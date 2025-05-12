import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, dataset_name, batch_size, img_shape, shuffle, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.C = img_shape[0]
        self.H = img_shape[1]
        self.W = img_shape[2]
        self.shuffle = shuffle
        self.device = device

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.H, self.W), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x: (x * 2) - 1)
            ]
        )

    def inverse_transform(self, images):
        return ((images.clamp(-1, 1) + 1.0) / 2.0) * 255.0


    def get_dataset(self):
        if self.dataset_name == "Flowers":
            dataset = datasets.Flowers102(root="ddpm/data", split="train", download=True, transform=self.transforms)
        return dataset

    def get_dataloader(self):
        dataset = self.get_dataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                shuffle=self.shuffle, num_workers=2, pin_memory=True, drop_last=True) 
        return dataloader

# Check if code works
# import matplotlib.pyplot as plt

# dataset_loader = DatasetLoader(dataset_name="Flowers", batch_size=16, img_shape=(3, 64, 64), shuffle=True, device="cuda")
# loader = dataset_loader.get_dataloader()
# plt.figure(figsize=(10, 10))
# for images, labels in loader:
#     images = images.to("cuda")
#     images = dataset_loader.inverse_transform(images)
#     grid_img = torchvision.utils.make_grid(images / 255, nrow=4, padding=2)
#     break
# plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
# plt.savefig('ddpm/grid_img2.png')

