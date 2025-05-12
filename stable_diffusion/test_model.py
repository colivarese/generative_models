import torch.nn.functional as F
import torch

from vae import VAE
from dataset import COCOLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torchvision

def inverse_transform(tensors):
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

coco_dataset =  COCOLoader(batch_size=16, img_shape=(3, 256, 256), shuffle=True, device="cuda")
loader = coco_dataset.get_coco_loader()


images = next(iter(loader))
images = images.to('cuda')

orig_images = inverse_transform(images)

grid_img = torchvision.utils.make_grid(orig_images / 255, nrow=4, padding=2)
plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
plt.savefig('original_grid_im3.png')

vae = VAE()
vae = vae.to('cuda')

vae.load_state_dict(torch.load('vae_best_5.pth'))

recon, _ ,_ = vae(images)




recon = inverse_transform(recon)

grid_img = torchvision.utils.make_grid(recon / 255, nrow=4, padding=2)
plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
plt.savefig('noisy_grid_img_beta3.png')

