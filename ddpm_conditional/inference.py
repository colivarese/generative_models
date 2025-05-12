import torch
from diffusion import Diffusion
from model import UNet
from dataset import DatasetLoader
from tqdm import tqdm
import torchvision.transforms as TF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import torchvision.transforms as TF
from torchvision.utils import make_grid
import time

IMG_SHAPE = (3, 32, 32)
IN_CHANNELS = 64
OUT_CHANNELS = 64
TIME_EMB_DIM = 256
DEVICE = "cuda"
BATCH_SIZE = 32

sd = Diffusion(num_steps=1000, img_shape=(1, 32, 32), device=DEVICE)
model = UNet(IN_CHANNELS, OUT_CHANNELS, TIME_EMB_DIM, num_resnet_blocks=2).to(DEVICE)
dataset_loader = DatasetLoader(dataset_name="MNIST", batch_size=BATCH_SIZE, img_shape=(1, 32, 32), shuffle=True, device="cuda")

model.load_state_dict(torch.load('ddpm_conditional/model.pt'))

# start_time = time.time()
# generated_images = sd.reverse_diffusion_ddim(num_steps = 100, num_images=8, model=model)
# ddim_time = time.time() - start_time

generated_images = sd.reverse_diffusion(num_images=8, model=model)
#reverse_diffusion_time = time.time() - start_time

#print("Time taken for ddim:", ddim_time)
#print("Time taken for reverse diffusion:", reverse_diffusion_time)

# generated_images = sd.reverse_diffusion(num_images=8, model=model)
images = dataset_loader.inverse_transform(generated_images)
grid_img = make_grid(images / 255, nrow=4, padding=2)
plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
plt.savefig('ddpm_conditional/out/ddpm_inference_img.png')

# generated_images = sd.reverse_diffusion_ddim(num_steps = 650, num_images=8, model=model)
# images = dataset_loader.inverse_transform(generated_images)
# grid_img = make_grid(images / 255, nrow=4, padding=2)
# plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
# plt.savefig('ddpm_conditional/out/noisy_grid_img2.png')