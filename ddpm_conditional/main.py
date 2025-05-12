from diffusion import Diffusion
from dataset import DatasetLoader
from model import UNet
#from model2 import Unet
from run import train

import torch
import gc
import matplotlib.pyplot as plt
import torchvision

NUM_STEPS = 1000
BATCH_SIZE = 256
IMG_SHAPE = (1, 32, 32)
IN_CHANNELS = 64
OUT_CHANNELS = 64
TIME_EMB_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25


dataset_loader = DatasetLoader(dataset_name="MNIST", batch_size=BATCH_SIZE, img_shape=IMG_SHAPE, shuffle=True, device="cuda")
diffusion = Diffusion(num_steps=1000, img_shape=IMG_SHAPE, device=DEVICE)
#model = Unet().to(DEVICE)
model = UNet(IN_CHANNELS, OUT_CHANNELS, TIME_EMB_DIM, num_resnet_blocks=2).to(DEVICE)
#print(model)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

loader = dataset_loader.get_dataloader()

for epoch in range(1, EPOCHS):
    torch.cuda.empty_cache()
    gc.collect()
    loss = train(loader, model, diffusion, loss_fn, optimizer)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    if epoch % 1 == 0:
        generated_images = diffusion.reverse_diffusion(num_images=8, model=model)

        images = dataset_loader.inverse_transform(generated_images)
        grid_img = torchvision.utils.make_grid(images / 255, nrow=4, padding=2)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.savefig('ddpm_conditional/out/noisy_grid_img.png')


torch.save(model.state_dict(), 'ddpm_conditional/model.pt')

