import torch
import random

from tqdm import tqdm
from torch.cuda.amp import GradScaler
from dataset import COCOLoaderUNEt
from text_encoder import CLIPTextEncoder
from vae import VAE
from unet import UNet
from diffusion import Diffusion
import torch.nn.functional as F


import wandb

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

#wandb.login()
#wandb.init()


text_encoder = CLIPTextEncoder()
coco_dataset =  COCOLoaderUNEt(batch_size=16, img_shape=(3, 256, 256), shuffle=True)
loader = coco_dataset.get_coco_loader()

EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Load VAE
vae = VAE()
vae.to(DEVICE)
vae.load_state_dict(torch.load('vae_best_5.pth'))
vae.eval()
for param in vae.encoder.parameters():
    param.requires_grad = False
#Load UNet
unet = UNet(in_channels=4, out_channels=4, hidden_dim=320, time_emb_dim=64, time_emb_hidden_dim=512, context_dim=77, num_blocks=3, num_resnet_blocks=2, in_channels_mult=(1,2,2,1),
             decoder_channels_mult=(1,2,4,4,), dropout_rate=0.1, device=DEVICE)
unet.to(DEVICE)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)
#optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=0.01)


#Load Diffusion
diffusion = Diffusion(num_steps=1000, img_shape=(3, 256, 256), device=DEVICE)

# Loss fn
#loss_fn = torch.nn.MSELoss(reduction='mean')

# Grad Scaler
scaler = GradScaler()

unet.train()
best_loss = float('inf')
patience = 7

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

counter = 0
for epoch in range(EPOCHS):
    torch.cuda.empty_cache()
    epoch_loss = 0

    for i, batch in enumerate(tqdm(loader, desc="Training...")):

        images, captions = batch
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()


        with torch.no_grad():
            h = vae.encoder(images, None)
            mu, logvar = h.chunk(2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std 
        z = torch.clamp(z, -1.0, 1.0)

        t = torch.randint(0, diffusion.num_steps, (loader.batch_size, ), device=DEVICE)
        sample, noise = diffusion.forward_diffusion(z, t)

        with torch.amp.autocast("cuda"):
            predicted_noise = unet(sample, t, captions)
            #loss = #loss_fn(noise, predicted_noise)
            #loss = F.mse_loss(predicted_noise, noise)
            loss = F.smooth_l1_loss(noise, predicted_noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


        epoch_loss += loss.item()

    mean_loss = epoch_loss / len(loader)

    # wandb.log({
    #     "epoch": epoch,
    #     "mean_loss": mean_loss})

    print(f"Epoch [{epoch+1}/{EPOCHS}], Mean Loss: {mean_loss:.4f}") 
    
    if mean_loss < best_loss:
        best_loss = mean_loss
        counter = 0
        torch.save(unet.state_dict(), f"unet2.pth")  # Save the best model
    else:
        counter += 1
        print(f"Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered.")
            break
