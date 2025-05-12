import torch
from torch.cuda.amp import autocast, GradScaler
from dataset import COCOLoader
import torch.nn.functional as F

from vae import VAE
from tqdm import tqdm
from loss_fn import vae_loss_function, VGGPerceptualLoss

import wandb

wandb.login()

wandb.init()

vae = VAE()
vae = vae.to("cuda")

optimizer = torch.optim.Adam(vae.parameters(), lr=2e-4)
#optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)

scaler = GradScaler()

coco_dataset =  COCOLoader(batch_size=16, img_shape=(3, 256, 256), shuffle=True, device="cuda")
loader = coco_dataset.get_coco_loader()

# def vae_loss_function(x, x_hat, mu, logvar):
#         recon_loss = F.mse_loss(x_hat, x, reduction='sum')
#         # KL divergence
#         kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return recon_loss + 1 * kl, recon_loss, kl

EPOCHS = 280

perceptual_loss_fn = VGGPerceptualLoss().to('cuda')

vae.train()
best_loss = float('inf')
patience = 7
counter = 0

for epoch in range(EPOCHS):
    torch.cuda.empty_cache()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_div = 0
    for i, images in enumerate(tqdm(loader, desc="Training Progress")):
        
        images = images.to("cuda")
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            recon_images, mu, logvar = vae(images)
            #loss, recon_loss, kl_div = vae_loss_function(recon_images, images, mu, logvar, perceptual_fn=perceptual_loss_fn, lambda_perceptual=0.0)
            loss, recon_loss, kl_div = vae_loss_function(recon_images, images, mu, logvar, fn=perceptual_loss_fn, lambda_perceptual=5, beta=0.5)
        scaler.scale(loss).backward()       # backward pass with scaled loss
        scaler.step(optimizer)              # optimizer step
        scaler.update()
        
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_div += kl_div.item()
    
    mean_loss = epoch_loss / len(loader)
    mean_recon_loss = epoch_recon_loss / len(loader)
    mean_kl_div = epoch_kl_div / len(loader)

    wandb.log({"epoch":epoch,"mean_loss": mean_loss, "mean_recon_loss": mean_recon_loss, "mean_kl_div":mean_kl_div})

    print(f"Epoch [{epoch+1}/{EPOCHS}], Mean Loss: {mean_loss:.4f}, Mean Recon Loss: {mean_recon_loss:.4f}, Mean KL Div: {mean_kl_div:.4f}") 
    
    # Early stopping logic
    if mean_loss < best_loss:
        best_loss = mean_loss
        counter = 0
        torch.save(vae.state_dict(), f"vae_best_5.pth")  # Save the best model
    else:
        counter += 1
        print(f"Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered.")
            break

torch.save(vae.state_dict(), f"vae_best_5.pth")    
