import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset import COCOLoaderUNEt
from text_encoder import CLIPTextEncoder
from vae import VAE
from unet import UNet
from diffusion import Diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load text encoder
text_encoder = CLIPTextEncoder()

# Load dataset (for testing)
coco_dataset = COCOLoaderUNEt(batch_size=4, img_shape=(3, 256, 256), shuffle=False)
loader = coco_dataset.get_coco_loader()

# Load VAE
vae = VAE()
vae.to(DEVICE)
vae.load_state_dict(torch.load('vae_best_5.pth'))
vae.eval()

# Load U-Net
unet = UNet(
    in_channels=4,
    out_channels=4,
    hidden_dim=320,
    time_emb_dim=64,
    time_emb_hidden_dim=512,
    context_dim=77,
    num_blocks=3,
    num_resnet_blocks=2,
    in_channels_mult=(1, 2, 2, 1),
    decoder_channels_mult=(1, 2, 4, 4),
    dropout_rate=0.1,
    device=DEVICE
)
unet.to(DEVICE)
unet.load_state_dict(torch.load('unet2.pth'))
unet.eval()

# Load Diffusion
diffusion = Diffusion(num_steps=1000, img_shape=(3, 256, 256), device=DEVICE)

import torchvision

def inverse_transform(tensors):
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

@torch.no_grad()
def inference(save_path='generated_diffusion_image.png'):
    for batch in loader:
        _, captions = batch
        captions = captions.to(DEVICE)

        # Run reverse diffusion to get latent z
        z = diffusion.reverse_diffusion(context=captions, model=unet)

        # Decode with VAE decoder
        recon = vae.decoder(z)

        # Scale from [0, 1] to [0, 255] and save
        recon = inverse_transform(recon)
        grid_img = torchvision.utils.make_grid(recon / 255, nrow=4, padding=2)  # keep it consistent
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.savefig(save_path)
        print(f"Saved image to {save_path}")
        break  # Just one batch for demo

inference('sample_generated_image.png')

