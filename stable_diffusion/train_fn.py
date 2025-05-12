import torch

from vae import Encoder, Decoder
from diffusion import Diffusion
from text_encoder import CLIPTextEncoder
from unet import UNet

BATCH_SIZE = 1

diffusion = Diffusion(num_steps=1000, img_shape=(3, 256, 256), device="cuda")
encoder = Encoder(in_channels=3, hidden_dim=128, num_downsample_blocks=4, num_bottleneck_blocks=2)
decoder = Decoder(in_channels=4, out_channels=3, hidden_dim=512, num_upsample_blocks=4, num_resnet_blocks=3)
text_encoder = CLIPTextEncoder()
unet = UNet(in_channels=32, out_channels=3, hidden_dim=512, time_emb_dim=64, time_emb_hidden_dim=512, context_dim=77, num_blocks=4, num_resnet_blocks=2, in_channels_mult=(1,2,2,1),
            decoder_channels_mult=(1,2,4,4,), dropout_rate=0.1, device="cuda")
unet = unet.to("cuda")


fake_image = torch.randn(BATCH_SIZE, 3, 256, 256)
t = torch.randint(0, 1000, (BATCH_SIZE, ), device="cuda")
prompt = "A beautiful landscape with mountains and a river"

h = encoder(fake_image, None)
text_embeddings = text_encoder(prompt)

mu, logvar = h.chunk(2, dim=1)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
z = z.to("cuda")

# Add noise 
z = diffusion.forward_diffusion(z, t)

# Predict noise with UNet
noise_pred = unet(z, t, text_embeddings)
a = 5
