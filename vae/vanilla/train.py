from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VanillaVAE
from torch import optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

def loss_fn(generated_x, x, mu, log_var):
    kld_weight = 1
    #recons_loss =F.mse_loss(generated_x, x, reduction='sum')
    #recons_loss = F.binary_cross_entropy(generated_x, x, reduction="sum")
    recons_loss = nn.BCELoss(reduction="sum")(generated_x, x)
    #kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    loss = recons_loss + kld *kld_weight
    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')

LATENT_DIM = 5

vae = VanillaVAE(in_dim=1, latent_dim=LATENT_DIM, hidden_dims=[16, 32])
vae.compile()
vae = vae.to(device)

# Load data
mnist = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist, batch_size=128, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=0.001)
random_latent_vector = torch.randn(16, LATENT_DIM).to(device)

# Train
EPOCHS = 20
for epoch in tqdm(range(EPOCHS)):
    vae.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):#tqdm(enumerate(train_loader), total=len(train_loader)):
        x, y = batch
        x = x.to(device)

        optimizer.zero_grad()
        out, mu, log_var = vae(x)

        loss = loss_fn(out, x, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss / len(train_loader.dataset)}')
    vae.eval()
    generated_images = vae.decode(random_latent_vector)
    #bin_generated_images = (generated_images > 0.25).float()
    #bin_grid = torchvision.utils.make_grid(bin_generated_images, nrow=4)
    grid = torchvision.utils.make_grid(generated_images, nrow=4)
    #torchvision.utils.save_image(bin_grid, f'vae/vanilla/generations/bin_generated_image_{epoch}.png')
    torchvision.utils.save_image(grid, f'vae/vanilla/generations/generated_image_{epoch}.png')
    
    # # Print average loss for the epoch
    # if epoch % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss / len(train_loader.dataset)}')
    #     vae.eval()
    #     generated_images = vae.decode(random_latent_vector)
    #     #generated_images = (generated_images > 0.5).float()
    #     grid = torchvision.utils.make_grid(generated_images, nrow=4)
    #     torchvision.utils.save_image(grid, f'vae/vanilla/generations/generated_image_{epoch}.png')

torch.save(vae, 'mnist_vae_full.pth')
