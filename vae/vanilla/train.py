from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VanillaVAE
from torch import optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt

def loss_fn(generated_x, x, mu, log_var):
    kld_weight = 0.01
    #recons_loss =F.mse_loss(generated_x, x, reduction='sum')
    recons_loss = F.binary_cross_entropy(generated_x, x, reduction="sum")
    #kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    loss = recons_loss + kld *kld_weight
    return loss

vae = VanillaVAE(in_dim=1, latent_dim=5, hidden_dims=[16, 32])

# Load data
mnist = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist, batch_size=128, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=0.001)

random_latent_vector = torch.randn(16, 5)


# Train
EPOCHS = 100
random_latent_vector = torch.randn(16, 5)
for epoch in range(EPOCHS):
    vae.train()
    train_loss = 0
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        x, y = batch
        #batch = batch.unsqueeze(1)

        optimizer.zero_grad()
        out, mu, log_var = vae(x)

        loss = loss_fn(out, x, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # Print average loss for the epoch
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss / len(train_loader.dataset)}')
        vae.eval()
        generated_images = vae.decode(random_latent_vector)
        generated_images = (generated_images > 0.5).float()
        a = generated_images.max()
        grid = torchvision.utils.make_grid(generated_images, nrow=4)
        #plt.imshow(grid.permute(1, 2, 0))
        #plt.show()
        # Save generated image
        torchvision.utils.save_image(grid, f'vae/generations/generated_image_{epoch}.png')

