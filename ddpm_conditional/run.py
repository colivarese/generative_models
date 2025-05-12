import torch

from torch.cuda import amp
from tqdm import tqdm

scaler = torch.amp.GradScaler("cuda")

def train(loader, model, diffusion, loss_fn, optimizer):

    model.train()
    for images, labels in tqdm(loader):
        # Load images to device
        images = images.to(diffusion.device)
        # Convert labels to one-hot encoding
        #labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10)
        labels = labels.to(diffusion.device)

        # Create timestamp tensor
        t = torch.randint(0, diffusion.num_steps, (loader.batch_size, ), device=diffusion.device)

        # Add noise to images with forward diffusion
        noisy_images, noise = diffusion.forward_difussion(images, t)

        with torch.amp.autocast('cuda'):
            # Predict noise with model
            predicted_noise = model(noisy_images, t, labels)
            # Calculate loss 
            loss = loss_fn(noise, predicted_noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # Backpropagation
        #loss.backward()
        #optimizer.step()

    return loss.item()