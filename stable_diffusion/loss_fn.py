from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], resize=True):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]
        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            self.blocks.append(nn.Sequential(*vgg[prev:l]))
            prev = l
        for param in self.parameters():
            param.requires_grad = False
        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss
    

def vae_loss_function(x, x_hat, mu, logvar, fn, lambda_perceptual, beta =1):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')

        if fn:
            perceptual_loss = fn(x_hat, x)
            recon_loss += lambda_perceptual * perceptual_loss

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss, kl
    

# def vae_loss_function(x, x_hat, mu, logvar, perceptual_fn=None, lambda_perceptual=0.05, beta=0.5):
#     # MSE reconstruction loss
#     recon_loss = F.mse_loss(x_hat, x, reduction='mean')

#     # Optional perceptual loss
#     if perceptual_fn is not None:
#         perceptual_loss = perceptual_fn(x_hat, x)
#         recon_loss += lambda_perceptual * perceptual_loss

#     # KL divergence
#     kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     loss = recon_loss + beta * kl
#     return loss, recon_loss, kl
