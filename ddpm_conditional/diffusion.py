import torch

from dataset import DatasetLoader
from tqdm import tqdm
import torchvision.transforms as TF
from torchvision.utils import make_grid

import torchvision.transforms as TF
from torchvision.utils import make_grid

class Diffusion:
    def __init__(self, num_steps, img_shape, device):
        self.num_steps = num_steps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        self.beta = self.get_betas()
        self.alpha = 1. - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cum = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1 - self.alpha_cum)

    def get_betas(self):
        # linear schedule
        scale = 1000 / self.num_steps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32, device=self.device)
        return betas


    def forward_difussion(self, x, t):
        noise = torch.randn_like(x)
        alpha_cum_t = self.alpha_cum.gather(-1,t).reshape(-1,1,1,1)
        mean = self.sqrt_alpha_cum.gather(-1,t).reshape(-1,1,1,1) * x
        std_dev = self.sqrt_one_minus_alpha_cum.gather(-1,t).reshape(-1,1,1,1)

        sample = mean + std_dev * noise
        return sample, noise

    @torch.no_grad()
    def reverse_diffusion_ddim(self, num_steps, num_images, model, eta = 0.0):
        images = torch.randn((num_images, *self.img_shape), device=self.device)
        model.eval()

        for t in tqdm(reversed(range(1, num_steps))):
            ts = torch.full((num_images,), t, dtype=torch.long, device=self.device)

            alpha_t = self.alpha.gather(0, ts).reshape(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_cum.gather(0, ts).reshape(-1, 1, 1, 1)

            ts_prev = torch.full_like(ts, max(t-1, 0))
            alpha_prev = self.alpha.gather(0, ts_prev).reshape(-1, 1, 1, 1)
            alpha_bar_prev = self.alpha_cum.gather(0, ts_prev).reshape(-1, 1, 1, 1)

            predicted_noise = model(images, ts)

            x0 = (images - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
            x0 = x0.clamp(-1, 1)
            eps = 1e-5
            term = ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_t / alpha_bar_prev))
            term = torch.clamp(term, min=0.0)  # Avoid sqrt of negative
            sigma = eta * torch.sqrt(term + eps)

            noise = torch.randn_like(images) if t > 1 else torch.zeros_like(images)
            images = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev).sqrt() * predicted_noise + sigma * noise

        return images

    @torch.no_grad()
    def reverse_diffusion(self, num_images, model):
        images = torch.randn((num_images, *self.img_shape), device=self.device)
        labels = torch.arange(8, device=self.device)
        model.eval()
        for t in tqdm(reversed(range(1, self.num_steps))):
            ts = torch.ones(num_images, dtype=torch.long, device=self.device) * t
            z = torch.randn_like(images) if t > 1 else torch.zeros_like(images)

            predicted_noise = model(images, ts, labels)
            beta_t = self.beta.gather(-1, ts).reshape(-1, 1, 1, 1)
            one_by_sqrt_alpha = self.one_by_sqrt_alpha.gather(-1, ts).reshape(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.gather(-1, ts).reshape(-1, 1, 1, 1)

            images = (
            one_by_sqrt_alpha
            * (images - (beta_t / sqrt_one_minus_alpha_cum) * predicted_noise)
            + torch.sqrt(beta_t) * z
            )
        
        return images

# diffusion = Diffusion(num_steps=1000, img_shape=(3, 64, 64), device="cuda")

# import torchvision

# dataset_loader = DatasetLoader(dataset_name="Flowers", batch_size=16, img_shape=(3, 64, 64), shuffle=True, device="cuda")
# loader = dataset_loader.get_dataloader()
# plt.figure(figsize=(10, 10))
# for images, labels in loader:
#     images,_ = diffusion.forward_difussion(images.to("cuda"), torch.tensor(100).to("cuda"))
#     images = dataset_loader.inverse_transform(images)
#     grid_img = torchvision.utils.make_grid(images / 255, nrow=4, padding=2)
#     break
# plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
# plt.savefig('ddpm/noisy_grid_img.png')