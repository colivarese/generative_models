import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, num_steps, img_shape, device):
        self.num_steps = num_steps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        self.beta = self.get_betas()
        self.beta = self.beta.to(self.device)
        self.alpha = 1. - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cum = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1 - self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.clamp(self.sqrt_one_minus_alpha_cum, min=1e-5, max=0.9999)


    def get_betas(self):
        scale = 1000 / self.num_steps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32, device=self.device)
        return betas
    
    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        mean = self.sqrt_alpha_cum.gather(-1, t).reshape(-1, 1, 1, 1) * x
        std_dev = self.sqrt_one_minus_alpha_cum.gather(-1, t).reshape(-1, 1, 1, 1)
        sample = mean + std_dev * noise
        return sample, noise


    @torch.no_grad()
    def reverse_diffusion(self, context, model):
        batch_size = context.size(0)
        images = torch.randn((batch_size, 4, 64, 64), device=self.device)
        model.eval()

        for t in tqdm(reversed(range(1, self.num_steps)), desc="Sampling..."):
            ts = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            z = torch.randn_like(images) if t > 1 else torch.zeros_like(images)
            predicted_noise = model(images, ts, context)

            beta_t = self.beta.gather(-1, ts).reshape(-1, 1, 1, 1)
            beta_t = torch.clamp(beta_t, min=1e-5, max=0.999)  # Clamp for sqrt stability

            one_by_sqrt_alpha = self.one_by_sqrt_alpha.gather(-1, ts).reshape(-1, 1, 1, 1)

            sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.gather(-1, ts).reshape(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cum = torch.clamp(sqrt_one_minus_alpha_cum, min=1e-5, max=0.9999)  # Already clamped, but good for runtime safety

            update = (beta_t / sqrt_one_minus_alpha_cum) * predicted_noise
            images = one_by_sqrt_alpha * (images - update) + torch.sqrt(beta_t) * z

        return images
    
