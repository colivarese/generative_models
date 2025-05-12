import torch

class SimpleDiffusion:
    def __init__(self, num_steps = 1000, img_shape=(3,64,64), device="cpu"):
        self.num_steps = num_steps
        self.img_shape = img_shape
        self.device = device
        self.initialize()

    def initialize(self):
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

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


    def forward_diffusion(self, x0, t):
        eps = torch.randn_like(x0)
        alpha_cum_t = self.alpha_cum.gather(-1,t).reshape(-1,1,1,1)
        mean = self.sqrt_alpha_cum.gather(-1,t).reshape(-1,1,1,1) * x0
        std_dev = self.sqrt_one_minus_alpha_cum.gather(-1,t).reshape(-1,1,1,1)
        sample = mean + std_dev * eps
        return sample, eps

