from diffusion import SimpleDiffusion
from model import Unet
from train_fn import train_one_epoch, reverse_diffusion

from dataset import BaseConfig, TrainingConfig, get_dataloader, inverse_transform
from torchvision.utils import make_grid
import gc
import torch
import os
import torch.nn as nn
from torch.cuda import amp
import matplotlib.pyplot as plt
device="cuda"

class ModelConfig:
    BASE_CH = 64
    BASE_CH_MULT = (1, 2, 4, 4)
    APPLY_ATTENTION = (False, True, True, False)
    TIME_EMB_MULT = 4
    DROPOUT_RATE = 0.1

model = Unet(
    input_channels          = TrainingConfig.IMG_SHAPE[0],
    output_channels         = TrainingConfig.IMG_SHAPE[0],
    base_channels           = ModelConfig.BASE_CH,
    base_channels_mult = ModelConfig.BASE_CH_MULT,
    apply_attention         = ModelConfig.APPLY_ATTENTION,
    dropout_rate            = ModelConfig.DROPOUT_RATE,
    time_multiple           = ModelConfig.TIME_EMB_MULT,
)

print(model)

model.to(BaseConfig.DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR) # Original → Adam

dataloader = get_dataloader(
    dataset_name  = BaseConfig.DATASET,
    batch_size    = TrainingConfig.BATCH_SIZE,
    device        = BaseConfig.DEVICE,
    pin_memory    = True,
    num_workers   = TrainingConfig.NUM_WORKERS,
)

loss_fn = nn.MSELoss()
 
sd = SimpleDiffusion(
    num_steps = TrainingConfig.TIMESTEPS,
    img_shape               = TrainingConfig.IMG_SHAPE,
    device                  = BaseConfig.DEVICE,
)


scaler = amp.GradScaler()
#sd = SimpleDiffusion(num_steps = TrainingConfig.TIMESTEPS, img_shape=TrainingConfig.IMG_SHAPE, device="cuda")

total_epochs = TrainingConfig.NUM_EPOCHS + 1

for epoch in range(1, total_epochs):
    torch.cuda.empty_cache()
    gc.collect()
     
    # Algorithm 1: Training
    train_one_epoch(model, dataloader, sd, optimizer, scaler, loss_fn, epoch=epoch,
                    base_config=BaseConfig, training_config=TrainingConfig)
 
    if epoch % 20 == 0:
        #save_path = os.path.join(log_dir, f"{epoch}{ext}")
         
        # Algorithm 2: Sampling
        reverse_diffusion(model, sd, epoch=epoch, timesteps=TrainingConfig.TIMESTEPS, 
                          num_images=32, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE, nrow=4,
        )
 
        # clear_output()
        checkpoint_dict = {
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint_dict, "ckpt.pt")
        del checkpoint_dict
#log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

# loader = iter(get_dataloader(
#     dataset_name = BaseConfig.DATASET,
#     batch_size = 6, 
#     device=device
# ))

# x0s, eps = next(loader)

# noisy_images = []
# ts = [0, 10, 50, 100, 150, 200, 300, 400, 600, 800, 999]

# for t in ts:
#     t = torch.as_tensor(t, dtype=torch.long, device=device)
#     xts, epst = sd.forward_diffusion(x0s.to(device), t)
#     xts = inverse_transform(xts)
#     xts = make_grid(xts / 255, nrow=1, padding=1)

#     noisy_images.append(xts)

# _, ax = plt.subplots(1, len(noisy_images), figsize=(20, 10))

# for i, (t, sample) in enumerate(zip(ts, noisy_images)):
#     ax[i].imshow(sample.permute(1, 2, 0).cpu().numpy())
#     ax[i].set_title(f"t = {t}")
#     ax[i].axis("off")

# plt.savefig('ddpm/plot.png')