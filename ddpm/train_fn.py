from torchmetrics import MeanMetric
from tqdm import tqdm
import torch
from torch.cuda import amp
import torchvision.transforms as TF
from dataset import inverse_transform
from torchvision.utils import make_grid


def train_one_epoch(model, loader, diffusion, optimizer, scaler, loss_fn, epoch,
                    base_config, training_config):

    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Epoch {epoch+1}/{training_config.NUM_EPOCHS}")

        for x0, _ in loader:
            tq.update(1)
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = diffusion.forward_diffusion(x0.to(base_config.DEVICE), ts)

            with torch.amp.autocast('cuda'):
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()
        tq.set_postfix_str(s=f"Epoch loss: {mean_loss:.4f}")
    return mean_loss


@torch.no_grad()
def reverse_diffusion(model, diffusion, epoch, timesteps=1000, img_shape=(3, 64, 64), num_images=5,
                     device="cuda", nrow=8, **kwargs):
    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    for t in tqdm(iterable=reversed(range(1, timesteps)),
                total=timesteps-1, dynamic_ncols=False, desc="Sampling :: ", position=0):
        ts = torch.ones(num_images, dtype=torch.long, device=device) * t
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t = diffusion.beta.gather(-1,ts).reshape(-1, 1, 1, 1) #diffusion.beta[ts]
        one_by_sqrt_alpha = diffusion.one_by_sqrt_alpha.gather(-1,ts).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum = diffusion.sqrt_one_minus_alpha_cum.gather(-1,ts).reshape(-1, 1, 1, 1)


        x = (
            one_by_sqrt_alpha
            * (x - (beta_t / sqrt_one_minus_alpha_cum) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        # alpha_t = diffusion.alpha.gather(-1, ts).reshape(-1, 1, 1, 1)
        # alpha_t_prev = diffusion.alpha.gather(-1, ts-1).reshape(-1, 1, 1, 1)
        # # beta_t = diffusion.beta.gather(-1, ts).reshape(-1, 1, 1, 1)
        # sqrt_one_minus_alpha_cum = diffusion.sqrt_one_minus_alpha_cum.gather(-1, ts).reshape(-1, 1, 1, 1)

        # x_0 = ( x - sqrt_one_minus_alpha_cum * predicted_noise ) / alpha_t.sqrt()
        # dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise

        # eta = 0

        # noise = eta * torch.sqrt((1.0 - alpha_t_prev) / (1.0 - alpha_t)) * z
        # x = torch.sqrt(alpha_t_prev) * x_0 + dir_xt + noise
        # x = (
        #     1 / alpha_t.sqrt() * (x - (1 - alpha_t).sqrt() * predicted_noise)
        #     + beta_t.sqrt() * z
        # )

    x = inverse_transform(x).type(torch.uint8)
    grid = make_grid(x, nrow=nrow, pad_value= 255.0).to("cpu")
    pil_image = TF.functional.to_pil_image(grid)
    pil_image.save(f"ddpm/out_{epoch}.png")
    return None
