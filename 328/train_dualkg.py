import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models.dual_kg_diffusion import DualKGDiffusionModel


class DummyDeblurDataset(Dataset):
    def __init__(self, length: int = 64, image_size: int = 256) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        blur = torch.rand(3, self.image_size, self.image_size)
        sharp = torch.rand(3, self.image_size, self.image_size)
        return {"blur": blur, "sharp": sharp}


def build_linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alpha_bars


def q_sample(clean_img: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    sqrt_ab = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars[t]).view(-1, 1, 1, 1)
    return sqrt_ab * clean_img + sqrt_one_minus_ab * noise


def train_one_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    alpha_bars: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
) -> Dict[str, float]:
    blur = batch["blur"].to(device)
    sharp = batch["sharp"].to(device)

    sampled_timesteps = torch.randint(0, num_timesteps, (blur.size(0),), device=device)
    noise = torch.randn_like(sharp)
    noisy_latent = q_sample(sharp, sampled_timesteps, alpha_bars, noise)

    out = model(blur_img=blur, noisy_latent=noisy_latent, timestep=sampled_timesteps)
    pred_noise = out["pred_noise"]
    loss_noise = F.mse_loss(pred_noise, noise)
    loss_base = F.l1_loss(out["base_img"], sharp)
    loss = loss_noise + 0.1 * loss_base

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_noise": loss_noise.item(),
        "loss_base": loss_base.item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-Expert Guided Diffusion training scaffold")
    parser.add_argument("--general_weights", type=str, required=True, help="Path to GoPro-trained expert weights")
    parser.add_argument("--kinematic_weights", type=str, required=True, help="Path to kinematic expert weights")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualKGDiffusionModel(
        general_weights_path=args.general_weights,
        kinematic_weights_path=args.kinematic_weights,
        use_deform_in_feat=True,
        use_deform_in_encoder=True,
        unet_base_channels=64,
        device=device,
    ).to(device)

    dataset = DummyDeblurDataset(length=64, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    optimizer = optim.AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    _, alpha_bars = build_linear_beta_schedule(args.timesteps, device=device)

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, 1):
            metrics = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                alpha_bars=alpha_bars,
                num_timesteps=args.timesteps,
                device=device,
            )
            print(
                f"[epoch {epoch + 1} step {step}] "
                f"loss={metrics['loss']:.6f} "
                f"noise={metrics['loss_noise']:.6f} "
                f"base={metrics['loss_base']:.6f}"
            )


if __name__ == "__main__":
    main()
