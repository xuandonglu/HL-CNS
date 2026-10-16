import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
from train_hlcns import PackedTensorDataset2D, create_bc_mask_and_value_like_2d, grad_2d_scalar


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()


def make_grid_2d(batch_size, height, width, device, dtype):
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1)


def interior_mask_like(u: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(u, dtype=torch.bool)
    mask[..., 0, :] = False
    mask[..., -1, :] = False
    mask[..., :, 0] = False
    mask[..., :, -1] = False
    return mask


def compute_stats(dataset: PackedTensorDataset2D) -> dict[str, torch.Tensor]:
    interior = interior_mask_like(dataset.u)
    stats = {
        "f_mean": dataset.f.mean().view(1, 1, 1, 1),
        "f_std": dataset.f.std().clamp_min(1e-6).view(1, 1, 1, 1),
        "u_mean": dataset.u[interior].mean().view(1, 1, 1, 1),
        "u_std": dataset.u[interior].std().clamp_min(1e-6).view(1, 1, 1, 1),
    }
    return stats


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(weights.contiguous()))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNOStableThermal2D(nn.Module):
    """Normalized, padded FNO baseline. Parameter budget is about 0.75M."""

    def __init__(self, in_channels=5, out_channels=1, width=18, modes1=12, modes2=12, n_layers=4, padding=6):
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1),
            nn.GELU(),
            nn.Conv2d(2 * width, out_channels, 1),
        )
        self.register_buffer("f_mean", torch.zeros(1, 1, 1, 1))
        self.register_buffer("f_std", torch.ones(1, 1, 1, 1))
        self.register_buffer("u_mean", torch.zeros(1, 1, 1, 1))
        self.register_buffer("u_std", torch.ones(1, 1, 1, 1))

    def set_stats(self, stats: dict[str, torch.Tensor]) -> None:
        for name, value in stats.items():
            getattr(self, name).copy_(value.to(device=getattr(self, name).device, dtype=getattr(self, name).dtype))

    def normalize_f(self, f):
        return (f - self.f_mean) / self.f_std

    def normalize_u(self, u):
        return (u - self.u_mean) / self.u_std

    def denormalize_u(self, u):
        return u * self.u_std + self.u_mean

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1.0 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask, bc_val):
        grid = make_grid_2d(f.size(0), f.size(-2), f.size(-1), f.device, f.dtype)
        x = torch.cat([self.normalize_f(f), bc_mask, self.normalize_u(bc_val), grid], dim=1)
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        x = self.blocks(x)
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]
        out = self.denormalize_u(self.project(x))
        return self._apply_bc(out, bc_mask, bc_val)


def train_2d_fno_stable(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = "ckpt_2d_fno_stable.pt",
    best_ckpt: str = "best_2d_fno_stable.pt",
    num_epochs: int = 100,
    batch_size: int = 10,
    lr: float = 5e-4,
    use_grad_loss: bool = True,
    grad_weight: float = 0.02,
    device: str | None = None,
    log_csv_path: str = "training_log_fno_stable.csv",
    seed: int | None = None,
):
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = PackedTensorDataset2D(train_pt)
    val_ds = PackedTensorDataset2D(val_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    _, _, height, _ = train_ds.f.shape
    model = FNOStableThermal2D().to(device)
    model.set_stats(compute_stats(train_ds))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")
    print(
        "Stats -- "
        f"f_mean:{model.f_mean.item():.6e} f_std:{model.f_std.item():.6e} "
        f"u_mean:{model.u_mean.item():.6e} u_std:{model.u_std.item():.6e}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.05)
    loss_fn = nn.MSELoss()
    best_val_rel = float("inf")

    ensure_parent_dir(model_ckpt)
    ensure_parent_dir(best_ckpt)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)", "LR"])

        for epoch in range(num_epochs):
            model.train()
            total_mse = 0.0
            total_rel = 0.0
            for f, u in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                f = f.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)

                pred = model(f, bc_mask, bc_val)
                interior = bc_mask < 0.5
                mse_field = loss_fn(pred[interior], u[interior])
                pred_n = model.normalize_u(pred)
                u_n = model.normalize_u(u)
                loss = loss_fn(pred_n[interior], u_n[interior])
                if use_grad_loss:
                    dx = 1.0 / (height - 1)
                    txp, typ = grad_2d_scalar(pred_n, dx)
                    txt, tyt = grad_2d_scalar(u_n, dx)
                    mse_grad = 0.5 * (loss_fn(txp[interior], txt[interior]) + loss_fn(typ[interior], tyt[interior]))
                    loss = loss + grad_weight * mse_grad

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_mse += mse_field.item()
                total_rel += relative_l2(pred[interior], u[interior])

            avg_train = total_mse / max(1, len(train_loader))
            avg_rel = total_rel / max(1, len(train_loader))

            model.eval()
            val_mse = 0.0
            val_rel = 0.0
            with torch.no_grad():
                for f, u in val_loader:
                    f = f.to(device, non_blocking=True)
                    u = u.to(device, non_blocking=True)
                    bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                    pred = model(f, bc_mask, bc_val)
                    interior = bc_mask < 0.5
                    val_mse += loss_fn(pred[interior], u[interior]).item()
                    val_rel += relative_l2(pred[interior], u[interior])
            avg_val = val_mse / max(1, len(val_loader))
            avg_val_rel = val_rel / max(1, len(val_loader))
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
            torch.save(state, model_ckpt)
            if avg_val_rel < best_val_rel:
                best_val_rel = avg_val_rel
                torch.save(model.state_dict(), best_ckpt)

            print(f"Epoch {epoch + 1}/{num_epochs} | Train MSE: {avg_train:.6e} | Train Rel: {avg_rel:.6e}")
            print(f"Val MSE:{avg_val:.6e} | Val Rel:{avg_val_rel:.6e} | LR:{current_lr:.6e}")
            writer.writerow([epoch, avg_train, avg_rel, avg_val, avg_val_rel, current_lr])


if __name__ == "__main__":
    mode = "linear"
    for i in range(1, 6):
        model_dir = f"model_{mode}/fno_stable"
        train_2d_fno_stable(
            train_pt=f"data/training_data_N32_{mode}.pt",
            val_pt=f"data/validation_data_N32_{mode}.pt",
            model_ckpt=f"{model_dir}/ckpt_2d_fno_stable_{mode}_{i}.pt",
            best_ckpt=f"{model_dir}/best_2d_fno_stable_{mode}_{i}.pt",
            num_epochs=100,
            batch_size=10,
            lr=5e-4,
            use_grad_loss=True,
            grad_weight=0.02,
            seed=2000 + i,
            log_csv_path=f"{model_dir}/fno_stable_{mode}_{i}.csv",
        )
