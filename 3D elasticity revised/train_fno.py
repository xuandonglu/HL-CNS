import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
from plot_tools import compute_strain_tensor
from train_hlcns import FieldDataset, GlobalScaler, relative_l2


def make_grid_3d(batch_size, depth, height, width, device, dtype):
    z = torch.linspace(0.0, 1.0, depth, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1, -1)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=6, modes2=6, modes3=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3, 2)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def compl_mul3d(x, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", x, torch.view_as_complex(weights.contiguous()))

    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            depth,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, depth)
        m2 = min(self.modes2, height)
        m3 = min(self.modes3, width // 2 + 1)
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3])
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3])
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3])
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3])
        return torch.fft.irfftn(out_ft, s=(depth, height, width), dim=(-3, -2, -1), norm="ortho")


class FNOBlock3d(nn.Module):
    def __init__(self, width, modes1, modes2, modes3):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.pointwise = nn.Conv3d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNOElasticity3D(nn.Module):
    """Parameter budget: about 1.0M real parameters with width=12 and 4 FNO blocks."""

    def __init__(self, in_channels=12, out_channels=3, width=12, modes1=6, modes2=6, modes3=6, n_layers=4):
        super().__init__()
        self.lift = nn.Conv3d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[FNOBlock3d(width, modes1, modes2, modes3) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv3d(width, 2 * width, 1), nn.GELU(), nn.Conv3d(2 * width, out_channels, 1))

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1.0 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask, bc_val):
        grid = make_grid_3d(f.size(0), f.size(-3), f.size(-2), f.size(-1), f.device, f.dtype)
        x = torch.cat([f, bc_mask, bc_val, grid], dim=1)
        x = self.lift(x)
        x = self.blocks(x)
        return self._apply_bc(self.project(x), bc_mask, bc_val)


def train_elasticity_fno(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 10,
    lr: float = 1e-3,
    log_csv_path: str = "training_log_fno.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    ood_dataset = FieldDataset(data_dir, prefix="dualforce_")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = FNOElasticity3D().to(device)
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  |  Total: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    loss_fn = nn.MSELoss()
    scaler = GlobalScaler(quantile=0.99)
    start_epoch = 0
    best_val = float("inf")

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        scaler.S = float(ckpt.get("scale_S", 0.0)) or None
        if scaler.S is None:
            scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"Resuming from epoch {start_epoch}; S = {scaler.S:.3e}")
    else:
        print("Starting fresh.")
        scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"[Scaler] S = {scaler.S:.3e} (u,f,bc_val divided by S)")

    ensure_parent_dir(model_path)
    ensure_parent_dir(best_model_path)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)", "OOD(MSE)", "OOD(RelL2)"])

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_mse = 0.0
            train_rel = 0.0
            for f, u, m, bv in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                pred = model(f_s, m, bv_s)
                mse_disp = loss_fn(pred, u_s)
                eps_pred = compute_strain_tensor(pred, 1.0 / (u_s.shape[-1] - 1))
                eps_true = compute_strain_tensor(u_s, 1.0 / (u_s.shape[-1] - 1))
                mse_strain = loss_fn(eps_pred, eps_true)
                loss = 0.2 * mse_disp + 0.8 * mse_strain
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_mse += mse_disp.item()
                train_rel += relative_l2(pred, u_s)

            avg_train = train_mse / max(1, len(train_loader))
            avg_train_rel = train_rel / max(1, len(train_loader))

            def evaluate(loader):
                mse = 0.0
                rel = 0.0
                model.eval()
                with torch.no_grad():
                    for f, u, m, bv in loader:
                        f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                        f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                        pred = model(f_s, m, bv_s)
                        mse += loss_fn(pred, u_s).item()
                        rel += relative_l2(pred, u_s)
                return mse / max(1, len(loader)), rel / max(1, len(loader))

            avg_val, avg_val_rel = evaluate(val_loader)
            avg_ood, avg_ood_rel = evaluate(ood_loader)
            scheduler.step(avg_val)

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scale_S": scaler.S,
            }
            torch.save(state, model_path)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train MSE: {avg_train:.6e} | RelL2: {avg_train_rel:.6e}")
            print(f"  Val   MSE: {avg_val:.6e} | RelL2: {avg_val_rel:.6e}")
            print(f"  OOD   MSE: {avg_ood:.6e} | RelL2: {avg_ood_rel:.6e}")
            writer.writerow([epoch, avg_train, avg_train_rel, avg_val, avg_val_rel, avg_ood, avg_ood_rel])


if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    data_dir = f"./data_{mode}"
    for i in range(1, 6):
        model_dir = f"model_{mode}/fno"
        train_elasticity_fno(
            data_dir=data_dir,
            model_path=f"{model_dir}/checkpoint_fno_{i}.pt",
            best_model_path=f"{model_dir}/best_fno_{i}.pt",
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            log_csv_path=f"{model_dir}/fno_{i}.csv",
        )
