import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
from train_hlcns import PackedTensorDataset3D, create_bc_mask_and_value_like_3d, grad_3d_scalar


def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()


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


class FNOThermal3D(nn.Module):
    """Parameter budget: about 0.72M real parameters with width=10 and 4 FNO blocks."""

    def __init__(self, in_channels=6, out_channels=1, width=10, modes1=6, modes2=6, modes3=6, n_layers=4):
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
        out = self.project(x)
        return self._apply_bc(out, bc_mask, bc_val)


def train_3d_fno(
    train_pt: str,
    val_pt: str,
    ood_pt: str,
    model_ckpt: str = "ckpt_3d_fno.pt",
    best_ckpt: str = "best_3d_fno.pt",
    num_epochs: int = 100,
    batch_size: int = 10,
    lr: float = 1e-3,
    use_grad_loss: bool = True,
    device: str | None = None,
    log_csv_path: str = "training_log_fno.csv",
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = PackedTensorDataset3D(train_pt)
    val_ds = PackedTensorDataset3D(val_pt)
    ood_ds = PackedTensorDataset3D(ood_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    _, _, depth, _, _ = train_ds.f.shape
    model = FNOThermal3D().to(device)
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  |  Total: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    ensure_parent_dir(model_ckpt)
    ensure_parent_dir(best_ckpt)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)", "OOD(MSE)", "OOD(RelL2)"])

        for epoch in range(num_epochs):
            model.train()
            total_mse = 0.0
            total_rel = 0.0
            for f, u in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                f = f.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)
                bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                pred = model(f, bc_mask, bc_val)
                interior = bc_mask < 0.5
                mse_field = loss_fn(pred[interior], u[interior])
                if use_grad_loss:
                    dx = 1.0 / (depth - 1)
                    txp, typ, tzp = grad_3d_scalar(pred, dx)
                    txt, tyt, tzt = grad_3d_scalar(u, dx)
                    mse_grad = (loss_fn(txp[interior], txt[interior]) + loss_fn(typ[interior], tyt[interior]) + loss_fn(tzp[interior], tzt[interior])) / 3.0
                    loss = 0.5 * mse_field + 0.5 * mse_grad
                else:
                    loss = mse_field
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_mse += mse_field.item()
                total_rel += relative_l2(pred[interior], u[interior])

            avg_train = total_mse / max(1, len(train_loader))
            avg_train_rel = total_rel / max(1, len(train_loader))

            def evaluate(loader):
                mse = 0.0
                rel = 0.0
                model.eval()
                with torch.no_grad():
                    for f, u in loader:
                        f = f.to(device, non_blocking=True)
                        u = u.to(device, non_blocking=True)
                        bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                        pred = model(f, bc_mask, bc_val)
                        interior = bc_mask < 0.5
                        mse += loss_fn(pred[interior], u[interior]).item()
                        rel += relative_l2(pred[interior], u[interior])
                return mse / max(1, len(loader)), rel / max(1, len(loader))

            avg_val, avg_val_rel = evaluate(val_loader)
            avg_ood, avg_ood_rel = evaluate(ood_loader)
            scheduler.step(avg_val)

            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, model_ckpt)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), best_ckpt)

            print(f"Epoch {epoch + 1}/{num_epochs} | Train MSE: {avg_train:.6e} | Train Rel: {avg_train_rel:.6e}")
            print(f"Val MSE:{avg_val:.6e} | Val Rel:{avg_val_rel:.6e}")
            print(f"OOD MSE:{avg_ood:.6e} | OOD Rel:{avg_ood_rel:.6e}")
            writer.writerow([epoch, avg_train, avg_train_rel, avg_val, avg_val_rel, avg_ood, avg_ood_rel])


if __name__ == "__main__":
    mode = "linear"
    for i in range(1, 6):
        model_dir = f"model_{mode}/fno"
        train_3d_fno(
            train_pt=f"data/training_data_N32_{mode}.pt",
            val_pt=f"data/validation_data_N32_{mode}.pt",
            ood_pt=f"data/dualforce_data_N32_{mode}.pt",
            model_ckpt=f"{model_dir}/ckpt_fno_{mode}_{i}.pt",
            best_ckpt=f"{model_dir}/best_fno_{mode}_{i}.pt",
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            use_grad_loss=True,
            log_csv_path=f"{model_dir}/fno_{mode}_{i}.csv",
        )
