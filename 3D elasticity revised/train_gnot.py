import csv
import os

import torch
import torch.nn as nn
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


class BranchEncoder3D(nn.Module):
    def __init__(self, in_channels=12, hidden=96, token_shape=(2, 4, 4)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(token_shape)

    def forward(self, x):
        x = self.pool(self.net(x))
        return x.flatten(2).transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden=96, heads=4, dropout=0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden)
        self.kv_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, query, branch_tokens):
        kv = self.kv_norm(branch_tokens)
        attn_out, _ = self.attn(self.q_norm(query), kv, kv, need_weights=False)
        query = query + attn_out
        return query + self.ffn(self.ffn_norm(query))


class GNOTElasticity3D(nn.Module):
    """Grid-adapted GNOT baseline. Parameter budget is about 0.95M with hidden=96."""

    def __init__(self, in_channels=12, out_channels=3, hidden=96, layers=4, heads=4):
        super().__init__()
        self.branch = BranchEncoder3D(in_channels=in_channels, hidden=hidden, token_shape=(2, 4, 4))
        self.trunk = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden=hidden, heads=heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_channels))

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1.0 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask, bc_val):
        batch_size, _, depth, height, width = f.shape
        grid = make_grid_3d(batch_size, depth, height, width, f.device, f.dtype)
        branch_in = torch.cat([f, bc_mask, bc_val, grid], dim=1)
        branch_tokens = self.branch(branch_in)
        coords = grid.permute(0, 2, 3, 4, 1).reshape(batch_size, depth * height * width, 3)
        query = self.trunk(coords)
        for block in self.blocks:
            query = block(query, branch_tokens)
        out = self.out(query).transpose(1, 2).reshape(batch_size, 3, depth, height, width)
        return self._apply_bc(out, bc_mask, bc_val)


def train_elasticity_gnot(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 10,
    lr: float = 1e-3,
    log_csv_path: str = "training_log_gnot.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    ood_dataset = FieldDataset(data_dir, prefix="dualforce_")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = GNOTElasticity3D().to(device)
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  |  Total: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        model_dir = f"model_{mode}/gnot"
        train_elasticity_gnot(
            data_dir=data_dir,
            model_path=f"{model_dir}/checkpoint_gnot_{i}.pt",
            best_model_path=f"{model_dir}/best_gnot_{i}.pt",
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            log_csv_path=f"{model_dir}/gnot_{i}.csv",
        )
