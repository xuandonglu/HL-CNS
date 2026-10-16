import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
from train_hlcns import PackedTensorDataset2D, create_bc_mask_and_value_like_2d, grad_2d_scalar


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


class BranchEncoder2D(nn.Module):
    def __init__(self, in_channels=5, hidden=96, token_grid=8):
        super().__init__()
        self.token_grid = token_grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

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
        attn_out, _ = self.attn(self.q_norm(query), self.kv_norm(branch_tokens), self.kv_norm(branch_tokens), need_weights=False)
        query = query + attn_out
        return query + self.ffn(self.ffn_norm(query))


class GNOTThermal2D(nn.Module):
    """Grid-adapted GNOT baseline. Parameter budget is about 0.55M with hidden=96."""

    def __init__(self, in_channels=5, out_channels=1, hidden=96, layers=4, heads=4):
        super().__init__()
        self.branch = BranchEncoder2D(in_channels=in_channels, hidden=hidden, token_grid=8)
        self.trunk = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden=hidden, heads=heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_channels))

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1.0 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask, bc_val):
        batch_size, _, height, width = f.shape
        grid = make_grid_2d(batch_size, height, width, f.device, f.dtype)
        branch_in = torch.cat([f, bc_mask, bc_val, grid], dim=1)
        branch_tokens = self.branch(branch_in)
        coords = grid.permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
        query = self.trunk(coords)
        for block in self.blocks:
            query = block(query, branch_tokens)
        out = self.out(query).transpose(1, 2).reshape(batch_size, 1, height, width)
        return self._apply_bc(out, bc_mask, bc_val)


def train_2d_gnot(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = "ckpt_2d_gnot.pt",
    best_ckpt: str = "best_2d_gnot.pt",
    num_epochs: int = 100,
    batch_size: int = 10,
    lr: float = 1e-3,
    use_grad_loss: bool = True,
    device: str | None = None,
    log_csv_path: str = "training_log_gnot.csv",
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = PackedTensorDataset2D(train_pt)
    val_ds = PackedTensorDataset2D(val_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    _, _, height, _ = train_ds.f.shape

    model = GNOTThermal2D().to(device)
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  |  Total: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    ensure_parent_dir(model_ckpt)
    ensure_parent_dir(best_ckpt)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)"])

        for epoch in range(num_epochs):
            model.train()
            train_mse = 0.0
            train_rel = 0.0
            for f, u in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                f = f.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                pred = model(f, bc_mask, bc_val)
                interior = bc_mask < 0.5
                mse_field = loss_fn(pred[interior], u[interior])
                if use_grad_loss:
                    dx = 1.0 / (height - 1)
                    txp, typ = grad_2d_scalar(pred, dx)
                    txt, tyt = grad_2d_scalar(u, dx)
                    mse_grad = 0.5 * (loss_fn(txp[interior], txt[interior]) + loss_fn(typ[interior], tyt[interior]))
                    loss = 0.5 * mse_field + 0.5 * mse_grad
                else:
                    loss = mse_field
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_mse += mse_field.item()
                train_rel += relative_l2(pred[interior], u[interior])

            avg_train = train_mse / max(1, len(train_loader))
            avg_train_rel = train_rel / max(1, len(train_loader))
            val_mse = 0.0
            val_rel = 0.0
            model.eval()
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
            scheduler.step(avg_val)

            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, model_ckpt)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), best_ckpt)
            print(f"Epoch {epoch + 1}/{num_epochs} | Train MSE: {avg_train:.6e} | Train Rel: {avg_train_rel:.6e}")
            print(f"Val MSE:{avg_val:.6e} | Val Rel:{avg_val_rel:.6e}")
            writer.writerow([epoch, avg_train, avg_train_rel, avg_val, avg_val_rel])


if __name__ == "__main__":
    mode = "linear"
    for i in range(1, 6):
        model_dir = f"model_{mode}/gnot"
        train_2d_gnot(
            train_pt=f"data/training_data_N32_{mode}.pt",
            val_pt=f"data/validation_data_N32_{mode}.pt",
            model_ckpt=f"{model_dir}/ckpt_2d_gnot_{mode}_{i}.pt",
            best_ckpt=f"{model_dir}/best_2d_gnot_{mode}_{i}.pt",
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            use_grad_loss=True,
            log_csv_path=f"{model_dir}/gnot_{mode}_{i}.csv",
        )
