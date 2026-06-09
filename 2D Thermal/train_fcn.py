# train_plot_fcn_2d_thermal.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

# ===================== FCN (2D Thermal) =====================
class FCNThermal2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 72, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(72, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 72, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(72, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, out_channels, kernel_size=1)
        )
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[FCNThermal2D] Total parameters: {total_params:,}")

    def forward(self, f, bc_mask, bc_val):
        out = self.net(f)
        return out * (1 - bc_mask) + bc_val * bc_mask

class PackedTensorDataset2D(Dataset):
    def __init__(self, path: str):
        super().__init__()
        d = torch.load(path, map_location='cpu')
        self.f = d['f'].float()  # [N,1,H,W]
        self.u = d['u'].float()
        assert self.f.shape == self.u.shape and self.f.ndim == 4

    def __len__(self): return self.f.shape[0]
    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]

def _build_deriv_kernels_2d(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1,1,1,3)
    Ky = k.view(1,1,3,1)
    return Kx, Ky

def grad_2d_scalar(T: torch.Tensor, dx: float):
    Kx, Ky = _build_deriv_kernels_2d(T.dtype, T.device)
    Tx = F.conv2d(F.pad(T, (1,1,0,0), mode='replicate'), Kx) / dx
    Ty = F.conv2d(F.pad(T, (0,0,1,1), mode='replicate'), Ky) / dx
    return Tx, Ty

@torch.no_grad()
def create_bc_mask_and_value_like_2d(T: torch.Tensor):
    B, C, H, W = T.shape
    assert C == 1
    bc_mask = torch.zeros_like(T)
    bc_val  = torch.zeros_like(T)

    bc_mask[:, :, 0,  :] = 1.0
    bc_val[:,  :, 0,  :] = 1.0
    bc_mask[:, :, -1, :] = 1.0
    bc_val[:,  :, -1, :] = 0.0

    v = torch.linspace(1.0, 0.0, H, device=T.device, dtype=T.dtype).view(1, 1, H)  # [1,1,H]
    vB = v.expand(B, 1, H)  # [B,1,H]

    bc_mask[:, :, :, 0]  = 1.0
    bc_val[:,  :, :, 0]  = vB
    bc_mask[:, :, :, -1] = 1.0
    bc_val[:,  :, :, -1] = vB

    return bc_mask, bc_val


def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

def train_thermal_2d_fcn(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = 'checkpoint_thermal2d_fcn.pt',
    best_ckpt: str  = 'best_thermal2d_fcn.pt',
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    kappa: float = 1.0,
    use_grad_loss: bool = True,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = PackedTensorDataset2D(train_pt)
    val_ds   = PackedTensorDataset2D(val_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    _, _, H, W = train_ds.f.shape
    N  = H
    dx = 1.0 / (N - 1)

    model = FCNThermal2D(in_channels=1, out_channels=1).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    start_epoch = 0

    if os.path.exists(model_ckpt):
        ckpt = torch.load(model_ckpt, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_field = 0.0
        total_train_rel   = 0.0
        total_train_loss  = 0.0

        for f, u in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            f = f.to(device, non_blocking=True)  # [B,1,H,W]
            u = u.to(device, non_blocking=True)  # [B,1,H,W]
            bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)

            f_s, u_s, bv_s = f, u, bc_val
            pred = model(f_s, bc_mask, bv_s)
            interior = (bc_mask < 0.5)
            field_mse = loss_fn(pred[interior], u_s[interior])

            if use_grad_loss:
                Txp, Typ = grad_2d_scalar(pred, dx)
                Txt, Tyt = grad_2d_scalar(u_s, dx)
                grad_mse = 0.5 * (loss_fn(Txp[interior], Txt[interior]) +
                                  loss_fn(Typ[interior], Tyt[interior]))
                loss = 0.5 * field_mse + 0.5 * grad_mse
            else:
                loss = field_mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_field += float(field_mse.item())
            total_train_loss  += float(loss.item())
            total_train_rel   += relative_l2(pred[interior], u_s[interior])

        avg_train_field = total_train_field / max(1, len(train_loader))
        avg_train_loss  = total_train_loss  / max(1, len(train_loader))
        avg_train_rel   = total_train_rel   / max(1, len(train_loader))

        model.eval()
        val_field = 0.0
        val_rel   = 0.0
        with torch.no_grad():
            for f, u in val_loader:
                f = f.to(device); u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                f_s, u_s, bv_s = f, u, bc_val
                pred = model(f_s, bc_mask, bv_s)
                interior = (bc_mask < 0.5)
                val_field += loss_fn(pred[interior], u_s[interior]).item()
                val_rel   += relative_l2(pred[interior], u_s[interior])

        avg_val_field = val_field / max(1, len(val_loader))
        avg_val_rel   = val_rel   / max(1, len(val_loader))

        if scheduler: scheduler.step(avg_val_field)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train FieldMSE: {avg_train_field:.6e} | Train Loss: {avg_train_loss:.6e} | Train Rel: {avg_train_rel:.6e}")
        print(f"  Val   FieldMSE: {avg_val_field:.6e} | Val   Rel : {avg_val_rel:.6e}")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, model_ckpt)
        if avg_val_field < best_val:
            best_val = avg_val_field
            torch.save(model.state_dict(), best_ckpt)
            print("  ✓ New best model saved")

        # log
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_field, avg_val_field])

if __name__ == "__main__":
    mode = "linear"
    train_pt = f"data/training_data_N32_{mode}.pt"
    val_pt = f"data/validation_data_N32_{mode}.pt"

    for i in range(1, 6):
        model_path = f"model_{mode}/fcn/ckpt_2d_fcn_{mode}_{i}.pt"
        best_model_path = f"model_{mode}/fcn/best_2d_fcn_{mode}_{i}.pt"
        log_path = f"model_{mode}/fcn/fcn_{mode}_{i}.csv"

        train_thermal_2d_fcn(
            train_pt=train_pt,
            val_pt=val_pt,
            model_ckpt=model_path,
            best_ckpt=best_model_path,
            num_epochs=50,
            batch_size=10,
            lr=1e-3,
            kappa=1.0,
            use_grad_loss=True,
            log_csv_path=log_path
        )
