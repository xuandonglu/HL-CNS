
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import csv

# ===================== 3D utilities =====================

def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

def _build_deriv_kernels_3d(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 1, 3)  # d/dx
    Ky = k.view(1, 1, 1, 3, 1)  # d/dy
    Kz = k.view(1, 1, 3, 1, 1)  # d/dz
    return Kx, Ky, Kz

def _build_lap_kernels_3d(dtype, device):
    k2 = torch.tensor([1.0, -2.0, 1.0], dtype=dtype, device=device)
    Kxx = k2.view(1,1,1,1,3)
    Kyy = k2.view(1,1,1,3,1)
    Kzz = k2.view(1,1,3,1,1)
    return Kxx, Kyy, Kzz

@torch.no_grad()
def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))

@torch.no_grad()
def feature_gradmag_3d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """|grad T| for 3D: returns [B,1,D,H,W]."""
    assert u.size(1) == 1, "Thermal expects 1 channel."
    Kx, Ky, Kz = _build_deriv_kernels_3d(u.dtype, u.device)
    u_pad_x = F.pad(u, (1, 1, 0, 0, 0, 0), mode='replicate')   # pad x
    u_pad_y = F.pad(u, (0, 0, 1, 1, 0, 0), mode='replicate')   # pad y
    u_pad_z = F.pad(u, (0, 0, 0, 0, 1, 1), mode='replicate')   # pad z
    Tx = F.conv3d(u_pad_x, Kx) / dx
    Ty = F.conv3d(u_pad_y, Ky) / dx
    Tz = F.conv3d(u_pad_z, Kz) / dx
    gmag = _safe_sqrt(Tx.square() + Ty.square() + Tz.square())
    return gmag

@torch.no_grad()
def create_bc_mask_and_value_like_3d(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3D Dirichlet BCs generalized from 2D:
     - z=0 (top) = 1.0, z=-1 (bottom) = 0.0
     - x=0/x=-1, y=0/y=-1 sides use vertical (z) linear gradient 1->0
    Args:
        T: [B,1,D,H,W]
    Returns:
        bc_mask, bc_val: both [B,1,D,H,W]
    """
    B, C, D, H, W = T.shape
    assert C == 1, f"Expected channel=1 for thermal, got {C}"

    bc_mask = torch.zeros_like(T)
    bc_val  = torch.zeros_like(T)

    # vertical gradient along z: 1 -> 0, shape [1,1,D,1,1]
    vgrad = torch.linspace(1.0, 0.0, D, device=T.device, dtype=T.dtype).view(1, 1, D, 1, 1)

    # Top/Bottom (z faces)
    bc_mask[:, :, 0, :, :]  = 1.0
    bc_val[:,  :, 0, :, :]  = 1.0
    bc_mask[:, :, -1, :, :] = 1.0
    bc_val[:,  :, -1, :, :] = 0.0

    # Left/Right (x faces: w-index)
    bc_mask[:, :, :, :, 0]  = 1.0
    bc_val[:,  :, :, :, 0]  = vgrad.expand(B, 1, D, H, 1).squeeze(-1)
    bc_mask[:, :, :, :, -1] = 1.0
    bc_val[:,  :, :, :, -1] = vgrad.expand(B, 1, D, H, 1).squeeze(-1)

    # Front/Back (y faces: h-index)
    bc_mask[:, :, :, 0, :]  = 1.0
    bc_val[:,  :, :, 0, :]  = vgrad.expand(B, 1, D, 1, W).squeeze(-2)
    bc_mask[:, :, :, -1, :] = 1.0
    bc_val[:,  :, :, -1, :] = vgrad.expand(B, 1, D, 1, W).squeeze(-2)

    return bc_mask, bc_val

def interior_mask_from_bc_3d(bc_mask):
    # bc_mask: [B,1,D,H,W] -> True for free nodes
    return ~(bc_mask.bool())

# ================ Packed dataset: training/validation .pt =================

class PackedTensorDataset3D(Dataset):
    """Loads a packed .pt with keys {'f': [N,1,D,H,W], 'u': [N,1,D,H,W]}"""
    def __init__(self, path: str):
        super().__init__()
        data = torch.load(path)
        self.f = data['f'].float()
        self.u = data['u'].float()
        assert self.f.ndim == 5 and self.u.ndim == 5, "Expect shapes [N,1,D,H,W]"
        assert self.f.shape == self.u.shape, "f and u must match shape"
    def __len__(self):
        return self.f.shape[0]
    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]

# ===================== Sampling / gather helpers (3D) =====================

def random_indices_3d(B, M, N, device, pad=0):
    lo, hi = pad, N - pad
    iz = torch.randint(lo, hi, (B, M), device=device)
    iy = torch.randint(lo, hi, (B, M), device=device)
    ix = torch.randint(lo, hi, (B, M), device=device)
    return iz, iy, ix

def make_coords_from_indices_3d(ix, iy, iz, N, space="[0,1]"):
    if space == "[-1,1]":
        s = 2.0 / (N - 1)
        x = -1.0 + ix.float() * s
        y = -1.0 + iy.float() * s
        z = -1.0 + iz.float() * s
    else:
        s = 1.0 / (N - 1)
        x = ix.float() * s
        y = iy.float() * s
        z = iz.float() * s
    return torch.stack([x, y, z], dim=-1)  # [B,M,3]

def gather_points_3d(u, iz, iy, ix):
    # u: [B,1,D,H,W] -> [B,M,1]
    B, C, D, H, W = u.shape
    b = torch.arange(B, device=u.device).view(B, 1).expand_as(ix)
    vals = u[b, 0, iz, iy, ix]  # [B,M]
    return vals.unsqueeze(-1)

def gather_mask_3d(mask3d, iz, iy, ix):
    # mask3d: [B,1,D,H,W] -> [B,M]
    B, _, D, H, W = mask3d.shape
    b = torch.arange(B, device=mask3d.device).view(B, 1).expand_as(ix)
    m = mask3d[b, 0, iz, iy, ix]
    return m.float()

def sample_boundary_indices_3d(bc_mask, Mb):
    # pick Mb indices where mask==1 for each batch
    B, _, D, H, W = bc_mask.shape
    iz = torch.zeros(B, Mb, dtype=torch.long, device=bc_mask.device)
    iy = torch.zeros_like(iz)
    ix = torch.zeros_like(iz)
    for b in range(B):
        mask = bc_mask[b,0]  # [D,H,W]
        idx = mask.nonzero(as_tuple=False)  # [K,3] (z,y,x)
        if idx.numel() == 0:
            iz[b] = torch.randint(0, D, (Mb,), device=bc_mask.device)
            iy[b] = torch.randint(0, H, (Mb,), device=bc_mask.device)
            ix[b] = torch.randint(0, W, (Mb,), device=bc_mask.device)
        else:
            sel = idx[torch.randint(0, idx.shape[0], (Mb,), device=bc_mask.device)]
            iz[b], iy[b], ix[b] = sel[:, 0], sel[:, 1], sel[:, 2]
    return iz, iy, ix

def masked_point_mse_scalar(pred_pts, true_pts, mask_pts):
    # pred_pts/true_pts: [B,M,1]; mask_pts: [B,M] (1=keep)
    w = (mask_pts > 0.5).float().unsqueeze(-1)  # [B,M,1]
    num = ((pred_pts - true_pts) ** 2 * w).sum()
    den = (w.sum()).clamp_min(1.0)
    return num / den

@torch.no_grad()
def rel_l2_points_scalar(pred_pts, true_pts, mask_pts):
    w = (mask_pts > 0.5).float().unsqueeze(-1)
    num = torch.linalg.vector_norm(((pred_pts - true_pts) * w).reshape(pred_pts.size(0), -1), dim=1).sum()
    den = torch.linalg.vector_norm((true_pts * w).reshape(true_pts.size(0), -1), dim=1).sum() + 1e-12
    return (num / den).item()


# ---------------- ResNet (3D Thermal) ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.short = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        r = self.short(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu(y + r)
        return y

class ResNetThermal3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 12, 3, padding=1, bias=False),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock3D(12, 24, stride=1)
        self.layer2 = ResidualBlock3D(24, 36, stride=2)
        self.layer3 = ResidualBlock3D(36, 48, stride=2)
        self.layer4 = ResidualBlock3D(48, 60, stride=2)

        self.up1 = nn.ConvTranspose3d(60, 36, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(36, 24, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.head = nn.Sequential(
            nn.Conv3d(12, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True),
            nn.Conv3d(12, out_ch, 1)
        )

    def forward(self, net_in, bc_mask, bc_val):
        x = self.stem(net_in)
        x = self.layer1(x)
        x = self.layer2(x)  # /2
        x = self.layer3(x)  # /4
        x = self.layer4(x)  # /8
        x = self.up1(x)     # /4
        x = self.up2(x)     # /2
        x = self.up3(x)     # /1
        u = self.head(x)
        return u * (1 - bc_mask) + bc_val * bc_mask

def grad_3d_scalar(T: torch.Tensor, dx: float):
    Kx, Ky, Kz = _build_deriv_kernels_3d(T.dtype, T.device)
    Tx = F.conv3d(F.pad(T, (1,1,0,0,0,0), mode='replicate'), Kx) / dx
    Ty = F.conv3d(F.pad(T, (0,0,1,1,0,0), mode='replicate'), Ky) / dx
    Tz = F.conv3d(F.pad(T, (0,0,0,0,1,1), mode='replicate'), Kz) / dx
    return Tx, Ty, Tz

def train_3d_resnet(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = 'ckpt_resnet3d.pt',
    best_ckpt: str = 'best_resnet3d.pt',
    num_epochs: int = 60,
    batch_size: int = 2,
    lr: float = 1e-3,
    use_grad_loss: bool = True,
    device: str = None,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    train_ds = PackedTensorDataset3D(train_pt)
    val_ds   = PackedTensorDataset3D(val_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    _, _, D, H, W = train_ds.f.shape
    N = D
    dx = 1.0 / (N - 1)

    model = ResNetThermal3D(in_ch=3, out_ch=1).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
    loss_fn = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for f, u in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            f = f.to(device); u = u.to(device)
            bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
            f_s, u_s, bv_s = f, u, bc_val

            net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)  # [B,3,D,H,W]
            pred = model(net_in, bc_mask, bv_s)

            interior = (bc_mask < 0.5)
            mse_field = loss_fn(pred[interior], u_s[interior])

            if use_grad_loss:
                Txp, Typ, Tzp = grad_3d_scalar(pred, dx)
                Txt, Tyt, Tzt = grad_3d_scalar(u_s, dx)
                mse_grad = (loss_fn(Txp[interior], Txt[interior]) +
                            loss_fn(Typ[interior], Tyt[interior]) +
                            loss_fn(Tzp[interior], Tzt[interior])) / 3.0
                loss = 0.5 * mse_field + 0.5 * mse_grad
            else:
                loss = mse_field

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # -------- validation --------
        model.eval()
        tot_loss = tot_rel = 0.0
        with torch.no_grad():
            for f, u in train_loader:
                f = f.to(device);
                u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                f_s, u_s, bv_s = f, u, bc_val
                net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)
                pred = model(net_in, bc_mask, bv_s)
                interior = (bc_mask < 0.5)
                loss = loss_fn(pred[interior], u_s[interior])
                tot_loss += float(loss.item())
                tot_rel += relative_l2(pred[interior], u_s[interior])
        avg_train = tot_loss / max(1, len(train_loader))
        avg_train_rel = tot_rel / max(1, len(train_loader))

        val_loss = val_rel = 0.0
        with torch.no_grad():
            for f, u in val_loader:
                f = f.to(device); u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                f_s, u_s, bv_s = f, u, bc_val
                net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)
                pred = model(net_in, bc_mask, bv_s)
                interior = (bc_mask < 0.5)
                loss = loss_fn(pred[interior], u_s[interior])
                val_loss += float(loss.item())
                val_rel  += relative_l2(pred[interior], u_s[interior])

        avg_val = val_loss / max(1, len(val_loader))
        avg_val_rel = val_rel / max(1, len(val_loader))
        sch.step(avg_val)

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict()}, model_ckpt)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_ckpt)

        print(f"Epoch {epoch+1}/{num_epochs} | Train:{avg_train:.6e}  Rel:{avg_train_rel:.6e} "
              f"| Val:{avg_val:.6e}  Rel:{avg_val_rel:.6e}")

        # log
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val])

if __name__ == "__main__":
    mode = "linear"
    train_pt = f"data/training_data_N32_{mode}.pt"
    val_pt   = f"data/validation_data_N32_{mode}.pt"

    for i in range(1, 2):
        model_path = f"model_{mode}/resnet/ckpt_resnet_{mode}_{i}.pt"
        best_model_path = f"model_{mode}/resnet/best_resnet_{mode}_{i}.pt"
        log_path = f"model_{mode}/resnet/resnet_{i}.csv"

        train_3d_resnet(
            train_pt=train_pt,
            val_pt=val_pt,
            model_ckpt=model_path,
            best_ckpt=best_model_path,
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            use_grad_loss=True,
            log_csv_path=log_path
        )
