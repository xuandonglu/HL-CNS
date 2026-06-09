
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


# ===================== DeepONet 3D =====================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fourier_features(x, num_freqs=6, omega=2*np.pi):
    """
    x: [..., 3] in [0,1]
    returns: [..., 3 + 2*3*num_freqs]
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * omega  # [F]
    xb = x.unsqueeze(-1) * freqs  # [..., 3, F]
    return torch.cat([x, torch.sin(xb).reshape(*x.shape[:-1], -1), torch.cos(xb).reshape(*x.shape[:-1], -1)], dim=-1)

class ConvBranch3D(nn.Module):
    """
    3D convolutional branch.
    Input channels: f(1) + bc_mask(1) + bc_val(1) + coords(3) = 6 by default
    """
    def __init__(self, in_ch=6, width=128, p=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, width, 3, padding=1, stride=2, bias=False),  # /2
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, stride=2, bias=False),  # /4
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, stride=2, bias=False),  # /8
            nn.GELU(),
        )
        self.readout = nn.Linear(width, p, bias=False)

    def forward(self, x):
        # x: [B,in_ch,D,H,W]
        h = self.net(x)               # [B,width,D/8,H/8,W/8]
        h = h.mean(dim=(2,3,4))       # GAP -> [B,width]
        return self.readout(h)        # [B,p]

class TrunkMLP3D(nn.Module):
    """Map 3D coords -> p features for scalar output."""
    def __init__(self, p=128, num_freqs=10, hidden=256, omega=2*np.pi):
        super().__init__()
        self.p = p
        self.num_freqs = num_freqs
        self.omega = omega
        in_dim = 3 + 2*3*num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p)
        )
    def forward(self, coords01):
        z = fourier_features(coords01, self.num_freqs, self.omega)
        return self.mlp(z)  # [..., p]

class DeepONet3D(nn.Module):
    def __init__(self, N=32, p=128, branch_width=128, trunk_hidden=256, trunk_freqs=12, in_ch=6):
        super().__init__()
        self.N  = N
        self.dx = 1.0 / (N - 1)
        self.p  = p
        self.in_ch = in_ch

        # Branch & Trunk
        self.branch = ConvBranch3D(in_ch=in_ch, width=branch_width, p=p)
        self.trunk  = TrunkMLP3D(p=p, num_freqs=trunk_freqs, hidden=trunk_hidden)

        # precompute grid coords [1,3,D,H,W] in [0,1]
        lin = torch.linspace(0, 1, N)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # [1,3,D,H,W]
        self.register_buffer("grid01", grid, persistent=False)

        # coords list for dense evaluation
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N^3,3]
        self.register_buffer("coords01_list", coords, persistent=False)

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask=None, bc_val=None):
        # f: [B,1,D,H,W]; bc_*: [B,1,D,H,W]
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)
        coords = self.grid01.expand(f.size(0), -1, -1, -1, -1)        # [B,3,D,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)    # [B,6,D,H,W]
        b = self.branch(branch_in)                                    # [B,p]

        # trunk on dense grid -> [N^3,p] -> [p,D,H,W]
        t = self.trunk(self.coords01_list)                             # [N^3,p]
        trunk_out = t.view(self.N, self.N, self.N, self.p).permute(3,0,1,2).contiguous()  # [p,D,H,W]

        u = torch.einsum('bp,pdhw->bdhw', b, trunk_out).unsqueeze(1)  # [B,1,D,H,W]
        u = self._apply_bc(u, bc_mask, bc_val)
        return u

    def forward_points(self, f, bc_mask, bc_val, xyz):
        """
        xyz: [B,M,3] in [0,1]
        returns T(xyz): [B,M,1]   (no BC applied here)
        """
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)

        coords = self.grid01.expand(f.size(0), -1, -1, -1, -1)          # [B,3,D,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)      # [B,6,D,H,W]
        b = self.branch(branch_in)                                      # [B,p]

        B, M, _ = xyz.shape
        feats = self.trunk(xyz.reshape(B*M, 3)).view(B, M, self.p)      # [B,M,p]
        u_pts = torch.einsum('bmp,bp->bm', feats, b).unsqueeze(-1)      # [B,M,1]
        return u_pts

# ===================== physics residual (3D thermal) =====================
def poisson_residual_3d(T, f, kappa=1.0, dx=1.0):
    # residual r = kappa * ΔT + f
    Kxx, Kyy, Kzz = _build_lap_kernels_3d(T.dtype, T.device)
    Txp = F.conv3d(F.pad(T, (1,1,0,0,0,0), mode='replicate'), Kxx)
    Typ = F.conv3d(F.pad(T, (0,0,1,1,0,0), mode='replicate'), Kyy)
    Tzp = F.conv3d(F.pad(T, (0,0,0,0,1,1), mode='replicate'), Kzz)
    lap = (Txp + Typ + Tzp) / (dx*dx)
    return kappa * lap + f

# ===================== training loop (3D thermal) =====================
def train_thermal_3d_deeponet(
    train_pt: str,
    val_pt: str,
    model_path: str,
    best_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 4,
    points_per_sample: int = 8192,
    coord_space: str = "[0,1]",
    kappa: float = 1.0,
    alpha_phys: float = 0.0,  # physics residual weight
    alpha_bc: float = 0.0,    # BC penalty weight (via forward_points)
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets / loaders
    train_ds = PackedTensorDataset3D(train_pt)
    val_ds   = PackedTensorDataset3D(val_pt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # grid
    _, _, D, H, W = train_ds.f.shape
    assert D == H == W, "Assumes cubic grids"
    N = D

    # model
    model = DeepONet3D(N=N, p=64, branch_width=80, trunk_hidden=80, trunk_freqs=12, in_ch=6).to(device)

    # total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    loss_fn = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_data = 0.0
        total_phys = 0.0
        total_bc   = 0.0
        total_rel  = 0.0

        for f, u in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            f = f.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            # BC
            bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)

            # scale
            f_s, u_s, bv_s = f, u, bc_val

            B = f_s.size(0)
            M = points_per_sample
            iz, iy, ix = random_indices_3d(B, M, N, device=f_s.device, pad=0)
            xyz = make_coords_from_indices_3d(ix, iy, iz, N, space=coord_space)             # [B,M,3]

            pred_pts = model.forward_points(f_s, bc_mask, bv_s, xyz)                        # [B,M,1] (no BC)
            u_pts    = gather_points_3d(u_s, iz, iy, ix)                                     # [B,M,1]
            free_pts = gather_mask_3d(interior_mask_from_bc_3d(bc_mask).float(), iz, iy, ix) # [B,M]

            Mb = max(M // 8, 1024)
            biz, biy, bix = sample_boundary_indices_3d(bc_mask, Mb)
            xyz_b = make_coords_from_indices_3d(bix, biy, biz, N, space=coord_space)
            pred_b  = model.forward_points(f_s, bc_mask, bv_s, xyz_b)                        # [B,Mb,1]
            bc_vals = gather_points_3d(bv_s, biz, biy, bix)                                   # [B,Mb,1]
            bc_mask_pts = gather_points_3d(bc_mask, biz, biy, bix).squeeze(-1)                # [B,Mb]
            bc_loss = masked_point_mse_scalar(pred_b, bc_vals, bc_mask_pts)

            u_full = model(f_s, bc_mask, bv_s)
            interior = interior_mask_from_bc_3d(bc_mask)
            data_loss = ((u_full - u_s)[interior]**2).mean()

            if alpha_phys > 0.0:
                res = poisson_residual_3d(u_full, f_s, kappa=kappa, dx=model.dx)   # [B,1,D,H,W]
                w = interior.float()
                num = (res.pow(2) * w).sum()
                den = w.sum().clamp_min(1.0)
                phys_loss = num / den
            else:
                phys_loss = torch.tensor(0.0, device=f.device)

            loss = data_loss + alpha_bc * bc_loss + alpha_phys * phys_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_data += float(data_loss.item())
            total_bc   += float(bc_loss.item())
            total_phys += float(phys_loss.item())
            with torch.no_grad():
                total_rel += rel_l2_points_scalar(pred_pts, u_pts, free_pts)

        avg_train_mse = total_data / max(1, len(train_loader))
        avg_train_rel = total_rel / max(1, len(train_loader))

        model.eval()
        def eval_loader(loader):
            sum_num = 0.0
            sum_den = 0.0
            sum_mse = 0.0
            n_batches = 0
            with torch.no_grad():
                for f, u in loader:
                    f = f.to(device); u = u.to(device)
                    bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                    f_s, u_s, bv_s = f, u, bc_val
                    pred = model(f_s, bc_mask, bv_s)
                    interior = interior_mask_from_bc_3d(bc_mask)

                    e = (pred - u_s)[interior]
                    t = (u_s)[interior]
                    sum_num += (e*e).sum().item()
                    sum_den += (t*t).sum().item()
                    sum_mse += (e*e).mean().item()
                    n_batches += 1
            avg_mse = sum_mse / max(n_batches, 1)
            rel_l2  = (sum_num ** 0.5) / max(sum_den ** 0.5, 1e-12)
            return avg_mse, rel_l2

        avg_val_mse, avg_val_rel = eval_loader(val_loader)
        if scheduler: scheduler.step(avg_val_mse)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train MSE:{avg_train_mse:.6f} | RelL2≈{avg_train_rel:.6f}")
        print(f"  Val   MSE:{avg_val_mse:.6f} | Val RelL2:{avg_val_rel:.6f}")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, model_path)
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved")

        # log
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_mse, avg_val_mse])

# ============================ run ============================
if __name__ == "__main__":
    mode = "decrease"
    train_pt = f"data/training_data_N32_{mode}.pt"
    val_pt   = f"data/validation_data_N32_{mode}.pt"

    for i in range(1, 2):
        model_path = f"model_{mode}/deeponet/ckpt_deeponet_{mode}_{i}.pt"
        best_model_path = f"model_{mode}/deeponet/best_deeponet_{mode}_{i}.pt"
        log_path = f"model_{mode}/deeponet/deeponet_{i}.csv"

        train_thermal_3d_deeponet(
            train_pt=train_pt,
            val_pt=val_pt,
            model_path=model_path,
            best_model_path=best_model_path,
            num_epochs=100,
            batch_size=10,
            points_per_sample=8192,
            coord_space="[0,1]",
            kappa=1.0,
            alpha_phys=0.0,
            alpha_bc=0.0,
            log_csv_path=log_path
        )
