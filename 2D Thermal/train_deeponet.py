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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# ===================== 2D utilities =====================

def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

def _build_deriv_kernels_2d(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 3)  # d/dx
    Ky = k.view(1, 1, 3, 1)  # d/dy
    return Kx, Ky

def _build_lap_kernels_2d(dtype, device):
    k2 = torch.tensor([1.0, -2.0, 1.0], dtype=dtype, device=device)
    Kxx = k2.view(1,1,1,3)
    Kyy = k2.view(1,1,3,1)
    return Kxx, Kyy

@torch.no_grad()
def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))

@torch.no_grad()
def feature_gradmag_2d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Thermal feature for nets: |grad T|, returns [B,1,H,W]."""
    assert u.size(1) == 1
    Kx, Ky = _build_deriv_kernels_2d(u.dtype, u.device)
    u_pad_x = F.pad(u, (1, 1, 0, 0), mode='replicate')
    u_pad_y = F.pad(u, (0, 0, 1, 1), mode='replicate')
    Tx = F.conv2d(u_pad_x, Kx) / dx
    Ty = F.conv2d(u_pad_y, Ky) / dx
    return _safe_sqrt(Tx.square() + Ty.square())

@torch.no_grad()
def create_bc_mask_and_value_like_2d(T: torch.Tensor):
    """Dirichlet：top=1, bottom=0, left/right为竖直方向线性梯度(1->0)"""
    B, C, H, W = T.shape
    assert C == 1
    bc_mask = torch.zeros_like(T)
    bc_val  = torch.zeros_like(T)
    vgrad = torch.linspace(1.0, 0.0, H, device=T.device).view(H, 1)
    # top & bottom
    bc_mask[..., 0, :]  = 1.0; bc_val[..., 0, :]  = 1.0
    bc_mask[..., -1, :] = 1.0; bc_val[..., -1, :] = 0.0
    # left & right
    bc_mask[..., :, 0]  = 1.0; bc_val[..., :, 0]  = vgrad.T
    bc_mask[..., :, -1] = 1.0; bc_val[..., :, -1] = vgrad.T
    return bc_mask, bc_val

def interior_mask_from_bc_2d(bc_mask):
    # bc_mask: [B,1,H,W] -> [B,1,H,W] True for free nodes
    return ~(bc_mask.bool())

# ================ Packed dataset: training/validation .pt =================

class PackedTensorDataset2D(Dataset):
    """Loads a packed .pt with keys {'f': [N,1,H,W], 'u': [N,1,H,W]}"""
    def __init__(self, path: str):
        super().__init__()
        data = torch.load(path)
        self.f = data['f'].float()
        self.u = data['u'].float()
        assert self.f.ndim == 4 and self.u.ndim == 4, "Expect shapes [N,1,H,W]"
        assert self.f.shape == self.u.shape, "f and u must match shape"
    def __len__(self):
        return self.f.shape[0]
    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]

# ===================== DeepONet 2D =====================

def fourier_features(x, num_freqs=6, omega=2*np.pi):
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * omega  # [F]
    xb = x.unsqueeze(-1) * freqs  # [..., 2, F]
    return torch.cat([x, torch.sin(xb).reshape(*x.shape[:-1], -1), torch.cos(xb).reshape(*x.shape[:-1], -1)], dim=-1)

class ConvBranch2D(nn.Module):
    def __init__(self, in_ch=5, width=128, p=128):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, stride=2, bias=False),  # /2
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2, bias=False),  # /4
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2, bias=False),  # /8
            nn.GELU(),
        )
        self.readout = nn.Linear(width, p, bias=False)

    def forward(self, x):
        # x: [B,in_ch,H,W]
        h = self.net(x)               # [B,width,H/8,W/8]
        h = h.mean(dim=(2,3))         # GAP -> [B,width]
        return self.readout(h)        # [B,p]

class TrunkMLP2D(nn.Module):
    """Map 2D coords -> p features for scalar output."""
    def __init__(self, p=128, num_freqs=10, hidden=256, omega=2*np.pi):
        super().__init__()
        self.p = p
        self.num_freqs = num_freqs
        self.omega = omega
        in_dim = 2 + 2*2*num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p)
        )
    def forward(self, coords01):
        z = fourier_features(coords01, self.num_freqs, self.omega)
        return self.mlp(z)  # [..., p]

class DeepONet2D(nn.Module):
    def __init__(self, N=32, p=128, branch_width=128, trunk_hidden=256, trunk_freqs=12, in_ch=5):
        super().__init__()
        self.N  = N
        self.dx = 1.0 / (N - 1)
        self.p  = p
        self.in_ch = in_ch

        # Branch & Trunk
        self.branch = ConvBranch2D(in_ch=in_ch, width=branch_width, p=p)
        self.trunk  = TrunkMLP2D(p=p, num_freqs=trunk_freqs, hidden=trunk_hidden)

        # precompute grid coords [1,2,H,W] in [0,1]
        lin = torch.linspace(0, 1, N)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
        self.register_buffer("grid01", grid, persistent=False)

        # coords list for dense evaluation
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [N^2,2]
        self.register_buffer("coords01_list", coords, persistent=False)

    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask=None, bc_val=None):
        # f: [B,1,H,W]; bc_*: [B,1,H,W]
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)
        coords = self.grid01.expand(f.size(0), -1, -1, -1)           # [B,2,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)   # [B,5,H,W]
        b = self.branch(branch_in)                                    # [B,p]

        # trunk on dense grid -> [H*W,p] -> [p,H,W]
        t = self.trunk(self.coords01_list)                            # [N^2,p]
        trunk_out = t.view(self.N, self.N, self.p).permute(2,0,1).contiguous()  # [p,H,W]

        u = torch.einsum('bp,phw->bhw', b, trunk_out).unsqueeze(1)    # [B,1,H,W]
        u = self._apply_bc(u, bc_mask, bc_val)
        return u

    def forward_points(self, f, bc_mask, bc_val, xy):
        """
        xy: [B,M,2] in [0,1]
        returns T(xy): [B,M,1]   (no BC applied here)
        """
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)

        coords = self.grid01.expand(f.size(0), -1, -1, -1)           # [B,2,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)   # [B,5,H,W]
        b = self.branch(branch_in)                                    # [B,p]

        B, M, _ = xy.shape
        feats = self.trunk(xy.reshape(B*M, 2)).view(B, M, self.p)     # [B,M,p]
        u_pts = torch.einsum('bmp,bp->bm', feats, b).unsqueeze(-1)    # [B,M,1]
        return u_pts

# ===================== physics residual (2D thermal) =====================

def poisson_residual_2d(T, f, kappa=1.0, dx=1.0):
    # residual r = kappa * ΔT + f
    Kxx, Kyy = _build_lap_kernels_2d(T.dtype, T.device)
    Txp = F.conv2d(F.pad(T, (1,1,0,0), mode='replicate'), Kxx)
    Typ = F.conv2d(F.pad(T, (0,0,1,1), mode='replicate'), Kyy)
    lap = (Txp + Typ) / (dx*dx)
    return kappa * lap + f

# ===================== random points / gathering =====================

def random_indices_2d(B, M, N, device, pad=0):
    lo, hi = pad, N - pad
    iy = torch.randint(lo, hi, (B, M), device=device)
    ix = torch.randint(lo, hi, (B, M), device=device)
    return iy, ix

def make_coords_from_indices_2d(ix, iy, N, space="[0,1]"):
    if space == "[-1,1]":
        s = 2.0 / (N - 1)
        x = -1.0 + ix.float() * s
        y = -1.0 + iy.float() * s
    else:
        s = 1.0 / (N - 1)
        x = ix.float() * s
        y = iy.float() * s
    return torch.stack([x, y], dim=-1)  # [B,M,2]

def gather_points_2d(u, iy, ix):
    # u: [B,1,H,W] -> [B,M,1]
    B, C, H, W = u.shape
    b = torch.arange(B, device=u.device).view(B, 1).expand_as(iy)
    vals = u[b, 0, iy, ix]  # [B,M]
    return vals.unsqueeze(-1)

def gather_mask_2d(mask2d, iy, ix):
    # mask2d: [B,1,H,W] -> [B,M]
    B, _, H, W = mask2d.shape
    b = torch.arange(B, device=mask2d.device).view(B, 1).expand_as(iy)
    m = mask2d[b, 0, iy, ix]
    return m.float()

def sample_boundary_indices_2d(bc_mask, Mb):
    # bc_mask: [B,1,H,W] -> pick Mb indices where mask==1
    B, _, H, W = bc_mask.shape
    iy = torch.zeros(B, Mb, dtype=torch.long, device=bc_mask.device)
    ix = torch.zeros_like(iy)
    for b in range(B):
        mask = bc_mask[b,0]  # [H,W]
        idx = mask.nonzero(as_tuple=False)  # [K,2] (y,x)
        if idx.numel() == 0:
            iy[b] = torch.randint(0, H, (Mb,), device=bc_mask.device)
            ix[b] = torch.randint(0, W, (Mb,), device=bc_mask.device)
        else:
            sel = idx[torch.randint(0, idx.shape[0], (Mb,), device=bc_mask.device)]
            iy[b], ix[b] = sel[:, 0], sel[:, 1]
    return iy, ix

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

# ===================== superposition (optional) =====================

def superposition_mix(f, u, p=0.5):
    if torch.rand(1).item() >= p or f.size(0) < 2:
        return f, u
    idx = torch.randperm(f.size(0), device=f.device)
    a = torch.rand(1, device=f.device)
    f2, u2 = f[idx], u[idx]
    f_mix = a * f + (1 - a) * f2
    u_mix = a * u + (1 - a) * u2
    return f_mix, u_mix

# ===================== training loop (2D thermal) =====================

def train_thermal_2d(
    train_pt: str,
    val_pt: str,
    model_path: str,
    best_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    points_per_sample: int = 4096,
    coord_space: str = "[0,1]",
    kappa: float = 1.0,
    alpha_phys: float = 0.0,
    alpha_bc: float = 0.0,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets / loaders
    train_ds = PackedTensorDataset2D(train_pt)
    val_ds   = PackedTensorDataset2D(val_pt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # grid
    _, _, H, W = train_ds.f.shape
    N = H

    # model
    model = DeepONet2D(N=N, p=128, branch_width=136, trunk_hidden=260, trunk_freqs=14, in_ch=5).to(device)

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
            bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
            f, u = superposition_mix(f, u, p=0.5)
            f_s, u_s, bv_s = f, u, bc_val
            B = f_s.size(0)
            M = points_per_sample
            iy, ix = random_indices_2d(B, M, N, device=f_s.device, pad=0)
            xy = make_coords_from_indices_2d(ix, iy, N, space=coord_space)             # [B,M,2]
            pred_pts = model.forward_points(f_s, bc_mask, bv_s, xy)                    # [B,M,1] (no BC)
            u_pts    = gather_points_2d(u_s, iy, ix)                                   # [B,M,1]
            free_pts = gather_mask_2d(interior_mask_from_bc_2d(bc_mask).float(), iy, ix)  # [B,M]
            Mb = max(M // 8, 512)
            biy, bix = sample_boundary_indices_2d(bc_mask, Mb)
            xy_b = make_coords_from_indices_2d(bix, biy, N, space=coord_space)
            pred_b  = model.forward_points(f_s, bc_mask, bv_s, xy_b)                   # [B,Mb,1]
            bc_vals = gather_points_2d(bv_s, biy, bix)                                  # [B,Mb,1]
            bc_mask_pts = gather_points_2d(bc_mask, biy, bix).squeeze(-1)               # [B,Mb]
            bc_loss = masked_point_mse_scalar(pred_b, bc_vals, bc_mask_pts)
            u_full = model(f_s, bc_mask, bv_s)
            interior = interior_mask_from_bc_2d(bc_mask)
            data_loss = ((u_full - u_s)[interior]**2).mean()
            if alpha_phys > 0.0:
                res = poisson_residual_2d(u_full, f_s, kappa=kappa, dx=model.dx)   # [B,1,H,W]
                w = interior.float()
                num = (res.pow(2) * w).sum()
                den = w.sum().clamp_min(1.0)
                phys_loss = num / den
            else:
                phys_loss = torch.tensor(0.0, device=f.device)

            loss = data_loss + alpha_bc * bc_loss + alpha_phys * phys_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                    f_s, u_s, bv_s = f, u, bc_val
                    pred = model(f_s, bc_mask, bv_s)
                    interior = interior_mask_from_bc_2d(bc_mask)

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
            'scheduler': scheduler.state_dict(),
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
    mode = "linear"
    train_pt = f"data/training_data_N32_{mode}.pt"
    val_pt   = f"data/validation_data_N32_{mode}.pt"

    for i in range(1, 6):
        model_path = f"model_{mode}/deeponet/ckpt_2d_deeponet_{mode}_{i}.pt"
        best_model_path = f"model_{mode}/deeponet/best_2d_deeponet_{mode}_{i}.pt"
        log_path = f"model_{mode}/deeponet/deeponet_{mode}_{i}.csv"

        train_thermal_2d(
            train_pt=train_pt,
            val_pt=val_pt,
            model_path=model_path,
            best_model_path=best_model_path,
            num_epochs=50,
            batch_size=10,
            points_per_sample=4096,
            coord_space="[0,1]",
            kappa=1.0,
            alpha_phys=0.0,
            alpha_bc=0.0,
            log_csv_path=log_path
        )
