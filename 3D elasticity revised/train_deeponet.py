import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib.pyplot as plt

# If plot_tools is not available, we'll create dummy functions
try:
    from plot_tools import compute_strain_tensor, plot_comparison_3d, plot_stress_divergence
except ImportError:
    def compute_strain_tensor(*args, **kwargs):
        return None
    def plot_comparison_3d(*args, **kwargs):
        print("plot_comparison_3d not available")
    def plot_stress_divergence(*args, **kwargs):
        print("plot_stress_divergence not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# ===================== utility: random point sampling =====================

def make_coords_from_indices(ix, iy, iz, N, space="[0,1]"):
    if space == "[-1,1]":
        s = 2.0 / (N - 1)
        x = -1.0 + ix.float() * s
        y = -1.0 + iy.float() * s
        z = -1.0 + iz.float() * s
    else:  # [0,1]
        s = 1.0 / (N - 1)
        x = ix.float() * s
        y = iy.float() * s
        z = iz.float() * s
    return torch.stack([x, y, z], dim=-1)  # [B,M,3]

def random_indices(B, M, N, device, pad=0):
    lo, hi = pad, N - pad
    iz = torch.randint(lo, hi, (B, M), device=device)
    iy = torch.randint(lo, hi, (B, M), device=device)
    ix = torch.randint(lo, hi, (B, M), device=device)
    return iz, iy, ix

def gather_points(u, iz, iy, ix):
    # u: [B,C,D,H,W], iz/iy/ix: [B,M] -> [B,C,M]
    B, C, D, H, W = u.shape
    b = torch.arange(B, device=u.device).view(B, 1).expand_as(iz)
    return u[b, :, iz, iy, ix]

def gather_mask(mask3d, iz, iy, ix):
    # mask3d: [B,D,H,W] bool/float -> [B,M] float
    B, D, H, W = mask3d.shape
    b = torch.arange(B, device=mask3d.device).view(B, 1).expand_as(iz)
    m = mask3d[b, iz, iy, ix]
    return m.float()

def interior_mask_from_bc(bc_mask):
    # bc_mask: [B,3,D,H,W] -> [B,D,H,W] (True = free)
    return ~(bc_mask.bool().any(dim=1))

def _ensure_BM3(x: torch.Tensor) -> torch.Tensor:
    """Return [B, M, 3] from [B,3,M] or [B,M,3]."""
    if x.ndim != 3:
        raise ValueError(f"_ensure_BM3 expected 3D, got {tuple(x.shape)}")
    B, A, C = x.shape
    if C == 3:            # [B, M, 3]
        return x
    if A == 3:            # [B, 3, M] -> [B, M, 3]
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"_ensure_BM3 cannot find channels=3 in last dim; got {tuple(x.shape)}")

def _ensure_BM_mask(mask_pts: torch.Tensor) -> torch.Tensor:
    """
    Accept mask in any of:
      [B, M]            (scalar per point)          -> return as-is
      [B, 3, M]         (per-comp mask)             -> reduce any over channel -> [B,M]
      [B, M, 3]         (per-comp mask, last dim)   -> reduce any over last    -> [B,M]
    """
    if mask_pts.ndim == 2:
        return mask_pts
    if mask_pts.ndim == 3:
        if mask_pts.shape[1] == 3:   # [B,3,M]
            return (mask_pts > 0.5).any(dim=1).float()
        if mask_pts.shape[2] == 3:   # [B,M,3]
            return (mask_pts > 0.5).any(dim=2).float()
    raise ValueError(f"mask_pts must be [B,M], [B,3,M], or [B,M,3], got {tuple(mask_pts.shape)}")

def masked_point_mse(pred_pts: torch.Tensor,
                     true_pts: torch.Tensor,
                     mask_pts: torch.Tensor) -> torch.Tensor:
    """
    pred_pts, true_pts: [B,3,M] or [B,M,3]
    mask_pts: [B,M] or per-component mask ([B,3,M]/[B,M,3]) – reduced internally.
    """
    pred = _ensure_BM3(pred_pts)                 # [B, M, 3]
    true = _ensure_BM3(true_pts)                 # [B, M, 3]
    w    = _ensure_BM_mask(mask_pts).unsqueeze(-1)  # [B, M, 1]
    if pred.shape[:2] != true.shape[:2] or pred.shape[:2] != w.shape[:2]:
        raise RuntimeError(f"shape mismatch: pred {tuple(pred.shape)}, true {tuple(true.shape)}, mask {tuple(w.shape)}")
    num = ((pred - true) ** 2 * w).sum()
    den = (w.sum() * pred.size(-1)).clamp_min(1.0)  # *3 comps
    return num / den

@torch.no_grad()
def rel_l2_points(pred_pts: torch.Tensor, true_pts: torch.Tensor, mask_pts: torch.Tensor) -> float:
    pred = _ensure_BM3(pred_pts)
    true = _ensure_BM3(true_pts)
    w    = _ensure_BM_mask(mask_pts).unsqueeze(-1)  # [B,M,1]
    num = torch.linalg.vector_norm(((pred - true) * w).reshape(pred.size(0), -1), dim=1).sum()
    den = torch.linalg.vector_norm((true * w).reshape(pred.size(0), -1), dim=1).sum() + 1e-12
    return (num / den).item()



def sample_boundary_indices(bc_mask, M_b):
    # bc_mask: [B,3,D,H,W] -> pick M_b indices where any channel is 1
    B, _, D, H, W = bc_mask.shape
    iz = torch.zeros(B, M_b, dtype=torch.long, device=bc_mask.device)
    iy = torch.zeros_like(iz)
    ix = torch.zeros_like(iz)
    for b in range(B):
        mask = bc_mask[b].any(dim=0)  # [D,H,W]
        idx = mask.nonzero(as_tuple=False)  # [K,3] (z,y,x)
        if idx.numel() == 0:
            iz[b] = torch.randint(0, D, (M_b,), device=bc_mask.device)
            iy[b] = torch.randint(0, H, (M_b,), device=bc_mask.device)
            ix[b] = torch.randint(0, W, (M_b,), device=bc_mask.device)
        else:
            sel = idx[torch.randint(0, idx.shape[0], (M_b,), device=bc_mask.device)]
            iz[b], iy[b], ix[b] = sel[:, 0], sel[:, 1], sel[:, 2]
    return iz, iy, ix

# ===================== physics residual (Navier–Cauchy) =====================

def _d_dx(s, dx):
    sp = F.pad(s, (1,1,0,0,0,0), mode='replicate')
    k = s.new_tensor([-0.5, 0.0, 0.5]).view(1,1,1,1,3).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / dx

def _d_dy(s, dx):
    sp = F.pad(s, (0,0,1,1,0,0), mode='replicate')
    k = s.new_tensor([-0.5, 0.0, 0.5]).view(1,1,1,3,1).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / dx

def _d_dz(s, dx):
    sp = F.pad(s, (0,0,0,0,1,1), mode='replicate')
    k = s.new_tensor([-0.5, 0.0, 0.5]).view(1,1,3,1,1).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / dx

def _d2_dx2(s, dx):
    sp = F.pad(s, (1,1,0,0,0,0), mode='replicate')
    k = s.new_tensor([1.0,-2.0,1.0]).view(1,1,1,1,3).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / (dx*dx)

def _d2_dy2(s, dx):
    sp = F.pad(s, (0,0,1,1,0,0), mode='replicate')
    k = s.new_tensor([1.0,-2.0,1.0]).view(1,1,1,3,1).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / (dx*dx)

def _d2_dz2(s, dx):
    sp = F.pad(s, (0,0,0,0,1,1), mode='replicate')
    k = s.new_tensor([1.0,-2.0,1.0]).view(1,1,3,1,1).repeat(s.size(1), 1, 1, 1, 1)
    return F.conv3d(sp, k, groups=s.size(1)) / (dx*dx)

def _div(u, dx):
    ux, uy, uz = u[:,0:1], u[:,1:2], u[:,2:3]
    return _d_dx(ux,dx) + _d_dy(uy,dx) + _d_dz(uz,dx)

def _grad_scalar(s, dx):
    gx = _d_dx(s, dx); gy = _d_dy(s, dx); gz = _d_dz(s, dx)
    return torch.cat([gx,gy,gz], dim=1)

def _lap_vec(u, dx):
    return _d2_dx2(u,dx) + _d2_dy2(u,dx) + _d2_dz2(u,dx)

def navier_cauchy_residual(u, f, E=1.0, nu=0.3, dx=1.0):
    mu  = E/(2*(1+nu))
    lam = E*nu/((1+nu)*(1-2*nu) + 1e-8)
    lap_u = _lap_vec(u, dx)
    div_u = _div(u, dx)
    grad_div_u = _grad_scalar(div_u, dx)
    # sign convention matches training targets (u,f are in same scaled units)
    return mu*lap_u + (lam+mu)*grad_div_u + f

# ===================== dataset & scaler =====================

class ElasticityDataset(Dataset):
    def __init__(self, data_dir, prefix="sample_", max_index=None):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])
        if max_index is not None:
            self.file_list = [f for f in self.file_list 
                              if int(f[len(prefix):len(prefix)+4]) < max_index]

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        d = torch.load(os.path.join(self.data_dir, self.file_list[idx]))
        return d['f'].float(), d['u'].float(), d['bc_mask'].float(), d['bc_val'].float()

class GlobalScaler:
    def __init__(self, eps=1e-12, quantile=0.99):
        self.eps = eps; self.quantile = quantile; self.S = None

    @torch.no_grad()
    def fit_from_loader(self, loader, max_batches=8, device="cpu"):
        mags = []
        for b, (f, u, _, _) in enumerate(loader):
            u = u.to(device, non_blocking=True)
            m = u.abs().reshape(u.shape[0], -1).amax(dim=1).cpu()
            mags.append(m)
            if (b+1) >= max_batches: break
        mags = torch.cat(mags) if mags else torch.tensor([1.0])
        q = torch.quantile(mags, torch.as_tensor(self.quantile))
        self.S = float(max(q.item(), self.eps))
        return self

    def scale_in(self, f, u, bc_val):
        S = self.S
        return f / S, u / S, bc_val / S

    def u_to_phys(self, u_scaled): return u_scaled * self.S

def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

# ===================== DeepONet with spatially aware branch =====================

def fourier_features(x, num_freqs=6, omega=2*np.pi):
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * omega
    xb = x.unsqueeze(-1) * freqs  # [N,3,F]
    return torch.cat([x, torch.sin(xb).reshape(x.shape[0], -1), torch.cos(xb).reshape(x.shape[0], -1)], dim=-1)

class ConvBranch3D(nn.Module):
    """
    Convolutional branch; channel count is configurable.
    Expects input [B, in_ch, D, H, W]. We'll pass in_ch=9 (f, mask, bc_val).
    """
    def __init__(self, in_ch=9, width=128, p=128):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, width, 3, padding=1, stride=2, bias=False),  # D,H,W -> /2
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, stride=2, bias=False),  # /4
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, stride=2, bias=False),  # /8
            nn.GELU(),
        )
        self.readout = nn.Linear(width, p, bias=False)

    def forward(self, x):
        # x: [B, in_ch, D, H, W]
        h = self.net(x)                  # [B,width,D/8,H/8,W/8]
        h = h.mean(dim=(2,3,4))          # GAP -> [B,width]
        return self.readout(h)           # [B,p]

class TrunkMLP(nn.Module):
    def __init__(self, p=128, num_freqs=10, hidden=256, omega=2*np.pi):
        super().__init__()
        self.p = p
        self.num_freqs = num_freqs
        self.omega = omega
        in_dim = 3 + 2*3*num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3*p)
        )

    def forward(self, coords01):
        z = fourier_features(coords01, self.num_freqs, self.omega)
        return self.mlp(z)  # [N, 3*p]

class DeepONet3D(nn.Module):
    def __init__(self, N=32, p=128, branch_width=128, trunk_hidden=256, trunk_freqs=12, in_ch=9):
        super().__init__()
        self.N  = N
        self.dx = 1.0 / (N - 1)
        self.p  = p
        self.in_ch = in_ch

        # ---- Branch & Trunk ----
        self.branch = ConvBranch3D(in_ch=in_ch, width=branch_width, p=p)  # <= uses in_ch
        self.trunk  = TrunkMLP(p=p, num_freqs=trunk_freqs, hidden=trunk_hidden)

        # precompute grid coords [1,3,D,H,W] in [0,1]
        lin = torch.linspace(0, 1, N)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # [1,3,D,H,W]
        self.register_buffer("grid01", grid, persistent=False)

        # coords list for dense evaluation
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N^3,3]
        self.register_buffer("coords01_list", coords, persistent=False)

        self.solver = SimpleNamespace(dx=self.dx, _last_correction=None)

    def _apply_bc(self, u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    def forward(self, f, bc_mask=None, bc_val=None):
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)
        
        # Branch input: concat coords (broadcast) so branch keeps location info
        coords = self.grid01.expand(f.size(0), -1, -1, -1, -1)  # [B,3,D,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)  # [B,12,D,H,W]
        b = self.branch(branch_in)                                   # [B,p]

        # trunk basis on dense grid -> [p,3,D,H,W]
        trunk_out = self.trunk(self.coords01_list)                   # [N^3,3*p]
        trunk_out = trunk_out.view(self.N, self.N, self.N, 3, self.p).permute(4, 3, 0, 1, 2).contiguous()

        u = torch.einsum('bp,pchwd->bchwd', b, trunk_out)            # [B,3,D,H,W]
        u = self._apply_bc(u, bc_mask, bc_val)
        self.solver._last_correction = torch.zeros_like(u)
        return u

    def forward_points(self, f, bc_mask, bc_val, xyz):
        """
        xyz: [B,M,3] in [0,1] (or the same space used to build the trunk)
        returns u(xyz): [B,M,3]
        """
        if bc_mask is None: bc_mask = torch.zeros_like(f)
        if bc_val  is None: bc_val  = torch.zeros_like(f)
        
        # Branch input: concat coords (broadcast) so branch keeps location info
        coords = self.grid01.expand(f.size(0), -1, -1, -1, -1)  # [B,3,D,H,W]
        branch_in = torch.cat([f, bc_mask, bc_val, coords], dim=1)  # [B,12,D,H,W]
        b = self.branch(branch_in)                                   # [B,p]
        
        B, M, _ = xyz.shape
    
        # Trunk: flatten all coords, map -> [B*M,3*p], then reshape to [B,M,3,p]
        z = xyz.reshape(B*M, 3)
        t = self.trunk(z)                                   # [B*M, 3*p]
        feats = t.view(B, M, 3, self.p)                     # [B,M,3,p]
    
        # Compose: sum over p  -> [B,M,3]
        u_pts = torch.einsum('bmcp,bp->bmc', feats, b)      # [B,M,3]
        return u_pts

# ===================== training loop =====================

def superposition_mix(f, u, bc_mask, bc_val, p=0.5):
    if torch.rand(1).item() >= p or f.size(0) < 2:
        return f, u, bc_mask, bc_val
    idx = torch.randperm(f.size(0), device=f.device)
    a = torch.rand(1, device=f.device)
    f2, u2 = f[idx], u[idx]
    f_mix = a * f + (1 - a) * f2
    u_mix = a * u + (1 - a) * u2
    return f_mix, u_mix, bc_mask, bc_val  # keep BCs of the first

def train_elasticity(
    data_dir, model_path, best_model_path, num_epochs=20, batch_size=8,
    log_csv_path="training_log.csv", points_per_sample=8192, coord_space="[0,1]",
    alpha_phys=0.5, alpha_bc=1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets / loaders
    train_dataset = ElasticityDataset(data_dir, prefix="sample_")
    val_dataset   = ElasticityDataset(data_dir, prefix="validation_")
    ood_dataset   = ElasticityDataset(data_dir, prefix="dualforce_")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader   = DataLoader(ood_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model
    f0, u0, _, _ = train_dataset[0]
    N = f0.shape[-1]
    model = DeepONet3D(N=N, p=128, branch_width=144, trunk_hidden=288, trunk_freqs=14, in_ch=12).to(device)

    # total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # global scale
    scaler = GlobalScaler(quantile=0.99)
    start_epoch = 0
    best_val = float('inf')

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        scaler.S = float(ckpt.get('scale_S', 0.0)) or None
        if scaler.S is None:
            scaler.fit_from_loader(train_loader, max_batches=8, device=device)
        print(f"Resuming from epoch {start_epoch}; S = {scaler.S:.3e}")
    else:
        print("No saved model found. Starting training from scratch.")
        scaler.fit_from_loader(train_loader, max_batches=8, device=device)
        print(f"[Scaler] Using global scale S = {scaler.S:.3e} (u,f,bc_val all divided by S)")

    # validation sampling size
    val_M = min(N*N*N, 16384)
    loss_fn = nn.MSELoss()
    with open(log_csv_path, mode='w', newline='') as log_file:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_data = 0.0
            total_phys = 0.0
            total_bc   = 0.0
            total_rel  = 0.0

            for f, u, bc_mask, bc_val in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                f  = f.to(device, non_blocking=True)
                u  = u.to(device, non_blocking=True)
                m  = bc_mask.to(device, non_blocking=True)
                bv = bc_val.to(device, non_blocking=True)

                f, u, m, bv = superposition_mix(f, u, m, bv, p=0.5)

                # normalized
                f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                B = f_s.size(0)
                M = points_per_sample

                # random interior samples
                # ----- interior/data samples -----
                iz, iy, ix = random_indices(B, M, N, device=f_s.device, pad=0)
                xyz = make_coords_from_indices(ix, iy, iz, N, space=coord_space)       # [B, M, 3]
                
                pred_pts = _ensure_BM3(model.forward_points(f_s, m, bv_s, xyz))        # [B, M, 3]
                u_pts    = _ensure_BM3(gather_points(u_s, iz, iy, ix))                  # [B, M, 3]
                free_pts = gather_mask(interior_mask_from_bc(m).float(), iz, iy, ix)    # [B, M]
                
                data_loss = masked_point_mse(pred_pts, u_pts, free_pts)

                # boundary samples
                # ----- boundary samples *on actual BC* -----
                Mb = max(M // 8, 1024)
                biz, biy, bix = sample_boundary_indices(m, Mb)                     # <-- boundary-only
                xyz_b = make_coords_from_indices(bix, biy, biz, N, space=coord_space)  # [B, Mb, 3]
                
                pred_b  = model.forward_points(f_s, m, bv_s, xyz_b)                # [B, Mb, 3] (or [B,3,Mb], both ok)
                bc_vals = gather_points(bv_s, biz, biy, bix)                       # [B, 3, Mb] (values we must match)
                bc_mask_pts = gather_points(m, biz, biy, bix)                      # [B, 3, Mb] (per-comp; reduced inside)
                
                # Loss will internally reduce the 3 comp mask to [B,Mb]
                bc_loss = masked_point_mse(pred_b, bc_vals, bc_mask_pts)


                # physics residual on dense field (cheap + stable)
                # physics residual on dense field (cheap + stable)
                u_full = model(f_s, m, bv_s)
                # Strain comparison in normalized units (consistent)
                eps_pred = compute_strain_tensor(u_full, 1/(N-1))
                eps_true = compute_strain_tensor(u_s, 1/(N-1))
                
                loss_strain= loss_fn(eps_pred, eps_true)
                
                res = navier_cauchy_residual(u_full, f_s, E=1.0, nu=0.3, dx=model.dx)  # [B,3,D,H,W]
                w = interior_mask_from_bc(m).unsqueeze(1).float()                      # [B,1,D,H,W]
                num = (res.pow(2) * w).sum()
                den = (w.sum() * res.size(1)).clamp_min(1.0)                           # multiply by channels
                phys_loss = num / den


                loss = 0.2*data_loss + alpha_bc * bc_loss + alpha_phys * phys_loss + 0.8*loss_strain

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_data += data_loss.item()
                total_bc   += bc_loss.item()
                total_phys += phys_loss.item()

                with torch.no_grad():
                    total_rel += rel_l2_points(pred_pts, u_pts, free_pts)

            # validation & ood on sampled points
            model.eval()
            def eval_loader(loader, M_eval):
                acc = 0.0; cnt = 0
                with torch.no_grad():
                    for f, u, bc_mask, bc_val in loader:
                        f  = f.to(device); u  = u.to(device); m = bc_mask.to(device); bv = bc_val.to(device)
                        f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                        B = f_s.size(0)
                        iz, iy, ix = random_indices(B, M_eval, N, device=f_s.device, pad=0)
                        xyz = make_coords_from_indices(ix, iy, iz, N, space=coord_space)
                        free_pts = gather_mask(interior_mask_from_bc(m).float(), iz, iy, ix)
                        u_pts    = gather_points(u_s, iz, iy, ix)
                        pred_pts = model.forward_points(f_s, m, bv_s, xyz)
                        mse_pts = masked_point_mse(pred_pts, u_pts, free_pts)
                        acc += mse_pts; cnt += 1
                return acc / max(1, cnt)

            avg_train_data = total_data / max(1, len(train_loader))
            avg_train_bc   = total_bc   / max(1, len(train_loader))
            avg_train_phys = total_phys / max(1, len(train_loader))
            avg_rel_l2     = total_rel  / max(1, len(train_loader))
            avg_val_mse    = eval_loader(val_loader, val_M).detach().cpu().numpy()
            avg_ood_mse    = eval_loader(ood_loader, val_M).detach().cpu().numpy()

            if scheduler: scheduler.step(avg_val_mse)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Data:{avg_train_data:.6f}  BC:{avg_train_bc:.6f}  Phys:{avg_train_phys:.6f}  | RelL2≈{avg_rel_l2:.6f}")
            print(f"  Val (sampled) MSE:{avg_val_mse:.6f}  OOD (sampled) MSE:{avg_ood_mse:.6f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scale_S': scaler.S,
            }
            torch.save(state, model_path)
            if avg_val_mse < best_val:
                best_val = avg_val_mse
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ New best model saved")

    # ===== Final evaluation (physical units) =====
    model.eval()
    with torch.no_grad():
        
        
        f_sample, u_sample, bc_mask, bc_val = val_dataset[0]
        f_sample = f_sample.unsqueeze(0).to(device)
        u_sample = u_sample.unsqueeze(0).to(device)
        bc_mask  = bc_mask.unsqueeze(0).to(device)
        bc_val   = bc_val .unsqueeze(0).to(device)

        f_s, u_s, bv_s = scaler.scale_in(f_sample, u_sample, bc_val)
        u_pred_s = model(f_s, bc_mask, bv_s)

        u_pred_phys = scaler.u_to_phys(u_pred_s)
        u_true_phys = u_sample

        interior = ~bc_mask.bool()
        final_mse_phys = nn.MSELoss()(u_pred_phys[interior], u_true_phys[interior]).item()
        max_err_phys   = (u_pred_phys - u_true_phys).abs().max().item()

        print("\nFinal Test (PHYSICAL):")
        print(f"  MSE: {final_mse_phys:.6f}  |  MaxErr: {max_err_phys:.6f}")

# ============================ run ============================
if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    data_dir = f"./data_{mode}"

    for i in range(1, 2):
        model_path = f"model_{mode}/deeponet/checkpoint_deeponet_{i}.pt"
        best_model_path = f"model_{mode}/deeponet/best_deeponet_{i}.pt"

        train_elasticity(
            data_dir=data_dir,
            model_path=model_path,
            best_model_path=best_model_path,
            num_epochs=100,
            batch_size=10,
            log_csv_path="training_loss_log.csv",
            points_per_sample=8192,      # more points → lower variance
            coord_space="[0,1]",         # trunk expects [0,1]
            alpha_phys=0.0,              # physics residual weight
            alpha_bc=0.0                 # BC penalty weight
        )