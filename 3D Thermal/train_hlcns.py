
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import csv

from benchmark_utils import ensure_parent_dir

# =============================================================
# Utilities (3D derivatives, gradients, BC helpers)
# =============================================================

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

@torch.no_grad()
def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))

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

# =============================================================
# Networks (3D UNet for correction/noise; light conv for omega)
# =============================================================

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.out_conv.weight, nonlinearity="linear")
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(self.up2(b))
        d1 = self.dec1(self.up1(d2))
        y = self.out_conv(d1)
        return y

# =============================================================
# Voxel-Dependent Convolution (3D per-voxel 3x3x3 kernels)
# =============================================================

class VoxelDependentConv3d(nn.Module):
    """
    field: [B, C_in, D, H, W]
    K:     [B, C_out, C_in, 3, 3, 3, D, H, W]
    return: [B, C_out, D, H, W]
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, field: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, Cin, D, H, W = field.shape
        k = self.kernel_size
        assert K.shape[0] == B and K.shape[3:6] == (k, k, k) and K.shape[-3:] == (D, H, W)

        # extract 3D patches via tensor.unfold on padded tensor
        pad = (1, 1, 1, 1, 1, 1)  # (w_left, w_right, h_top, h_bottom, d_front, d_back)
        patches = F.pad(field, pad, mode='replicate') \
                    .unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)   # [B,Cin,D,H,W,k,k,k]
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7)              # [B,D,H,W,Cin,k,k,k]
        patches = patches.reshape(B, D, H, W, Cin, k * k * k)          # [B,D,H,W,Cin,27]

        # K flatten to match
        Cout = K.shape[1]
        Kflat = K.permute(0, 6, 7, 8, 1, 2, 3, 4, 5).reshape(B, D, H, W, Cout, Cin, k * k * k)  # [B,D,H,W,Cout,Cin,27]

        out = torch.einsum('bdhwck,bdhwo ck->bdhwo', patches, Kflat)  # [B,D,H,W,Cout]
        return out.permute(0, 4, 1, 2, 3).contiguous()

# =============================================================
# Fixed MG operators (restriction/prolongation)
# =============================================================

class RestrictFullWeighting3D(nn.Module):
    """3D full-weighting: outer([1,2,1])^3 / 64 with stride 2."""
    def __init__(self):
        super().__init__()
        w = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        W = (w[:, None, None] * w[None, :, None] * w[None, None, :])  # 3x3x3
        k = (W / 64.0).view(1, 1, 3, 3, 3)
        self.register_buffer('kernel', k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = self.kernel.expand(C, 1, 3, 3, 3).to(dtype=x.dtype)
        return F.conv3d(x, weight=weight, stride=2, padding=1, groups=C)

class ProlongClassic3D(nn.Module):
    """Trilinear upsample (default)."""
    def __init__(self, align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=self.align_corners)

# =============================================================
# HL-CNS Solver (3D Thermal)
# =============================================================

class HLCNSSolver3D(nn.Module):
    """
    3D thermal multigrid with per-voxel kernels and three learned modules:
    - noise_net: perturb base Laplacian per-voxel 3x3x3
    - correction_net: UNet3D correction field
    - omega_net: voxel-wise relaxation in [omega_min, omega_max]
    """
    def __init__(self,
                 N: int = 32,
                 n_cycles: int = 1,
                 max_levels: int = 3,
                 omega: float = 0.99,
                 kappa: float = 1.0):
        super().__init__()
        self.N = N
        self.dx = 1.0 / (N - 1)
        self.n_cycles = n_cycles
        self.max_levels = max_levels
        self.kappa = kappa
        self.C = 1  # scalar thermal

        # base kernel (7-point Laplacian) scaled per-level by 1/dx^2
        self.register_buffer('base_kernel', self._generate_base_kernel())  # [1,1,3,3,3]

        # nets
        corr_in_ch = 4 * self.C + 1  # r_norm + u + rhs + f + feat = 5
        self.correction_net = UNet3D(in_channels=1, base_channels=16, out_channels=self.C)
        self.omega_net = nn.Sequential(
            nn.Conv3d(corr_in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, self.C, 1)
        )
        with torch.no_grad():
            last = self.omega_net[-1]; last.weight.zero_()
            self.omega_min, self.omega_max = 0.0, 0.99
            span = max(self.omega_max - self.omega_min, 1e-6)
            t = torch.tensor((omega - self.omega_min) / span).clamp(1e-4, 1 - 1e-4)
            bias0 = torch.log(t / (1 - t)); last.bias.fill_(bias0)

        self.K_elems = 27 * self.C * self.C
        self.noise_net = UNet3D(in_channels=1, base_channels=16, out_channels=self.K_elems)

        # MG ops
        self.restrict = RestrictFullWeighting3D()
        self.prolong  = ProlongClassic3D()
        self.vd_conv  = VoxelDependentConv3d()

        # misc
        self.psi_weight = 1.0
        self.corr_clip = None
        self.noise_scale = 1.0
        self._last_correction = None

    # -------- base kernel (7-point Laplacian) --------
    def _generate_base_kernel(self) -> torch.Tensor:
        k = self.kappa
        K = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        K[0, 0, 1, 1, 1] = -6.0 * k
        K[0, 0, 1, 1, 0] = k; K[0, 0, 1, 1, 2] = k
        K[0, 0, 1, 0, 1] = k; K[0, 0, 1, 2, 1] = k
        K[0, 0, 0, 1, 1] = k; K[0, 0, 2, 1, 1] = k
        return K

    # -------- BCs --------
    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    # -------- kernel assembly + noise --------
    def _compute_kernel_field(self, u, f, bc_mask, bc_val, level: int):
        dx_l = self.dx * (2 ** level)
        feat = feature_gradmag_3d(u, dx_l)  # [B,1,D,H,W]
        raw = torch.tanh(self.noise_net(feat))  # [B,27,D,H,W]
        noise = self.noise_scale * raw  # [B,27,D,H,W]
        B, _, D, H, W = noise.shape
        K = noise.view(B, 1, 1, 3, 3, 3, D, H, W)  # [B,1,1,3,3,3,D,H,W]
        main = K[:, 0, 0, 1, 1, 1]  # [B,D,H,W]
        eps = 1000
        main_proj = -torch.clamp(-main, min=eps)
        K[:, 0, 0, 1, 1, 1] = main_proj
        return K

    # -------- smoother --------
    def _smooth(self, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
        free = (1.0 - bc_mask)
        B, _, D, H, W = u.shape
        dx_l = self.dx * (2 ** level)
        for _ in range(n_iter):
            Au  = self.vd_conv(u, Kfield)                    # [B,1,D,H,W]
            res = rhs - Au                                   # [B,1,D,H,W]
            Dv = Kfield[:, 0, 0, 1, 1, 1].unsqueeze(1)       # [B,1,D,H,W]
            Dsafe = Dv.abs().clamp_min(1e-6)
            delta = res / Dsafe
            r_norm = res / Dsafe
            feat   = feature_gradmag_3d(u, dx_l)
            corr_in = torch.cat([r_norm, u, rhs, f, feat], dim=1)  # [B,5,D,H,W]
            if min(D, H, W) >= 4:
                corr = self.correction_net(res)
                if self.corr_clip is not None:
                    corr = torch.clamp(corr, -self.corr_clip, self.corr_clip)
                omega_sigma = torch.sigmoid(self.omega_net(corr_in))
                omega_field = self.omega_min + (self.omega_max - self.omega_min) * omega_sigma
            else:
                corr = torch.zeros_like(u)
                omega0 = (self.omega_min + self.omega_max) * 0.5
                omega_field = torch.full_like(u, fill_value=omega0)
            upd = (omega_field * delta + self.psi_weight * corr) * free
            u = self._apply_bc(u + upd, bc_mask, bc_val)
            self._last_correction = corr.detach()
        return u

    # -------- V-cycle --------
    def _v_cycle(self, u, rhs, f, bc_mask, bc_val, level: int = 0):
        Kfield = self._compute_kernel_field(u, f, bc_mask, bc_val, level)
        if level >= self.max_levels or min(u.shape[-3:]) <= 3:
            return self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # pre-smooth
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # residual
        Au = self.vd_conv(u, Kfield)
        residual = rhs - Au
        # restrict
        rhs_c  = self.restrict(residual)
        f_c    = self.restrict(f)
        mask_c = (self.restrict(bc_mask) > 0.5).float()
        val_c  = self.restrict(bc_val) * mask_c
        # coarse solve
        u_c = torch.zeros_like(rhs_c)
        u_c = self._apply_bc(u_c, mask_c, val_c)
        u_c = self._v_cycle(u_c, rhs_c, f_c, mask_c, val_c, level + 1)
        # prolong + self-prolong consistency
        corr = self.prolong(u_c)
        self_prolong = self.prolong(self.restrict(u))
        u = u + corr - self_prolong
        u = self._apply_bc(u, bc_mask, bc_val)
        # post-smooth with refreshed kernel
        Kfield = self._compute_kernel_field(u, f, bc_mask, bc_val, level)
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        return u

    def forward(self, f, bc_mask, bc_val, u0=None):
        rhs = f
        u = torch.zeros_like(rhs) if u0 is None else u0
        u = self._apply_bc(u, bc_mask, bc_val)
        for _ in range(self.n_cycles):
            u = self._v_cycle(u, rhs, f, bc_mask, bc_val)
        return u

    @torch.no_grad()
    def get_noise_kernel(self, u, f, bc_mask, bc_val):
        feat = feature_gradmag_3d(u, self.dx)
        return self.noise_net(feat)  # [B,27,D,H,W]

# =============================================================
# Dataset for packed .pt (f,u) pairs in 3D
# =============================================================

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

# =============================================================
# Training loop (3D thermal)
# =============================================================

def grad_3d_scalar(T: torch.Tensor, dx: float):
    Kx, Ky, Kz = _build_deriv_kernels_3d(T.dtype, T.device)
    Tx = F.conv3d(F.pad(T, (1, 1, 0, 0, 0, 0), mode='replicate'), Kx) / dx
    Ty = F.conv3d(F.pad(T, (0, 0, 1, 1, 0, 0), mode='replicate'), Ky) / dx
    Tz = F.conv3d(F.pad(T, (0, 0, 0, 0, 1, 1), mode='replicate'), Kz) / dx
    return Tx, Ty, Tz

def train_3d_thermal(
    train_pt: str,
    val_pt: str,
    ood_pt: str,
    model_ckpt: str = 'ckpt_3d.pt',
    best_ckpt: str = 'best_3d.pt',
    num_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    n_cycles: int = 1,
    max_levels: int = 3,
    kappa: float = 1.0,
    omega0: float = 0.99,
    use_grad_loss: bool = True,
    device: str = None,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # datasets
    train_ds = PackedTensorDataset3D(train_pt)
    val_ds   = PackedTensorDataset3D(val_pt)
    ood_ds   = PackedTensorDataset3D(ood_pt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader   = DataLoader(ood_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # grid
    _, _, D, H, W = train_ds.f.shape
    N = D

    # model
    model = HLCNSSolver3D(
        N=N,
        n_cycles=n_cycles,
        max_levels=max_levels,
        omega=omega0,
        kappa=kappa,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    loss_fn = nn.MSELoss()

    best_val = float('inf')

    ensure_parent_dir(model_ckpt)
    ensure_parent_dir(best_ckpt)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ["Epoch", "Train(MSE)", "RelL2", "Reg", "Val(MSE)"]
        writer.writerow(header)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            train_rel = 0.0
            for f, u in train_loader:
                f = f.to(device)
                u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)

                pred = model(f, bc_mask, bc_val)

                interior = (bc_mask < 0.5)
                mse_field = loss_fn(pred[interior], u[interior])
                if use_grad_loss:
                    dx = 1.0 / (N - 1)
                    Txp, Typ, Tzp = grad_3d_scalar(pred, dx)
                    Txt, Tyt, Tzt = grad_3d_scalar(u, dx)
                    mse_grad = (loss_fn(Txp[interior], Txt[interior]) +
                                loss_fn(Typ[interior], Tyt[interior]) +
                                loss_fn(Tzp[interior], Tzt[interior])) / 3.0
                    loss = 0.5 * mse_field + 0.5 * mse_grad
                else:
                    loss = mse_field
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(mse_field.item())
                train_rel += relative_l2(pred[interior], u[interior])

            avg_train = total_loss / max(1, len(train_loader))
            avg_train_rel = train_rel / max(1, len(train_loader))

            # validation
            model.eval()
            total_loss = 0.0
            total_rel = 0
            with torch.no_grad():
                for f, u in val_loader:
                    f = f.to(device)
                    u = u.to(device)
                    bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                    pred = model(f, bc_mask, bc_val)
                    interior = (bc_mask < 0.5)
                    loss = loss_fn(pred[interior], u[interior])
                    total_loss += loss.item()
                    total_rel += relative_l2(pred[interior], u[interior])
            avg_val_loss = total_loss / len(val_loader)
            avg_val_rel = total_rel / len(val_loader)
            if scheduler: scheduler.step(avg_val_loss)

            # OOD
            model.eval()
            total_loss = 0.0
            total_rel = 0
            with torch.no_grad():
                for f, u in ood_loader:
                    f = f.to(device)
                    u = u.to(device)
                    bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
                    pred = model(f, bc_mask, bc_val)
                    interior = (bc_mask < 0.5)
                    loss = loss_fn(pred[interior], u[interior])
                    total_loss += loss.item()
                    total_rel += relative_l2(pred[interior], u[interior])
            avg_ood_loss = total_loss / len(ood_loader)
            avg_ood_rel = total_rel / len(ood_loader)

            # save
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, model_ckpt)
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(model.state_dict(), best_ckpt)

            print(f"Epoch {epoch+1}/{num_epochs} | Train MSE: {avg_train:.6f} | Train Rel: {avg_train_rel:.6f}")
            print(f"Val MSE:{avg_val_loss:.6f} | Val rel:{avg_val_rel:.6f}")
            print(f"OOD MSE:{avg_ood_loss:.6f} | OOD rel:{avg_ood_rel:.6f}")

            # log
            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train, avg_val_loss])

# =============================================================
# Entry point (edit paths if needed)
# =============================================================
if __name__ == "__main__":
    mode = "linear"
    for n in range(1, 2):
        for l in range(1, 2):
            for i in range(2, 6):
                train_pt = f"data/training_data_N32_{mode}.pt"
                val_pt   = f"data/validation_data_N32_{mode}.pt"
                ood_pt   = f"data/validation_data_N32_{mode}.pt"
                log_path = f"model_{mode}/hlcns/{mode}_l{l}_n{n}_{i}.csv"

                train_3d_thermal(
                    train_pt=train_pt,
                    val_pt=val_pt,
                    ood_pt=ood_pt,      model_ckpt=f'model_{mode}/hlcns/ckpt_hlcns_{mode}_l{l}_n{n}_{i}.pt',
                    best_ckpt=f'model_{mode}/hlcns/best_hlcns_{mode}_l{l}_n{n}_{i}.pt',
                    num_epochs=40,
                    batch_size=10,
                    lr=1e-3,
                    n_cycles=n,
                    max_levels=l,
                    kappa=1.0,
                    omega0=0.99,
                    use_grad_loss=True,
                    log_csv_path=log_path
                )
