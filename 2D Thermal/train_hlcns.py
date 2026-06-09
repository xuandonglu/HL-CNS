import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import csv

from benchmark_utils import ensure_parent_dir

# =============================================================
# Utilities (2D derivatives, gradients, BC helpers)
# =============================================================
def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

def _build_deriv_kernels_2d(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 3)  # d/dx
    Ky = k.view(1, 1, 3, 1)  # d/dy
    return Kx, Ky

@torch.no_grad()
def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))


def feature_gradmag_2d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Thermal feature for nets: |grad T|, returns [B,1,H,W]."""
    assert u.size(1) == 1, "Thermal expects 1 channel."
    Kx, Ky = _build_deriv_kernels_2d(u.dtype, u.device)
    u_pad_x = F.pad(u, (1, 1, 0, 0), mode='replicate')
    u_pad_y = F.pad(u, (0, 0, 1, 1), mode='replicate')
    Tx = F.conv2d(u_pad_x, Kx) / dx
    Ty = F.conv2d(u_pad_y, Ky) / dx
    gmag = _safe_sqrt(Tx.square() + Ty.square())
    return gmag

@torch.no_grad()
def create_bc_mask_and_value_like_2d(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linearly varying Dirichlet BCs used in your 2D scripts.
       - top = 1.0, bottom = 0.0
       - left/right = vertical gradient from 1 -> 0
       Args: T: [B,1,H,W]
       Returns: (bc_mask, bc_val) both [B,1,H,W]
    """
    B, C, H, W = T.shape
    assert C == 1
    bc_mask = torch.zeros_like(T)
    bc_val  = torch.zeros_like(T)
    vgrad = torch.linspace(1.0, 0.0, H, device=T.device).view(H, 1)
    # top & bottom
    bc_mask[..., 0, :] = 1.0; bc_val[..., 0, :] = 1.0
    bc_mask[..., -1, :] = 1.0; bc_val[..., -1, :] = 0.0
    # left & right with vertical gradient
    bc_mask[..., :, 0]  = 1.0; bc_val[..., :, 0]  = vgrad.T
    bc_mask[..., :, -1] = 1.0; bc_val[..., :, -1] = vgrad.T
    return bc_mask, bc_val

# =============================================================
# Networks (2D UNet for correction/noise; light MLP conv for omega)
# =============================================================

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
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
# Voxel-Dependent Convolution (2D per-voxel 3x3 kernels)
# =============================================================

class VoxelDependentConv2d(nn.Module):
    """
    field: [B, C_in, H, W]
    K:     [B, C_out, C_in, 3, 3, H, W]
    return: [B, C_out, H, W]
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, field: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = field.shape
        k = self.kernel_size
        assert K.shape[0] == B and K.shape[3:5] == (k, k) and K.shape[-2:] == (H, W)
        patches = F.unfold(field, kernel_size=k, padding=k // 2)               # [B, Cin*k*k, H*W]
        patches = patches.view(B, Cin, k * k, H, W).permute(0, 3, 4, 1, 2)     # [B,H,W,Cin,9]
        Cout = K.shape[1]
        Kflat = K.permute(0, 5, 6, 1, 2, 3, 4).reshape(B, H, W, Cout, Cin, k * k)  # [B,H,W,Cout,Cin,9]
        out = torch.einsum('bhwci,bhwoci->bhwo', patches, Kflat)               # [B,H,W,Cout]
        return out.permute(0, 3, 1, 2).contiguous()

# =============================================================
# Fixed MG operators (restriction/prolongation)
# =============================================================

class RestrictFullWeighting2D(nn.Module):
    """Classic full-weighting: (1 2 1; 2 4 2; 1 2 1)/16 with stride 2."""
    def __init__(self):
        super().__init__()
        w1 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        w2 = (w1[:, None] * w1[None, :])  # 3x3
        k = (w2 / 16.0).view(1, 1, 3, 3)
        self.register_buffer('kernel', k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = self.kernel.expand(C, 1, 3, 3).to(dtype=x.dtype)
        return F.conv2d(x, weight=weight, stride=2, padding=1, groups=C)

class ProlongClassic2D(nn.Module):
    """Bilinear upsample (default) or adjoint of full-weighting via transposed conv."""
    def __init__(self, mode: str = 'bilinear', align_corners: bool = True):
        super().__init__()
        assert mode in ('bilinear', 'adjoint')
        self.mode = mode
        self.align_corners = align_corners
        if mode == 'adjoint':
            w1 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
            w2 = (w1[:, None] * w1[None, :])
            k = (w2 / 4.0).view(1, 1, 3, 3)  # adjoint scaling
            self.register_buffer('kernel_t', k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        if self.mode == 'bilinear':
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        w = self.kernel_t.expand(C, 1, 3, 3).to(dtype=x.dtype)
        return F.conv_transpose2d(x, weight=w, stride=2, padding=1, output_padding=1, groups=C)

# =============================================================
# HL-CNS Solver (2D Thermal)
# =============================================================

class HLCNSSolver2D(nn.Module):
    """
    2D thermal multigrid with per-voxel kernels and three learned modules:
    - noise_net: perturb base Laplacian per-voxel 3x3
    - correction_net: UNet2D correction field
    - omega_net: voxel-wise relaxation weight in [omega_min, omega_max]
    """
    def __init__(self,
                 N: int = 32,
                 n_cycles: int = 1,
                 max_levels: int = 3,
                 omega: float = 0.99,
                 kappa: float = 1.0,
                 prolong_mode: str = 'bilinear'):
        super().__init__()
        self.N = N
        self.dx = 1.0 / (N - 1)
        self.n_cycles = n_cycles
        self.max_levels = max_levels
        self.kappa = kappa
        self.C = 1  # scalar thermal

        # base kernel (5-point Laplacian) scaled per-level by 1/dx^2
        self.register_buffer('base_kernel', self._generate_base_kernel())  # [1,1,3,3]

        # nets
        corr_in_ch = 4 * self.C + 1  # r_norm(C) + u(C) + rhs(C) + f(C) + feat(1) = 5
        self.correction_net = UNet2D(in_channels=1, base_channels=32, out_channels=self.C)
        self.omega_net = nn.Sequential(
            nn.Conv2d(corr_in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.C, 1)
        )
        with torch.no_grad():
            last = self.omega_net[-1]; last.weight.zero_()
            self.omega_min, self.omega_max = 0.0, 0.99
            span = max(self.omega_max - self.omega_min, 1e-6)
            t = torch.tensor((omega - self.omega_min) / span).clamp(1e-4, 1 - 1e-4)
            bias0 = torch.log(t / (1 - t)); last.bias.fill_(bias0)

        self.K_elems = 9 * self.C * self.C
        self.noise_net = UNet2D(in_channels=1, base_channels=32, out_channels=self.K_elems)

        # MG ops
        self.restrict = RestrictFullWeighting2D()
        self.prolong  = ProlongClassic2D(mode=prolong_mode)
        self.vd_conv  = VoxelDependentConv2d()

        # misc
        self.psi_weight = 1.0
        self.corr_clip = None
        self.noise_scale = 1.0
        self._last_correction = None  # for optional regularization/logging

    # -------- base kernel (5-point Laplacian) --------
    def _generate_base_kernel(self) -> torch.Tensor:
        k = self.kappa
        K = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        K[0, 0, 1, 1] = -4.0 * k
        K[0, 0, 1, 0] = k; K[0, 0, 1, 2] = k
        K[0, 0, 0, 1] = k; K[0, 0, 2, 1] = k
        return K

    # -------- BCs --------
    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    # -------- kernel assembly + noise --------
    def _compute_kernel_field(self, u, f, bc_mask, bc_val, level: int):
        dx_l = self.dx * (2 ** level)
        feat = feature_gradmag_2d(u, dx_l)                 # [B,1,H,W]
        raw  = torch.tanh(self.noise_net(feat))            # [B,9,H,W]
        noise = self.noise_scale * raw         # [B,9,H,W]
        B, _, H, W = noise.shape
        K = noise.view(B, 1, 1, 3, 3, H, W)
        main = K[:, 0, 0, 1, 1]                           # [B,H,W]
        floor = 500
        main_proj = -torch.clamp((-main), min=floor)
        K[:, 0, 0, 1, 1] = main_proj
        return K

    # -------- smoother --------
    def _smooth(self, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
        free = (1.0 - bc_mask)
        B, _, H, W = u.shape
        dx_l = self.dx * (2 ** level)
        for _ in range(n_iter):
            Au  = self.vd_conv(u, Kfield)                  # [B,1,H,W]
            res = rhs - Au                                 # [B,1,H,W]
            # Jacobi-like update using voxel diagonal
            D = Kfield[:, 0, 0, 1, 1].unsqueeze(1)         # [B,1,H,W]
            Dsafe = D.abs().clamp_min(1e-6)
            delta = res / Dsafe
            # nets input
            r_norm = res / Dsafe
            feat   = feature_gradmag_2d(u, dx_l)
            corr_in = torch.cat([r_norm, u, rhs, f, feat], dim=1)  # [B,5,H,W]
            # small grids: skip UNet to avoid pooling to 0x0
            if min(H, W) >= 4:
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
        if level >= self.max_levels or min(u.shape[-2:]) <= 3:
            return self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # pre-smooth (optional)
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # residual
        Au = self.vd_conv(u, Kfield)
        residual = rhs - Au
        # restrict channel-wise
        rhs_c  = self.restrict(residual)
        f_c    = self.restrict(f)
        mask_c = (self.restrict(bc_mask) > 0.5).float()
        val_c  = self.restrict(bc_val) * mask_c
        # coarse solve
        u_c = torch.zeros_like(rhs_c)
        u_c = self._apply_bc(u_c, mask_c, val_c)
        u_c = self._v_cycle(u_c, rhs_c, f_c, mask_c, val_c, level + 1)
        # prolong correction and self-prolong consistency
        corr = self.prolong(u_c)
        self_prolong = self.prolong(self.restrict(u))
        u = u + corr - self_prolong
        u = self._apply_bc(u, bc_mask, bc_val)
        # post-smooth with refreshed kernel
        Kfield = self._compute_kernel_field(u, f, bc_mask, bc_val, level)
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        return u

    # -------- public API --------
    def forward(self, f, bc_mask, bc_val, u0=None):
        rhs = f
        u = torch.zeros_like(rhs) if u0 is None else u0
        u = self._apply_bc(u, bc_mask, bc_val)
        for _ in range(self.n_cycles):
            u = self._v_cycle(u, rhs, f, bc_mask, bc_val)
        return u

    @torch.no_grad()
    def get_noise_kernel(self, u, f, bc_mask, bc_val):
        feat = feature_gradmag_2d(u, self.dx)
        return self.noise_net(feat)  # [B,9,H,W]

# =============================================================
# Dataset for packed .pt (f,u) pairs
# =============================================================

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

# =============================================================
# Training loop (combined, 2D thermal)
# =============================================================

def grad_2d_scalar(T: torch.Tensor, dx: float):
    Kx, Ky = _build_deriv_kernels_2d(T.dtype, T.device)
    Tx = F.conv2d(F.pad(T, (1, 1, 0, 0), mode='replicate'), Kx) / dx
    Ty = F.conv2d(F.pad(T, (0, 0, 1, 1), mode='replicate'), Ky) / dx
    return Tx, Ty


def train_2d_hlcns(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = 'ckpt_2d.pt',
    best_ckpt: str = 'best_2d.pt',
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_cycles: int = 1,
    max_levels: int = 4,
    kappa: float = 1.0,
    omega0: float = 0.99,
    use_grad_loss: bool = True,
    device: str = None,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # datasets
    train_ds = PackedTensorDataset2D(train_pt)
    val_ds = PackedTensorDataset2D(val_pt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # grid
    _, _, H, W = train_ds.f.shape

    # model
    model = HLCNSSolver2D(
        N=H,
        n_cycles=n_cycles,
        max_levels=max_levels,
        omega=omega0,
        kappa=kappa,
        prolong_mode='bilinear',  # or 'adjoint'
    ).to(device)

    # total trainable parameters
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
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
    
                pred = model(f, bc_mask, bc_val)
    
                # interior-only loss
                interior = (bc_mask < 0.5)
                mse_field = loss_fn(pred[interior], u[interior])
                if use_grad_loss:
                    dx = 1.0 / (H - 1)
                    Txp, Typ = grad_2d_scalar(pred, dx)
                    Txt, Tyt = grad_2d_scalar(u, dx)
                    mse_grad = 0.5 * (loss_fn(Txp[interior], Txt[interior]) + loss_fn(Typ[interior], Tyt[interior]))
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
                    bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                    pred = model(f, bc_mask, bc_val)
                    interior = (bc_mask < 0.5)
                    loss = loss_fn(pred[interior], u[interior])
                    total_loss += loss.item()
                    total_rel += relative_l2(pred[interior], u[interior])
    
            avg_val = total_loss / len(val_loader)
            avg_rel = total_rel / len(val_loader)
            if scheduler: scheduler.step(avg_val)
    
            # save
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, model_ckpt)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), best_ckpt)
    
            print(f"Epoch {epoch+1}/{num_epochs} | Train MSE: {avg_train:.6f} | Train Rel: {avg_train_rel:.6f}")
            print(f"Val MSE:{avg_val:.6f} | Val rel:{avg_rel:.6f}")

            # log
            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train, avg_val])


# =============================================================
# Entry point (edit paths if needed)
# =============================================================
if __name__ == "__main__":
    mode = "linear"
    for n in range(1, 2):
        for l in range(3, 4):
            for i in range(6, 7):
                train_pt = f"data/training_data_N32_{mode}.pt"
                val_pt = f"data/validation_data_N32_{mode}.pt"
                log_path = f"model_{mode}/hlcns/{mode}_l{l}_n{n}_{i}.csv"

                train_2d_hlcns(
                    train_pt=train_pt,
                    val_pt=val_pt,
                    model_ckpt=f'model_{mode}/hlcns/ckpt_2d_{mode}_l{l}_n{n}_{i}.pt',
                    best_ckpt=f'model_{mode}/hlcns/best_2d_{mode}_l{l}_n{n}_{i}.pt',
                    num_epochs=100,
                    batch_size=10,
                    lr=1e-3,
                    n_cycles=n,
                    max_levels=l,
                    kappa=1.0,
                    omega0=0.99,
                    use_grad_loss=True,
                    log_csv_path=log_path
                )
