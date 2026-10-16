# HL-CNS 3D solver supporting Thermal (scalar) and Elasticity (vector)
# - Single codepath with problem_type: 'thermal' or 'elasticity'
# - Proper base kernels for each problem
# - All other logic (noise kernel, omega field, V-cycle, etc.) shared
#
# Notes:
# * For 'thermal', channels = 1 and the base kernel is the 6-point Laplacian scaled by thermal conductivity kappa.
# * For 'elasticity', channels = 3 and the base kernel is a 3x3x3 Navier–Cauchy stencil with (lambda, mu).
# * noise_net output size adapts automatically: 27 * C * C where C = #channels (1 or 3).
# * correction and omega nets adapt their input/output channel counts.
# * Restriction/prolongation are applied channel-wise to residual, f, bc_mask, bc_val separately (no concat hacks).
# * Feature channel for the noise/correction nets:
#     - Elasticity: von Mises strain (from displacement gradients)
#     - Thermal: |grad T| magnitude
#
# Replace your solver class with this HL-CNS version. Training/eval code should work unchanged
# if you pass problem_type appropriately and supply consistent data shapes and BCs.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -------------------- Small utilities --------------------

def _build_deriv_kernels(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 1, 3)  # d/dx (W)
    Ky = k.view(1, 1, 1, 3, 1)  # d/dy (H)
    Kz = k.view(1, 1, 3, 1, 1)  # d/dz (D)
    return Kx, Ky, Kz

@torch.no_grad()
def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))

# Elasticity feature: von Mises strain from u=[ux,uy,uz]
# Thermal feature: |grad T|

def feature_scalar_or_von_mises(u: torch.Tensor, dx: float, problem_type: str) -> torch.Tensor:
    """Return [B,1,D,H,W] feature.
    - elasticity: von-Mises strain
    - thermal: |grad T|
    """
    if problem_type == 'elasticity':
        assert u.size(1) == 3, "Elasticity expects 3 channels."
        B, _, D, H, W = u.shape
        Kx, Ky, Kz = _build_deriv_kernels(u.dtype, u.device)
        u_pad_x = F.pad(u, (1,1,0,0,0,0), mode='replicate')
        u_pad_y = F.pad(u, (0,0,1,1,0,0), mode='replicate')
        u_pad_z = F.pad(u, (0,0,0,0,1,1), mode='replicate')
        ux = F.conv3d(u_pad_x, Kx.expand(3,1,1,1,3), groups=3) / dx
        uy = F.conv3d(u_pad_y, Ky.expand(3,1,1,3,1), groups=3) / dx
        uz = F.conv3d(u_pad_z, Kz.expand(3,1,3,1,1), groups=3) / dx
        eps_xx = ux[:,0]; eps_yy = uy[:,1]; eps_zz = uz[:,2]
        eps_xy = 0.5*(uy[:,0] + ux[:,1])
        eps_yz = 0.5*(uz[:,1] + uy[:,2])
        eps_zx = 0.5*(ux[:,2] + uz[:,0])
        vm_sq = 0.5*((eps_xx - eps_yy).square() + (eps_yy - eps_zz).square() + (eps_zz - eps_xx).square() \
                     + 6.0*(eps_xy.square() + eps_yz.square() + eps_zx.square()))
        vm = _safe_sqrt(vm_sq)
        return vm.unsqueeze(1)
    else:
        assert u.size(1) == 1, "Thermal expects 1 channel."
        Kx, Ky, Kz = _build_deriv_kernels(u.dtype, u.device)
        u_pad_x = F.pad(u, (1,1,0,0,0,0), mode='replicate')
        u_pad_y = F.pad(u, (0,0,1,1,0,0), mode='replicate')
        u_pad_z = F.pad(u, (0,0,0,0,1,1), mode='replicate')
        Tx = F.conv3d(u_pad_x, Kx, groups=1) / dx
        Ty = F.conv3d(u_pad_y, Ky, groups=1) / dx
        Tz = F.conv3d(u_pad_z, Kz, groups=1) / dx
        gmag = _safe_sqrt(Tx.square() + Ty.square() + Tz.square())
        return gmag

# -------------------- Attention + UNet (unchanged, channel-flexible) --------------------

class AttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, max(1, channels // 4), 1),
            nn.ReLU(),
            nn.Conv3d(max(1, channels // 4), channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * (avg_out + max_out)
        avg_c = torch.mean(x, dim=1, keepdim=True)
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        att = self.spatial(torch.cat([avg_c, max_c], dim=1))
        return x * att

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, 3, padding=1), nn.ReLU(inplace=True)
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
        return self.out_conv(d1)

# -------------------- Voxel-Dependent Convolution (generalized C) --------------------

class VoxelDependentConv3d(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
    def forward(self, field: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        field: [B, C, D, H, W]
        K:     [B, C_out, C_in, 3, 3, 3, D, H, W]
        returns [B, C_out, D, H, W]
        """
        B, C, D, H, W = field.shape
        k = self.kernel_size
        assert K.shape[0] == B and K.shape[3:6] == (3,3,3) and K.shape[-3:] == (D,H,W)
        # Extract neighborhoods
        pad = (1,1,1,1,1,1)
        field_padded = F.pad(field, pad, mode='replicate')
        patches = field_padded.unfold(2,k,1).unfold(3,k,1).unfold(4,k,1)          # [B,C,D,H,W,3,3,3]
        patches = patches.contiguous().view(B, C, D, H, W, -1)                     # [B,C,D,H,W,27]
        patches = patches.permute(0,2,3,4,1,5)                                     # [B,D,H,W,C,27]
        # Reshape kernels
        Cout, Cin = K.shape[1], K.shape[2]
        lap = K.permute(0,6,7,8,1,2,3,4,5).reshape(B, D, H, W, Cout, Cin, 27)      # [B,D,H,W,Cout,Cin,27]
        out = torch.einsum('bdhwci,bdhwoci->bdhwo', patches, lap)                  # [B,D,H,W,Cout]
        return out.permute(0,4,1,2,3).contiguous()

# -------------------- Restriction / Prolongation (per-channel) --------------------

class RestrictFullWeighting3D(nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        w3 = (w1[:,None,None]*w1[None,:,None]*w1[None,None,:])
        k  = (w3/64.0).view(1,1,3,3,3)
        self.register_buffer('kernel', k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = self.kernel.expand(C,1,3,3,3).to(dtype=x.dtype)
        return F.conv3d(x, weight=weight, stride=2, padding=1, groups=C)

class ProlongClassic3D(nn.Module):
    def __init__(self, mode: str = 'trilinear', align_corners: bool = True):
        super().__init__()
        assert mode in ('trilinear','adjoint')
        self.mode = mode
        self.align_corners = align_corners
        if mode == 'adjoint':
            w1 = torch.tensor([1.0,2.0,1.0], dtype=torch.float32)
            w3 = (w1[:,None,None]*w1[None,:,None]*w1[None,None,:])
            k  = (w3/8.0).view(1,1,3,3,3)
            self.register_buffer('kernel_t', k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        if self.mode == 'trilinear':
            return F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=self.align_corners)
        w = self.kernel_t.expand(C,1,3,3,3).to(dtype=x.dtype)
        return F.conv_transpose3d(x, weight=w, stride=2, padding=1, output_padding=1, groups=C)

# -------------------- HL-CNS Solver --------------------

class HLCNSSolver3D(nn.Module):
    def __init__(self,
                 N: int = 32,
                 problem_type: str = 'elasticity',  # 'elasticity' or 'thermal'
                 n_cycles: int = 1,
                 max_levels: int = 3,
                 omega: float = 0.1,
                 # Elasticity params
                 E: float = 1.0, nu: float = 0.3,
                 # Thermal params
                 kappa: float = 1.0,
                 prolong_mode: str = 'trilinear'):
        super().__init__()
        assert problem_type in ('elasticity','thermal')
        self.problem_type = problem_type
        self.N = N
        self.dx = 1.0/(N-1)
        self.n_cycles = n_cycles
        self.max_levels = max_levels
        self.omega = omega
        # Channels
        self.C = 3 if problem_type == 'elasticity' else 1
        # Physical params
        self.E = E; self.nu = nu
        self.mu = E/(2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))
        self.kappa = kappa
        # Base kernel
        self.register_buffer('base_kernel', self._generate_base_kernel())  # [C,C,3,3,3]
        # Nets
        # corr input channels: r_norm(C) + u(C) + rhs(C) + f(C) + feat(1) => 4C+1
        corr_in_ch = 4*self.C + 1
        self.correction_net = UNet3D(in_channels=corr_in_ch, base_channels=32, out_channels=self.C)
        # omega net: same inputs, out C
        self.omega_net = nn.Sequential(
            nn.Conv3d(corr_in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, self.C, 1)
        )
        with torch.no_grad():
            last = self.omega_net[-1]; last.weight.zero_()
            omega_min, omega_max = 0.0, 0.9
            self.omega_min = omega_min; self.omega_max = omega_max
            span = max(omega_max-omega_min, 1e-6)
            t = torch.tensor((omega-omega_min)/span).clamp(1e-4, 1-1e-4)
            bias0 = torch.log(t/(1-t)); last.bias.fill_(bias0)
        # noise net: input 1 (feature), output 27*C*C for voxel kernels
        self.K_elems = 27 * self.C * self.C
        self.noise_net = UNet3D(in_channels=1, base_channels=32, out_channels=self.K_elems)
        # MG ops
        self.restrict = RestrictFullWeighting3D()
        self.prolong  = ProlongClassic3D(mode=prolong_mode)
        # conv
        self.vd_conv = VoxelDependentConv3d()
        # misc
        self.psi_weight = 1.0
        self.corr_clip = None
        self.noise_scale = 1.0

    # ------------- Base kernels -------------
    def _generate_base_kernel(self) -> torch.Tensor:
        C = self.C
        K = torch.zeros((C, C, 3, 3, 3), dtype=torch.float32)
        if self.problem_type == 'thermal':
            # 6-point Laplacian for scalar T, scaled by kappa
            k = self.kappa
            cz = cy = cx = 1
            def add(dz,dy,dx,val):
                K[0,0,cz+dz,cy+dy,cx+dx] += val
            add(0,0,0, -6.0*k)
            add(0,0,-1, k); add(0,0,1, k)
            add(0,-1,0, k); add(0,1,0, k)
            add(-1,0,0, k); add(1,0,0, k)
            return K
        # elasticity: Navier–Cauchy with central differences
        lam, mu = self.lam, self.mu
        s = lam + mu
        c = (1,1,1)
        def add(o,i,dz,dy,dx,val):
            K[o,i,c[0]+dz,c[1]+dy,c[2]+dx] += val
        # mu * Laplacian (6-pt)
        for i in range(3):
            add(i,i,0,0,0,-6.0*mu)
            add(i,i,0,0,-1,mu); add(i,i,0,0,1,mu)
            add(i,i,0,-1,0,mu); add(i,i,0,1,0,mu)
            add(i,i,-1,0,0,mu); add(i,i,1,0,0,mu)
        # (lambda+mu) * second diff along axis i
        add(0,0,0,0,0,-2.0*s); add(0,0,0,0,-1,s); add(0,0,0,0,1,s)
        add(1,1,0,0,0,-2.0*s); add(1,1,0,-1,0,s); add(1,1,0,1,0,s)
        add(2,2,0,0,0,-2.0*s); add(2,2,-1,0,0,s); add(2,2,1,0,0,s)
        # mixed derivatives, signs (+,-;- ,+), weight w = (s)/4
        w = 0.25 * s
        # ∂x∂y u_y -> eq for u_x
        add(0,1, 0,-1,-1,+w); add(0,1, 0,-1,+1,-w); add(0,1, 0,+1,-1,-w); add(0,1, 0,+1,+1,+w)
        # ∂x∂z u_z -> eq for u_x
        add(0,2,-1, 0,-1,+w); add(0,2,-1, 0,+1,-w); add(0,2,+1, 0,-1,-w); add(0,2,+1, 0,+1,+w)
        # ∂y∂x u_x -> eq for u_y
        add(1,0, 0,-1,-1,+w); add(1,0, 0,-1,+1,-w); add(1,0, 0,+1,-1,-w); add(1,0, 0,+1,+1,+w)
        # ∂y∂z u_z -> eq for u_y
        add(1,2,-1,-1, 0,+w); add(1,2,-1,+1, 0,-w); add(1,2,+1,-1, 0,-w); add(1,2,+1,+1, 0,+w)
        # ∂z∂x u_x -> eq for u_z
        add(2,0,-1, 0,-1,+w); add(2,0,-1, 0,+1,-w); add(2,0,+1, 0,-1,-w); add(2,0,+1, 0,+1,+w)
        # ∂z∂y u_y -> eq for u_z
        add(2,1,-1,-1, 0,+w); add(2,1,-1,+1, 0,-w); add(2,1,+1,-1, 0,-w); add(2,1,+1,+1, 0,+w)
        return K

    # ------------- BCs -------------
    @staticmethod
    def _apply_bc(u, bc_mask, bc_val):
        return u * (1 - bc_mask) + bc_val * bc_mask

    # ------------- Kernel assembly + noise -------------
    # def _compute_kernel_field(self, u, f, bc_mask, bc_val, level: int):
    #     dx_l = self.dx * (2**level)
    #     scale = 1.0/(dx_l**2)
    #     base = self.base_kernel * scale                     # [C,C,3,3,3]
    #     feat = feature_scalar_or_von_mises(u, dx_l, self.problem_type)  # [B,1,D,H,W]
    #     raw = torch.tanh(self.noise_net(feat))             # [B, K_elems, D,H,W] in [-1,1]
    #     base_abs_flat = base.abs().view(1, -1, 1, 1, 1)    # [1, 27*C*C, 1,1,1]
    #     noise = self.noise_scale * raw * base_abs_flat     # scale noise to base magnitudes
    #     B, _, D, H, W = noise.shape
    #     # reshape to [B, C_out, C_in, 3,3,3, D,H,W]
    #     K = base.view(1, self.C, self.C, 3,3,3, 1,1,1).expand(B, -1,-1,-1,-1,-1, D,H,W) + \
    #         noise.view(B, self.C, self.C, 3,3,3, D,H,W)
    #     # Diagonal safety: ensure central main diagonals are sufficiently negative
    #     diag_blocks = K[:, :, :, 1,1,1, :, :, :]                # [B,C,C,D,H,W]
    #     main = torch.stack([diag_blocks[:, i, i] for i in range(self.C)], dim=1)  # [B,C,D,H,W]
    #     base_diag = base[:, :, 1,1,1]
    #     base_main = torch.stack([base_diag[i,i] for i in range(self.C)])         # [C]
    #     floor = 0.25 * base_main.abs().view(1, self.C, 1,1,1)
    #     main_proj = -torch.clamp((-main), min=floor)
    #     for i in range(self.C):
    #         K[:, i, i, 1,1,1, :, :, :] = main_proj[:, i]
    #     return K

    def _compute_kernel_field(self, u, f, bc_mask, bc_val, level: int):
        dx_l = self.dx * (2 ** level)
        scale = 1.0 / (dx_l ** 2)
        feat = feature_scalar_or_von_mises(u, dx_l, self.problem_type)
        raw = torch.tanh(self.noise_net(feat))
        noise = self.noise_scale * raw
        B, _, D, H, W = noise.shape
        K = noise.view(B, self.C, self.C, 3, 3, 3, D, H, W)
        diag_blocks = K[:, :, :, 1, 1, 1, :, :, :]
        main = torch.stack([diag_blocks[:, i, i] for i in range(self.C)], dim=1)
        diag_floor = getattr(self, "diag_floor", 6.0)
        floor = (diag_floor * scale) * torch.ones_like(main)
        main_proj = -torch.clamp((-main), min=floor)
        for i in range(self.C):
            K[:, i, i, 1, 1, 1, :, :, :] = main_proj[:, i]
        return K

    # ------------- Smoother -------------
    @staticmethod
    def _solve_block(D: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # D: [B,C,C,D,H,W], b: [B,C,D,H,W]
        B, C, _, Dn, Hn, Wn = D.shape
        A = D.permute(0,3,4,5,1,2).reshape(-1, C, C)
        y = b.permute(0,2,3,4,1).reshape(-1, C, 1)
        x = torch.linalg.solve(A, y)
        return x.reshape(B, Dn, Hn, Wn, C).permute(0,4,1,2,3).contiguous()

    def _smooth(self, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
        smooth_impl = getattr(self, "_smooth_impl", smooth_hlcns_full)
        return smooth_impl(self, u, rhs, f, Kfield, bc_mask, bc_val, level, n_iter)

    # ------------- V-cycle -------------
    def _v_cycle(self, u, rhs, f, bc_mask, bc_val, level: int = 0):
        Kfield = self._compute_kernel_field(u, f, bc_mask, bc_val, level)
        if level >= self.max_levels or min(u.shape[-3:]) <= 3:
            return self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # pre-smooth
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        # residual
        residual = rhs - self.vd_conv(u, Kfield)
        # restrict everything per-channel
        rhs_c   = self.restrict(residual)
        f_c     = self.restrict(f)
        mask_c  = (self.restrict(bc_mask) > 0.5).float()
        val_c   = self.restrict(bc_val) * mask_c
        # coarse solve
        u_c = torch.zeros_like(rhs_c)
        u_c = self._apply_bc(u_c, mask_c, val_c)
        u_c = self._v_cycle(u_c, rhs_c, f_c, mask_c, val_c, level+1)
        # prolong correction and also self-prolong for consistency
        corr = self.prolong(u_c)
        self_prolong = self.prolong(self.restrict(u))
        u = u + corr - self_prolong
        u = self._apply_bc(u, bc_mask, bc_val)
        # post-smooth
        Kfield = self._compute_kernel_field(u, f, bc_mask, bc_val, level)
        u = self._smooth(u, rhs, f, Kfield, bc_mask, bc_val, level)
        return u

    # ------------- Public API -------------
    def forward(self, f, bc_mask, bc_val):
        rhs = f
        # u = torch.randn_like(f, device=f.device)
        u = torch.zeros_like(f)
        u = self._apply_bc(u, bc_mask, bc_val)
        for _ in range(self.n_cycles):
            u = self._v_cycle(u, rhs, f, bc_mask, bc_val)
        return u

    @torch.no_grad()
    def get_noise_kernel(self, u, f, bc_mask, bc_val):
        feat = feature_scalar_or_von_mises(u, self.dx, self.problem_type)
        return self.noise_net(feat)


def _smooth_common_inputs(model, u, rhs, f, Kfield, level: int):
    dx_l = model.dx * (2**level)
    Au = model.vd_conv(u, Kfield)
    res = rhs - Au
    Dblk = Kfield[:, :, :, 1, 1, 1, :, :, :]
    delta = model._solve_block(Dblk, res)
    diag_main = torch.stack([Dblk[:, i, i] for i in range(model.C)], dim=1)
    diag_char = diag_main.abs().clamp_min(1e-6)
    r_norm = res / diag_char
    feat = feature_scalar_or_von_mises(u, dx_l, model.problem_type)
    corr_in = torch.cat([r_norm, u, rhs, f, feat], dim=1)
    return delta, corr_in


def smooth_hlcns_full(model, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
    free = 1.0 - bc_mask
    for _ in range(n_iter):
        delta, corr_in = _smooth_common_inputs(model, u, rhs, f, Kfield, level)
        corr = model.correction_net(corr_in)
        if model.corr_clip is not None:
            corr = torch.clamp(corr, -model.corr_clip, model.corr_clip)
        model._last_correction = corr

        omega_sigma = torch.sigmoid(model.omega_net(corr_in))
        omega_field = model.omega_min + (model.omega_max - model.omega_min) * omega_sigma
        upd = (omega_field * delta + model.psi_weight * corr) * free
        u = model._apply_bc(u + upd, bc_mask, bc_val)
    return u


def smooth_hlcns_no_correction(model, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
    free = 1.0 - bc_mask
    for _ in range(n_iter):
        delta, corr_in = _smooth_common_inputs(model, u, rhs, f, Kfield, level)
        model._last_correction = torch.zeros_like(delta)

        omega_sigma = torch.sigmoid(model.omega_net(corr_in))
        omega_field = model.omega_min + (model.omega_max - model.omega_min) * omega_sigma
        upd = (omega_field * delta) * free
        u = model._apply_bc(u + upd, bc_mask, bc_val)
    return u


def smooth_hlcns_no_relaxation(model, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
    free = 1.0 - bc_mask
    for _ in range(n_iter):
        delta, corr_in = _smooth_common_inputs(model, u, rhs, f, Kfield, level)
        corr = model.correction_net(corr_in)
        if model.corr_clip is not None:
            corr = torch.clamp(corr, -model.corr_clip, model.corr_clip)
        model._last_correction = corr

        omega_field = torch.full_like(delta, model.omega)
        upd = (omega_field * delta + model.psi_weight * corr) * free
        u = model._apply_bc(u + upd, bc_mask, bc_val)
    return u


def smooth_hlcns_jacobi(model, u, rhs, f, Kfield, bc_mask, bc_val, level: int, n_iter: int = 1):
    """Plain damped Jacobi smoother: u <- u + omega * D^{-1}(rhs - K u)."""
    free = 1.0 - bc_mask
    for _ in range(n_iter):
        delta, _ = _smooth_common_inputs(model, u, rhs, f, Kfield, level)
        model._last_correction = torch.zeros_like(delta)
        upd = (float(model.omega) * delta) * free
        u = model._apply_bc(u + upd, bc_mask, bc_val)
    return u


class HLCNSNoCorrection3D(HLCNSSolver3D):
    """Ablation: remove the learned correction network from the smoother."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_net = nn.Identity()
        self._smooth_impl = smooth_hlcns_no_correction


class HLCNSNoRelaxation3D(HLCNSSolver3D):
    """Ablation: remove the learned relaxation network from the smoother."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.omega_net = nn.Identity()
        self._smooth_impl = smooth_hlcns_no_relaxation


class HLCNSJacobi3D(HLCNSSolver3D):
    """Ablation/baseline: replace the learned smoother with a plain damped Jacobi step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_net = nn.Identity()
        self.omega_net = nn.Identity()
        self._smooth_impl = smooth_hlcns_jacobi

# -------------------- Example BC helpers --------------------

@torch.no_grad()
def make_elasticity_bc(N: int, batch: int = 1, device = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """Example: fix z=0 face to 0, prescribe ux=0.05 at z=-1 face."""
    mask = torch.zeros((batch, 3, N, N, N), device=device)
    val  = torch.zeros_like(mask)
    mask[..., 0] = 1.0             # z=0 all components fixed to 0
    mask[:, 0, :, :, -1] = 1.0     # z=-1 face: fix ux
    val[:, 0, :, :, -1] = 0.05
    return mask, val

@torch.no_grad()
def make_thermal_bc(N: int, batch: int = 1, device = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """Example: Dirichlet T=0 on z=0, T=1 on z=-1; other faces free."""
    mask = torch.zeros((batch, 1, N, N, N), device=device)
    val  = torch.zeros_like(mask)
    mask[..., 0]  = 1.0;  val[..., 0]  = 0.0   # z=0
    mask[..., -1] = 1.0;  val[..., -1] = 1.0   # z=-1
    return mask, val
