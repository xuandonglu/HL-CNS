from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import scipy
from matplotlib import pyplot as plt
from scipy.io import savemat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TopoFEMDataset(Dataset):
    def __init__(self, data_dir: str, prefix: str):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = torch.load(os.path.join(self.data_dir, self.file_list[idx]), map_location="cpu")
        f = data["f"].float()
        u = data["u"].float()
        bc_mask = data["bc_mask"].float()
        bc_val = data["bc_val"].float()
        x = data["x"].float()
        E = data["E"]
        nu = data["nu"]
        Lx = data["Lx"]
        Ly = data["Ly"]

        return f, u, x, bc_mask, bc_val, E, nu, Lx, Ly


def enforce_dirichlet_field(u_b2yx: torch.Tensor, mask_b2yx: torch.Tensor, val_b2yx: torch.Tensor) -> torch.Tensor:
    return u_b2yx * mask_b2yx + val_b2yx * (1.0 - mask_b2yx)


def _expand_field(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.dim() == 3:  # (2,ny,nx)
        return t.unsqueeze(0).expand(B, -1, -1, -1)
    if t.dim() == 4 and t.shape[0] == 1 and B > 1:
        return t.expand(B, -1, -1, -1)
    return t


def _field_norm(u_b2yx: torch.Tensor, mask_b2yx: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    B = u_b2yx.shape[0]
    v = (u_b2yx * mask_b2yx).reshape(B, -1)
    return torch.linalg.vector_norm(v, dim=1) + eps


def _expand_rows(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0).expand(B, -1)
    if t.dim() == 2 and t.shape[0] == 1 and B > 1:
        return t.expand(B, -1)
    return t


def _expand_ch(t: torch.Tensor, B: int) -> torch.Tensor:
    # channel mask for Conv2d is (B,2,ny,nx)
    if t.dim() == 3:
        return t.unsqueeze(0).expand(B, -1, -1, -1)
    if t.dim() == 4 and t.shape[0] == 1 and B > 1:
        return t.expand(B, -1, -1, -1)
    return t


# -----------------------------
# 2D element stiffness (Q4) (matches mfea_kernel.py)
# -----------------------------


def q4_ke_plane_elasticity(
    hx: float,
    hy: float,
    E: float,
    nu: float,
    plane_stress: bool = True,
    thickness: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    if plane_stress:
        coef = E / (1.0 - nu**2)
        D = coef * torch.tensor(
            [[1.0, nu, 0.0],
             [nu, 1.0, 0.0],
             [0.0, 0.0, (1.0 - nu) / 2.0]],
            device=dev,
            dtype=dtype,
        )
    else:
        coef = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = coef * torch.tensor(
            [[1.0 - nu, nu, 0.0],
             [nu, 1.0 - nu, 0.0],
             [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]],
            device=dev,
            dtype=dtype,
        )
    gp = 1.0 / math.sqrt(3.0)
    gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    Ke = torch.zeros((8, 8), device=dev, dtype=dtype)
    for xi, eta in gauss:
        dN_dxi = torch.tensor(
            [
                -(1.0 - eta),
                (1.0 - eta),
                (1.0 + eta),
                -(1.0 + eta),
            ],
            device=dev,
            dtype=dtype,
        ) * 0.25
        dN_deta = torch.tensor(
            [
                -(1.0 - xi),
                -(1.0 + xi),
                (1.0 + xi),
                (1.0 - xi),
            ],
            device=dev,
            dtype=dtype,
        ) * 0.25
        detJ = (hx * hy) / 4.0
        invJ = torch.tensor(
            [[2.0 / hx, 0.0], [0.0, 2.0 / hy]], device=dev, dtype=dtype
        )
        dN = torch.stack([dN_dxi, dN_deta], dim=0)
        dN_xy = invJ @ dN
        dN_dx = dN_xy[0]
        dN_dy = dN_xy[1]
        B = torch.zeros((3, 8), device=dev, dtype=dtype)
        for a in range(4):
            B[0, 2 * a] = dN_dx[a]
            B[1, 2 * a + 1] = dN_dy[a]
            B[2, 2 * a] = dN_dy[a]
            B[2, 2 * a + 1] = dN_dx[a]
        Ke = Ke + (B.T @ D @ B) * detJ * thickness
    return Ke


def _precompute_local_pair_maps_2d():

    node_y = np.array([0, 0, 1, 1], dtype=np.int64)
    node_x = np.array([0, 1, 1, 0], dtype=np.int64)

    loc_y = np.empty(8, dtype=np.int64)
    loc_x = np.empty(8, dtype=np.int64)
    loc_c = np.empty(8, dtype=np.int64)

    for k in range(8):
        a = k // 2
        c = k % 2
        loc_y[k] = node_y[a]
        loc_x[k] = node_x[a]
        loc_c[k] = c

    I = np.repeat(np.arange(8, dtype=np.int64), 8)
    J = np.tile(np.arange(8, dtype=np.int64), 8)

    co = loc_c[I]
    ci = loc_c[J]
    ky = (loc_y[J] - loc_y[I] + 1).astype(np.int64)
    kx = (loc_x[J] - loc_x[I] + 1).astype(np.int64)

    ly_i = loc_y[I]
    lx_i = loc_x[I]
    return I, J, co, ci, ky, kx, ly_i, lx_i


def build_fullfield_kernel_tab(
    nelx: int,
    nely: int,
    Ke8: np.ndarray,
    rho_e: np.ndarray,
) -> np.ndarray:

    ny, nx = nely + 1, nelx + 1
    Ktab = np.zeros((2, 2, 3, 3, ny, nx), dtype=np.float64)

    I, J, co, ci, ky, kx, ly_i, lx_i = _precompute_local_pair_maps_2d()
    Ke_pairs = Ke8[I, J].astype(np.float64, copy=False)

    for ey in range(nely):
        for ex in range(nelx):
            s = float(rho_e[ey, ex])
            iy = (ey + ly_i).astype(np.int64)
            ix = (ex + lx_i).astype(np.int64)
            np.add.at(Ktab, (co, ci, ky, kx, iy, ix), s * Ke_pairs)

    return Ktab

def apply_Ku_fullfield_torch(field_b2yx: torch.Tensor, Ktab: torch.Tensor) -> torch.Tensor:
    B, C, ny, nx = field_b2yx.shape
    assert C == 2

    field_padded = F.pad(field_b2yx, (1, 1, 1, 1), mode="replicate")
    patches = (
        field_padded.unfold(2, 3, 1)
        .unfold(3, 3, 1)
        .contiguous()
        .view(B, C, ny, nx, -1)
    )
    patches = patches.permute(0, 2, 3, 1, 4)  # (B,ny,nx,2,9)

    lap = Ktab.permute(4, 5, 0, 1, 2, 3).reshape(ny, nx, 2, 2, 9)  # (ny,nx,o,i,9)
    out = torch.einsum("b y x c i, y x o c i -> b y x o", patches, lap)
    return out.permute(0, 3, 1, 2).contiguous()  # (B,2,ny,nx)


def _block2x2_inv_from_center(Ktab: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    D = Ktab[:, :, 1, 1, :, :]  # (2,2,ny,nx)
    Dm = D.permute(2, 3, 0, 1).contiguous()  # (ny,nx,2,2)
    flat = Dm.view(-1, 2, 2)
    if alpha != 0.0:
        I = torch.eye(2, dtype=flat.dtype, device=flat.device).unsqueeze(0)
        flat = flat + float(alpha) * I
    try:
        inv = torch.linalg.inv(flat)
    except RuntimeError:
        inv = torch.linalg.pinv(flat)
    return inv.view_as(Dm)


def _apply_block_mat(Dinv_yx22: torch.Tensor, r_b2yx: torch.Tensor) -> torch.Tensor:
    r_yx2 = r_b2yx.permute(0, 2, 3, 1).contiguous()  # (B,ny,nx,2)
    du = torch.einsum("y x i j, b y x j -> b y x i", Dinv_yx22, r_yx2)
    return du.permute(0, 3, 1, 2).contiguous()


_MIN_ELEM_STOP = 3
_FW_KERNEL_2D_CACHE = {}
def _aniso_stride_from_nodes(ny: int, nx: int) -> Tuple[int, int]:
    sy = 2 if (ny - 1) > _MIN_ELEM_STOP else 1
    sx = 2 if (nx - 1) > _MIN_ELEM_STOP else 1
    return sy, sx

def _fw_kernel_aniso(device: torch.device, dtype: torch.dtype, sy: int, sx: int) -> torch.Tensor:
    key = (str(device), str(dtype), int(sy), int(sx))
    w = _FW_KERNEL_2D_CACHE.get(key, None)
    if w is None:
        ay = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype) if sy == 2 else torch.tensor([1.0], device=device, dtype=dtype)
        ax = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype) if sx == 2 else torch.tensor([1.0], device=device, dtype=dtype)
        W = ay[:, None] * ax[None, :]
        denom = float(ay.sum() * ax.sum())
        W = W / denom
        w = W.view(1, 1, W.shape[0], W.shape[1]).contiguous()
        _FW_KERNEL_2D_CACHE[key] = w
    return w

def _restrict_avg(x: torch.Tensor) -> torch.Tensor:
    B, C, ny, nx = x.shape
    assert C == 2

    sy, sx = _aniso_stride_from_nodes(ny, nx)
    if sy == 1 and sx == 1:
        return x.contiguous()

    k = _fw_kernel_aniso(x.device, x.dtype, sy, sx)
    pad_y = 1 if sy == 2 else 0
    pad_x = 1 if sx == 2 else 0

    x1 = x.contiguous().view(B * C, 1, ny, nx)
    y = F.conv2d(x1, k, stride=(sy, sx), padding=(pad_y, pad_x))
    y = y * sy * sx
    nyc, nxc = int(y.shape[2]), int(y.shape[3])
    return y.view(B, C, nyc, nxc).contiguous()

def _prolong_bilinear(xc: torch.Tensor, size_fine: Tuple[int, int]) -> torch.Tensor:
    B, C, nyc, nxc = xc.shape
    assert C == 2
    ny, nx = int(size_fine[0]), int(size_fine[1])

    sy, sx = _aniso_stride_from_nodes(ny, nx)
    if sy == 1 and sx == 1:
        y = xc.contiguous().view(B * C, 1, nyc, nxc)
    else:
        k = _fw_kernel_aniso(xc.device, xc.dtype, sy, sx)
        pad_y = 1 if sy == 2 else 0
        pad_x = 1 if sx == 2 else 0
        xc1 = xc.contiguous().view(B * C, 1, nyc, nxc)
        y = F.conv_transpose2d(xc1, k, stride=(sy, sx), padding=(pad_y, pad_x))
        y = y * float(sy * sx)
    py = max(ny - y.shape[2], 0)
    px = max(nx - y.shape[3], 0)
    if py or px:
        y = F.pad(y, (0, px, 0, py))
    y = y[:, :, :ny, :nx]
    return y.view(B, C, ny, nx).contiguous()

def _coarsen_rho_avg(rho: torch.Tensor) -> torch.Tensor:
    nely, nelx = int(rho.shape[0]), int(rho.shape[1])
    sy = 2 if nely > _MIN_ELEM_STOP else 1
    sx = 2 if nelx > _MIN_ELEM_STOP else 1

    if sy == 1 and sx == 1:
        return rho.contiguous()

    x = rho.unsqueeze(0).unsqueeze(0)
    xc = F.avg_pool2d(x, kernel_size=(sy, sx), stride=(sy, sx), ceil_mode=False)
    return xc.squeeze(0).squeeze(0).contiguous()

def _inject_mask_value_to_coarse(mask_f: torch.Tensor, val_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ny, nx = int(mask_f.shape[0]), int(mask_f.shape[1])
    sy, sx = _aniso_stride_from_nodes(ny, nx)
    return (mask_f[::sy, ::sx, :].contiguous(),
        val_f[::sy, ::sx, :].contiguous())

@dataclass
class Level2D:
    Ktab: torch.Tensor
    Dinv: torch.Tensor
    ny: int
    nx: int
    K: Optional[torch.Tensor] = None
    # Optional matrix-free operator override (e.g. Galerkin RAP hierarchy)
    apply_op: Optional[object] = None


def _field_to_vec_node_major(u_b2yx: torch.Tensor) -> torch.Tensor:
    return u_b2yx.permute(0, 2, 3, 1).contiguous().view(u_b2yx.shape[0], -1)


def _vec_to_field_node_major(v_bndof: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
    B = v_bndof.shape[0]
    return v_bndof.view(B, ny, nx, 2).permute(0, 3, 1, 2).contiguous()


def apply_K_level_field(u_b2yx: torch.Tensor, level: Level2D) -> torch.Tensor:
    """Apply the level operator to a vector-field.

    Precedence:
      1) level.K (dense) if available
      2) level.apply_op (matrix-free override), e.g. Galerkin RAP hierarchy
      3) level.Ktab (rediscretized stencil)
    """
    if getattr(level, "K", None) is not None:
        u_vec = _field_to_vec_node_major(u_b2yx)
        Ku_vec = u_vec @ level.K
        return _vec_to_field_node_major(Ku_vec, level.ny, level.nx)
    apply_op = getattr(level, "apply_op", None)
    if apply_op is not None:
        return apply_op(u_b2yx)
    return apply_Ku_fullfield_torch(u_b2yx, level.Ktab)


def _block2x2_inv_from_dense(K: torch.Tensor, ny: int, nx: int, alpha: float = 0.0) -> torch.Tensor:
    nn = int(ny * nx)
    device = K.device
    dtype = K.dtype
    base = (torch.arange(nn, device=device, dtype=torch.int64) * 2).view(nn, 1)
    offs = torch.arange(2, device=device, dtype=torch.int64).view(1, 2)
    idx = base + offs  # (nn,2)
    D = K[idx[:, :, None], idx[:, None, :]]  # (nn,2,2)
    if alpha != 0.0:
        I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        D = D + float(alpha) * I
    try:
        inv = torch.linalg.inv(D)
    except RuntimeError:
        inv = torch.linalg.pinv(D)
    return inv.view(ny, nx, 2, 2).contiguous()


@torch.no_grad()
def _build_galerkin_dense_from_fine(
    fine_level: Level2D,
    ny_c: int,
    nx_c: int,
    block: int = 32,
) -> torch.Tensor:
    ny_f, nx_f = int(fine_level.ny), int(fine_level.nx)
    ndof_c = int(2 * ny_c * nx_c)
    device = fine_level.Dinv.device
    dtype = fine_level.Dinv.dtype

    Kc = torch.zeros((ndof_c, ndof_c), device=device, dtype=dtype)

    for j0 in range(0, ndof_c, int(block)):
        j1 = min(j0 + int(block), ndof_c)
        b = j1 - j0

        E = torch.zeros((b, ndof_c), device=device, dtype=dtype)
        rows = torch.arange(b, device=device)
        cols = torch.arange(j0, j1, device=device)
        E[rows, cols] = 1.0

        e_c = _vec_to_field_node_major(E, ny_c, nx_c)
        e_f = _prolong_bilinear(e_c, size_fine=(ny_f, nx_f))
        y_f = apply_K_level_field(e_f, fine_level)
        y_c = _restrict_avg(y_f)
        Y = _field_to_vec_node_major(y_c)

        Kc[:, j0:j1] = Y.t()

    return Kc.contiguous()


# -----------------------------
#  Galerkin (RAP) attachment
# -----------------------------


def build_dense_from_apply(apply_fn, ny: int, nx: int, device: torch.device, dtype: torch.dtype, block: int = 32) -> torch.Tensor:
    """Materialize a symmetric dense operator matrix from a matrix-free apply.

    Returns K such that: u_vec @ K equals vec(apply_fn(u)) in the same node-major ordering.

    Note: We rely on symmetry (SPD elasticity operator), so building columns is sufficient.
    """
    ndof = int(2 * ny * nx)
    K = torch.zeros((ndof, ndof), device=device, dtype=dtype)
    block = int(block)
    for j0 in range(0, ndof, block):
        j1 = min(j0 + block, ndof)
        b = j1 - j0
        E = torch.zeros((b, ndof), device=device, dtype=dtype)
        rows = torch.arange(b, device=device, dtype=torch.int64)
        cols = torch.arange(j0, j1, device=device, dtype=torch.int64)
        E[rows, cols] = 1.0
        e = _vec_to_field_node_major(E, ny, nx)
        y = apply_fn(e)
        Y = _field_to_vec_node_major(y)
        K[:, j0:j1] = Y.t()
    return K.contiguous()


def attach_galerkin_ops(
    levels: List[Level2D],
    transfers: List[Tuple[object, object]],
    max_dense_dof: int = 3000,
    dense_block: int = 64,
) -> None:
    """Attach Galerkin (RAP) matrix-free operators to an existing rediscretized hierarchy.

    This keeps Ktab (rediscretization) around for features / cheap diagonal preconditioners,
    but makes apply_K_level_field use the true Galerkin operator on coarse levels:
        A_c = R A_f P
    where (P,R) are exactly the transfer ops you pass in.

    Args:
        levels: output of build_levels_fixed (with Ktab/Dinv already populated).
        transfers: list of (prolong_fn, restrict_fn) per edge l (levels[l] -> levels[l+1]).
                   prolong_fn: (B,2,nyc,nxc) -> (B,2,nyf,nxf)
                   restrict_fn: (B,2,nyf,nxf) -> (B,2,nyc,nxc)
        max_dense_dof: if >0, materialize dense K on any level with ndof<=max_dense_dof.

    NOTE: You must call this separately for bilinear vs learned hierarchies, because the
          Galerkin coarse operators depend on P and R.
    """
    # clear existing overrides
    for l in range(1, len(levels)):
        levels[l].K = None
        levels[l].apply_op = None

    # build matrix-free RAP operators
    for l in range(len(levels) - 1):
        lev_f = levels[l]
        lev_c = levels[l + 1]
        prolong_fn, restrict_fn = transfers[l]

        def apply_c(u_c, lev_f=lev_f, prolong_fn=prolong_fn, restrict_fn=restrict_fn):
            ef = prolong_fn(u_c)
            Kef = apply_K_level_field(ef, lev_f)
            return restrict_fn(Kef)

        lev_c.apply_op = apply_c

    # optionally build dense operators for small levels (robust coarsest solve)
    max_dense_dof = int(max_dense_dof)
    if max_dense_dof > 0:
        for l in range(1, len(levels)):
            lev = levels[l]
            ndof = int(2 * lev.ny * lev.nx)
            if ndof <= max_dense_dof:
                apply_fn = lev.apply_op
                if apply_fn is None:
                    apply_fn = lambda u, lev=lev: apply_Ku_fullfield_torch(u, lev.Ktab)
                K = build_dense_from_apply(apply_fn, int(lev.ny), int(lev.nx), device=lev.Dinv.device, dtype=lev.Dinv.dtype, block=int(dense_block))
                lev.K = K
                lev.Dinv = _block2x2_inv_from_dense(K, int(lev.ny), int(lev.nx), alpha=0.0)


class PsiNet(nn.Module):
    def __init__(self, nb_layers: int = 3, channels: int = 2):
        super().__init__()
        self.smoother = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, bias=False) for _ in range(int(nb_layers))]
        )

    def forward(self, x, mask):
        for layer in self.smoother:
            x = layer(x) * mask
        return x


def jacobi_update(
    u: torch.Tensor,
    f: torch.Tensor,
    level: Level2D,
    mask: torch.Tensor,
    value: torch.Tensor,
    omega: float,
    mag,
    psi_net: Optional[PsiNet] = None,
    use_psi: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    B = u.shape[0]
    mask_b = _expand_field(mask, B)
    val_b = _expand_field(value, B)
    u = enforce_dirichlet_field(u, mask_b, val_b)
    Ku = apply_K_level_field(u, level)
    r = (f - Ku) * mask_b
    du = _apply_block_mat(level.Dinv, r) * float(omega)
    if use_psi and psi_net is not None:
        corr = psi_net(r, mask_b) * float(omega)
        u = enforce_dirichlet_field(u + du + corr, mask_b, val_b)
    else:
        u = enforce_dirichlet_field(u + du, mask_b, val_b)
    return u


def smooth(
    u: torch.Tensor,
    f: torch.Tensor,
    level: Level2D,
    mask: torch.Tensor,
    value: torch.Tensor,
    omega: float,
    mag,
    steps: int,
    psi_net: Optional[PsiNet] = None,
    use_psi: bool = False
) -> torch.Tensor:
    B = u.shape[0]
    mask_b = _expand_field(mask, B)
    val_b = _expand_field(value, B)
    for _ in range(int(steps)):
        u = jacobi_update(u, f, level, mask_b, val_b, omega, mag, psi_net=psi_net, use_psi=use_psi)
    return u


def coarse_direct_correction(
    u_b2yx: torch.Tensor,
    r_b2yx: torch.Tensor,
    level: Level2D,
    mask_b2yx: torch.Tensor,
    reg_eps: float = 1e-8,
) -> torch.Tensor:
    K = level.K
    B = u_b2yx.shape[0]
    u_vec = _field_to_vec_node_major(u_b2yx)
    r_vec = _field_to_vec_node_major(r_b2yx)
    m_vec = _field_to_vec_node_major(mask_b2yx)
    du = torch.zeros_like(u_vec)
    for b in range(B):
        free_mask = m_vec[b] > 0.5
        free_idx = torch.nonzero(free_mask, as_tuple=False).squeeze(1)
        if free_idx.numel() == 0:
            continue
        Kff = K.index_select(0, free_idx).index_select(1, free_idx)
        bf = r_vec[b].index_select(0, free_idx)
        tr = torch.trace(Kff)
        reg = reg_eps * (tr / (Kff.shape[0] + 1e-8))
        Kff = Kff + reg * torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
        try:
            sol = torch.linalg.solve(Kff, bf.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            sol = torch.linalg.lstsq(Kff, bf.unsqueeze(-1)).solution.squeeze(-1)
        du[b, free_idx] = sol
    u_vec = u_vec + du
    return _vec_to_field_node_major(u_vec, level.ny, level.nx)


def vcycle(
    u: torch.Tensor,
    f: torch.Tensor,
    levels: List[Level2D],
    mask_levels: List[torch.Tensor],
    val_levels: List[torch.Tensor],
    idx_ch_levels: List[torch.Tensor],
    lvl: int,
    pre: int,
    post: int,
    omega: float,
    mag,
    psi_net: Optional[PsiNet] = None,
    use_psi: bool = False,
) -> torch.Tensor:
    lev = levels[lvl]
    u = smooth(u, f, lev, mask_levels[lvl], val_levels[lvl], omega, mag, pre, psi_net=psi_net, use_psi=use_psi)
    B = u.shape[0]
    mask_b = _expand_field(mask_levels[lvl], B)
    val_b = _expand_field(val_levels[lvl], B)
    Ku = apply_K_level_field(u, lev)
    r = (f - Ku) * mask_b
    if lvl == len(levels) - 1:
        u = coarse_direct_correction(u, r, lev, mask_b)
        return enforce_dirichlet_field(u, mask_b, val_b)
    rc = _restrict_avg(r)
    ec = torch.zeros_like(rc)
    ec = vcycle(ec, rc, levels, mask_levels, val_levels, idx_ch_levels,
        lvl + 1, pre, post, omega, mag, psi_net=psi_net, use_psi=use_psi)
    ef = _prolong_bilinear(ec, size_fine=(lev.ny, lev.nx))
    u = enforce_dirichlet_field(u + ef, mask_b, val_b)
    u = smooth(u, f, lev, mask_levels[lvl], val_levels[lvl], omega, mag, post, psi_net=psi_net, use_psi=use_psi)
    return u


def build_levels_fixed(
    rho0: torch.Tensor,
    bc_mask0: torch.Tensor,
    bc_val0: torch.Tensor,
    Lx: float,
    Ly: float,
    E: float,
    nu: float,
    device: torch.device,
    dtype: torch.dtype,
    max_levels: int = 10,
    max_dense_dof: int = 3000,
) -> Tuple[List[Level2D], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    levels: List[Level2D] = []
    idx_levels: List[torch.Tensor] = []
    val_levels: List[torch.Tensor] = []
    idx_ch_levels: List[torch.Tensor] = []

    rho_l = rho0.to(device=device, dtype=dtype).contiguous()
    mask_l = bc_mask0.to(device=device, dtype=dtype).contiguous()
    val_l = bc_val0.to(device=device, dtype=dtype).contiguous()

    nelx_l = int(rho_l.shape[1])
    nely_l = int(rho_l.shape[0])

    for _ in range(int(max_levels)):
        ny, nx = nely_l + 1, nelx_l + 1
        hx = float(Lx) / float(max(nelx_l, 1))
        hy = float(Ly) / float(max(nely_l, 1))

        Ke8 = q4_ke_plane_elasticity(
            hx=hx,
            hy=hy,
            E=float(E),
            nu=float(nu),
            plane_stress=True,
            thickness=1.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        ).cpu().numpy()

        Ktab_np = build_fullfield_kernel_tab(
            nelx_l,
            nely_l,
            Ke8,
            rho_l.detach().cpu().numpy())
        Ktab = torch.from_numpy(Ktab_np).to(device=device, dtype=dtype)
        Dinv = _block2x2_inv_from_center(Ktab, alpha=0.0)

        levels.append(Level2D(Ktab=Ktab, Dinv=Dinv, ny=ny, nx=nx))

        # store bc fields (channel-first)
        mask_cf = mask_l.permute(2, 0, 1).contiguous()  # (2,ny,nx)
        val_cf = val_l.permute(2, 0, 1).contiguous()    # (2,ny,nx)
        idx_levels.append(mask_cf)
        val_levels.append(val_cf)
        idx_ch_levels.append(mask_cf)

        if max(nelx_l, nely_l) <= _MIN_ELEM_STOP:
            break

        rho_l = _coarsen_rho_avg(rho_l)
        nely_l, nelx_l = int(rho_l.shape[0]), int(rho_l.shape[1])
        mask_l, val_l = _inject_mask_value_to_coarse(mask_l, val_l)

    for l in range(len(levels) - 1):
        lev_f = levels[l]
        lev_c = levels[l + 1]
        ndof_c = int(2 * lev_c.ny * lev_c.nx)
        if int(max_dense_dof) <= 0 or ndof_c > int(max_dense_dof):
            continue
        Kc = _build_galerkin_dense_from_fine(lev_f, lev_c.ny, lev_c.nx, block=32)
        lev_c.K = Kc
        lev_c.Dinv = _block2x2_inv_from_dense(Kc, lev_c.ny, lev_c.nx, alpha=0.0)

    return levels, idx_levels, val_levels, idx_ch_levels


def res_norm(u: torch.Tensor, f: torch.Tensor, level: Level2D, mask: torch.Tensor, eps: float = 1e-30) -> float:
    B = u.shape[0]
    mask_b = _expand_field(mask, B)
    Ku = apply_K_level_field(u, level)
    r = (f - Ku) * mask_b
    fn = _field_norm(f, mask_b, eps=eps)
    rn = _field_norm(r, torch.ones_like(mask_b), eps=0.0)
    return (rn / fn).mean().item()


def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> float:
    if mask is None:
        num = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
        den = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1) + eps
    else:
        B = pred.shape[0]
        mask_b = _expand_field(mask, B)
        num = torch.linalg.vector_norm(((pred - target) * mask_b).reshape(B, -1), dim=1)
        den = torch.linalg.vector_norm((target * mask_b).reshape(B, -1), dim=1) + eps
    return (num / den).mean().item()


def train_fea(
    train_dir: str,
    model_path: str,
    nb_layers: int,
    batch_size: int,
    epochs: int,
    lr: float,
    omega: float,
    seed: int = 0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ds = TopoFEMDataset(train_dir, prefix="train_")

    f0, u0, rho0, bc_mask0, bc_val0, E, nu, Lx, Ly = ds[0]
    levels, idx_levels, val_levels, idx_ch_levels = build_levels_fixed(
        rho0, bc_mask0, bc_val0, Lx, Ly, E, nu, device, dtype
    )
    lev0 = levels[0]
    mask0 = idx_levels[0].to(device=device, dtype=dtype)  # (2,ny,nx)
    val0 = val_levels[0].to(device=device, dtype=dtype)   # (2,ny,nx)

    psi = PsiNet(nb_layers=nb_layers, channels=2).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(psi.parameters(), lr=float(lr))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    best = None
    for ep in range(int(epochs)):
        psi.train()
        running = 0.0
        count = 0
        for f, u_true, _, _, _, _, _, _, _ in loader:
            u_true = u_true / torch.max(u_true)
            f_t = f.to(device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
            u_t = u_true.to(device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
            B = f_t.shape[0]
            mask_b = _expand_field(mask0, B)
            val_b = _expand_field(val0, B)
            u_init = torch.randn_like(u_t)
            u_init = enforce_dirichlet_field(u_init, mask_b, val_b)
            u_pred = jacobi_update(u_init, f_t, lev0, mask_b, val_b, omega, 1, psi, True)
            loss = ((u_pred - u_t) * mask_b).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1

        avg = running / max(count, 1)
        print("epoch", ep, "loss", f"{avg:.3e}")
        if best is None or avg < best:
            best = avg
            d = os.path.dirname(model_path)
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(psi.state_dict(), model_path)


def test_compare_fixed(
    test_dir: str,
    psinet_path: Optional[str],
    penal: float,
    nb_layers: int,
    omega: float,
    pre: int,
    post: int,
    max_vcycles: int,
    seed: int = 0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ds = TopoFEMDataset(test_dir, prefix="test_")

    psi = None
    use_psi = False
    if psinet_path is not None:
        psi = PsiNet(nb_layers=nb_layers, channels=2).to(device=device, dtype=dtype)
        psi.load_state_dict(torch.load(psinet_path, map_location=device))
        psi.eval()
        use_psi = True

    f0, u0, rho0, bc_mask0, bc_val0, E, nu, Lx, Ly = ds[0]
    # rho0 = torch.tensor(scipy.io.loadmat('output_volfrac/pmfea_256_64.mat')['rho'][-1]).to(device=device, dtype=dtype)
    # rho0 = rho0 ** 3
    levels, idx_levels, val_levels, idx_ch_levels = build_levels_fixed(
        rho0, bc_mask0, bc_val0, Lx, Ly, E, nu, device, dtype)
    lev0 = levels[0]
    mask0 = idx_levels[0].to(device=device, dtype=dtype)
    val0 = val_levels[0].to(device=device, dtype=dtype)

    f_field = f0.unsqueeze(0).to(device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
    u_true = u0.unsqueeze(0).to(device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
    # u_true = u_true * 125

    mask_b = _expand_field(mask0, 1)
    val_b = _expand_field(val0, 1)
    u_true = enforce_dirichlet_field(u_true, mask_b, val_b)

    uj = torch.rand_like(f_field)
    up = torch.rand_like(f_field)
    mag = torch.min(1 / rho0).item()

    for k in range(int(max_vcycles) + 1):
        rj = res_norm(uj, f_field, lev0, mask0)
        rp = res_norm(up, f_field, lev0, mask0)
        relj = relative_l2(uj, u_true, mask0)
        relp = relative_l2(up, u_true, mask0)
        print(k, f"jac: res={rj:.3e} relu={relj:.3e} | psi: res={rp:.3e} relu={relp:.3e}")

        uj = vcycle(uj, f_field, levels, idx_levels, val_levels, idx_ch_levels, 0,
            pre, post, omega, mag, psi_net=None, use_psi=False).detach()
        up = vcycle(up, f_field, levels, idx_levels, val_levels, idx_ch_levels, 0,
            pre, post, omega, mag, psi_net=psi, use_psi=use_psi).detach()

        if (k + 1) % 1000 == 0:
            fig, ax = plt.subplots()
            ax.imshow((uj[0, 1]).detach().cpu().numpy(), aspect="equal")
            plt.colorbar(ax.images[0])
            ax.set_title(f"Iteration {k}")
            plt.show()


def test_fea(data_dir: str, model_path: Optional[str], penal: float, nb_layers: int, omega: float, pre: int, post: int, max_vcycles: int):
    test_compare_fixed(data_dir, model_path, penal, nb_layers, omega, pre, post, max_vcycles)


def main():
    data_dir = "./data_256_64"
    psinet_path = "./model/psi_256_64.pth"
    nb_layers = 3
    batch_size = 5
    epochs = 50
    lr = 1e-3
    omega = 2 / 3

    pre = 1
    post = 1
    max_vcycles = 50
    penal = 1.0

    train_fea(data_dir, psinet_path, nb_layers, batch_size, epochs, lr, omega)
    test_fea(data_dir, psinet_path, penal, nb_layers, omega, pre, post, max_vcycles)


if __name__ == "__main__":
    main()
