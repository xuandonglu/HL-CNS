from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enforce_dirichlet(u: torch.Tensor, bc_mask: torch.Tensor, bc_val: torch.Tensor) -> torch.Tensor:
    """bc_mask is 1 on prescribed Dirichlet DOFs and 0 on free DOFs."""
    return u * (1.0 - bc_mask) + bc_val * bc_mask


def free_dof_mask(bc_mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - bc_mask


def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> float:
    if mask is None:
        diff = pred - target
        ref = target
    else:
        diff = (pred - target) * mask
        ref = target * mask
    num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(ref.reshape(ref.shape[0], -1), dim=1) + eps
    return torch.mean(num / den).item()


class GlobalScaler:
    """Single scalar S applied to u, f, and bc_val so K(u/S) = rhs(f/S)."""

    def __init__(self, eps: float = 1e-12, quantile: float = 0.99):
        self.eps = eps
        self.quantile = quantile
        self.S: Optional[float] = None

    @torch.no_grad()
    def fit_from_loader(self, loader, max_batches: Optional[int] = None, device: str | torch.device = "cpu"):
        mags = []
        for batch_idx, (_, u, _, _) in enumerate(loader):
            u = u.to(device, non_blocking=True)
            mags.append(u.abs().reshape(u.shape[0], -1).amax(dim=1).cpu())
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
        vals = torch.cat(mags) if mags else torch.tensor([1.0])
        q = torch.quantile(vals, torch.as_tensor(self.quantile))
        self.S = float(max(q.item(), self.eps))
        return self

    def scale_in(self, f: torch.Tensor, u: torch.Tensor, bc_val: torch.Tensor):
        if self.S is None:
            raise RuntimeError("GlobalScaler.S has not been fitted or loaded.")
        return f / self.S, u / self.S, bc_val / self.S

    def u_to_phys(self, u_scaled: torch.Tensor) -> torch.Tensor:
        if self.S is None:
            raise RuntimeError("GlobalScaler.S has not been fitted or loaded.")
        return u_scaled * self.S


class PsiNet3D(nn.Module):
    """MFEA-Net residual correction network for 3D elasticity fields."""

    def __init__(self, nb_layers: int = 3, channels: int = 3):
        super().__init__()
        self.smoother = nn.ModuleList(
            [nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False) for _ in range(int(nb_layers))]
        )

    def forward(self, x: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.smoother:
            x = layer(x) * free_mask
        return x


def _dN_dxi_hex8(xi: float, eta: float, zeta: float) -> np.ndarray:
    c = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    dN = np.zeros((3, 8), dtype=np.float64)
    for a in range(8):
        cx, cy, cz = c[a]
        dN[0, a] = 0.125 * cx * (1.0 + cy * eta) * (1.0 + cz * zeta)
        dN[1, a] = 0.125 * cy * (1.0 + cx * xi) * (1.0 + cz * zeta)
        dN[2, a] = 0.125 * cz * (1.0 + cx * xi) * (1.0 + cy * eta)
    return dN


def _b_matrix_from_dN_dx(dN_dx: np.ndarray) -> np.ndarray:
    B = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        i = 3 * a
        dN = dN_dx[:, a]
        B[:, i : i + 3] = np.array(
            [
                [dN[0], 0.0, 0.0],
                [0.0, dN[1], 0.0],
                [0.0, 0.0, dN[2]],
                [dN[1], dN[0], 0.0],
                [0.0, dN[2], dN[1]],
                [dN[2], 0.0, dN[0]],
            ],
            dtype=np.float64,
        )
    return B


def _elasticity_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )


def element_stiffness_hex8(dx: float, E: float, nu: float) -> np.ndarray:
    gp = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    C = _elasticity_C(E, nu)
    Ke = np.zeros((24, 24), dtype=np.float64)
    detJ = (dx / 2.0) ** 3
    scale_grad = 2.0 / dx
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = _dN_dxi_hex8(xi, eta, zeta)
                dN_dx = scale_grad * dN_dxi
                B = _b_matrix_from_dN_dx(dN_dx)
                Ke += (B.T @ C @ B) * detJ
    return Ke


def _precompute_local_pair_maps_3d():
    node_i = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
    node_j = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    node_k = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    loc_i = np.empty(24, dtype=np.int64)
    loc_j = np.empty(24, dtype=np.int64)
    loc_k = np.empty(24, dtype=np.int64)
    loc_c = np.empty(24, dtype=np.int64)
    for dof in range(24):
        a = dof // 3
        c = dof % 3
        loc_i[dof] = node_i[a]
        loc_j[dof] = node_j[a]
        loc_k[dof] = node_k[a]
        loc_c[dof] = c

    I = np.repeat(np.arange(24, dtype=np.int64), 24)
    J = np.tile(np.arange(24, dtype=np.int64), 24)
    co = loc_c[I]
    ci = loc_c[J]
    di = (loc_i[J] - loc_i[I] + 1).astype(np.int64)
    dj = (loc_j[J] - loc_j[I] + 1).astype(np.int64)
    dk = (loc_k[J] - loc_k[I] + 1).astype(np.int64)
    li = loc_i[I]
    lj = loc_j[I]
    lk = loc_k[I]
    return I, J, co, ci, di, dj, dk, li, lj, lk


def build_fullfield_kernel_tab_3d(N: int, Ke24: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Assemble a node-centered 3x3x3 stiffness stencil from the Hex8 element matrix."""
    Ktab = np.zeros((3, 3, 3, 3, 3, N, N, N), dtype=np.float64)
    I, J, co, ci, di, dj, dk, li, lj, lk = _precompute_local_pair_maps_3d()
    Ke_pairs = Ke24[I, J].astype(np.float64, copy=False)

    for ei in range(N - 1):
        oi = (ei + li).astype(np.int64)
        for ej in range(N - 1):
            oj = (ej + lj).astype(np.int64)
            for ek in range(N - 1):
                ok = (ek + lk).astype(np.int64)
                np.add.at(Ktab, (co, ci, di, dj, dk, oi, oj, ok), Ke_pairs)
    return Ktab.astype(dtype, copy=False)


def apply_Ku_fullfield_3d(field_b3xyz: torch.Tensor, Ktab: torch.Tensor) -> torch.Tensor:
    B, C, N1, N2, N3 = field_b3xyz.shape
    assert C == 3 and N1 == N2 == N3
    field_padded = F.pad(field_b3xyz, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
    patches = (
        field_padded.unfold(2, 3, 1)
        .unfold(3, 3, 1)
        .unfold(4, 3, 1)
        .contiguous()
        .view(B, C, N1, N2, N3, -1)
    )
    patches = patches.permute(0, 2, 3, 4, 1, 5)  # (B,N,N,N,3,27)
    lap = Ktab.permute(5, 6, 7, 0, 1, 2, 3, 4).reshape(N1, N2, N3, 3, 3, 27)
    out = torch.einsum("bxyzcq,xyzocq->bxyzo", patches, lap)
    return out.permute(0, 4, 1, 2, 3).contiguous()


def _block3x3_inv_from_center(Ktab: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    D = Ktab[:, :, 1, 1, 1, :, :, :]  # (3,3,N,N,N)
    Dm = D.permute(2, 3, 4, 0, 1).contiguous()  # (N,N,N,3,3)
    flat = Dm.view(-1, 3, 3)
    if alpha != 0.0:
        eye = torch.eye(3, dtype=flat.dtype, device=flat.device).unsqueeze(0)
        flat = flat + float(alpha) * eye
    try:
        inv = torch.linalg.inv(flat)
    except RuntimeError:
        inv = torch.linalg.pinv(flat)
    return inv.view_as(Dm)


def _apply_block_mat(Dinv_xyz33: torch.Tensor, r_b3xyz: torch.Tensor) -> torch.Tensor:
    r_xyz3 = r_b3xyz.permute(0, 2, 3, 4, 1).contiguous()
    du = torch.einsum("xyzoc,bxyzc->bxyzo", Dinv_xyz33, r_xyz3)
    return du.permute(0, 4, 1, 2, 3).contiguous()


def _field_to_vec_node_major_3d(u_b3xyz: torch.Tensor) -> torch.Tensor:
    return u_b3xyz.permute(0, 2, 3, 4, 1).contiguous().view(u_b3xyz.shape[0], -1)


def _vec_to_field_node_major_3d(v_bndof: torch.Tensor, N: int) -> torch.Tensor:
    return v_bndof.view(v_bndof.shape[0], N, N, N, 3).permute(0, 4, 1, 2, 3).contiguous()


_FW_KERNEL_3D_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_INTERP_1D_CACHE: dict[tuple[str, torch.dtype, int, int], torch.Tensor] = {}


def _fw_kernel_3d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _FW_KERNEL_3D_CACHE:
        a = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        w = (a[:, None, None] * a[None, :, None] * a[None, None, :]) / float(a.sum() ** 3)
        _FW_KERNEL_3D_CACHE[key] = w.view(1, 1, 3, 3, 3).contiguous()
    return _FW_KERNEL_3D_CACHE[key]


def _coarse_size_from_fine(N: int) -> int:
    N = int(N)
    return max(2, (N + 1) // 2)


def _interp_matrix_1d(n_fine: int, n_coarse: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Linear interpolation matrix P with endpoint-aligned coarse/fine nodes."""
    key = (str(device), dtype, int(n_fine), int(n_coarse))
    cached = _INTERP_1D_CACHE.get(key)
    if cached is not None:
        return cached

    n_fine = int(n_fine)
    n_coarse = int(n_coarse)
    if n_fine == n_coarse:
        P = torch.eye(n_fine, device=device, dtype=dtype)
    else:
        xf = torch.linspace(0.0, float(n_coarse - 1), n_fine, device=device, dtype=dtype)
        left = torch.floor(xf).to(torch.int64).clamp(0, n_coarse - 1)
        right = (left + 1).clamp(0, n_coarse - 1)
        wr = (xf - left.to(dtype)).clamp(0.0, 1.0)
        wl = 1.0 - wr
        P = torch.zeros((n_fine, n_coarse), device=device, dtype=dtype)
        rows = torch.arange(n_fine, device=device, dtype=torch.int64)
        P[rows, left] += wl
        P[rows, right] += wr
    _INTERP_1D_CACHE[key] = P.contiguous()
    return _INTERP_1D_CACHE[key]


def _apply_axis_mat_3d(x: torch.Tensor, mat: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 2:
        return torch.einsum("oa,bcajk->bcojk", mat, x).contiguous()
    if axis == 3:
        return torch.einsum("oa,bciak->bciok", mat, x).contiguous()
    if axis == 4:
        return torch.einsum("oa,bcija->bcijo", mat, x).contiguous()
    raise ValueError(f"Unsupported axis {axis}")


def _restrict_avg3d(x: torch.Tensor) -> torch.Tensor:
    B, C, N1, N2, N3 = x.shape
    assert C == 3 and N1 == N2 == N3
    if N1 <= 2:
        return x.contiguous()
    Nc = _coarse_size_from_fine(N1)
    P = _interp_matrix_1d(N1, Nc, x.device, x.dtype)
    Pt = P.t().contiguous()
    y = _apply_axis_mat_3d(x, Pt, 2)
    y = _apply_axis_mat_3d(y, Pt, 3)
    y = _apply_axis_mat_3d(y, Pt, 4)
    return y.contiguous()


def _restrict_full_weighting3d(x: torch.Tensor) -> torch.Tensor:
    B, C, N1, N2, N3 = x.shape
    assert C == 3 and N1 == N2 == N3
    if N1 <= 2:
        return x.contiguous()
    x1 = x.contiguous().view(B * C, 1, N1, N2, N3)
    y = F.conv3d(x1, _fw_kernel_3d(x.device, x.dtype), stride=2, padding=1) * 8.0
    Nc = int(y.shape[-1])
    return y.view(B, C, Nc, Nc, Nc).contiguous()


def _prolong_trilinear3d(xc: torch.Tensor, size_fine: int) -> torch.Tensor:
    B, C, Nc1, Nc2, Nc3 = xc.shape
    assert C == 3 and Nc1 == Nc2 == Nc3
    N = int(size_fine)
    P = _interp_matrix_1d(N, Nc1, xc.device, xc.dtype)
    y = _apply_axis_mat_3d(xc, P, 2)
    y = _apply_axis_mat_3d(y, P, 3)
    y = _apply_axis_mat_3d(y, P, 4)
    return y.contiguous()


def _prolong_full_weighting3d(xc: torch.Tensor, size_fine: int) -> torch.Tensor:
    B, C, Nc1, Nc2, Nc3 = xc.shape
    assert C == 3 and Nc1 == Nc2 == Nc3
    N = int(size_fine)
    expected = 2 * int(Nc1) - 1
    output_padding = max(0, min(1, N - expected))
    xc1 = xc.contiguous().view(B * C, 1, Nc1, Nc2, Nc3)
    y = F.conv_transpose3d(
        xc1,
        _fw_kernel_3d(xc.device, xc.dtype),
        stride=2,
        padding=1,
        output_padding=output_padding,
    ) * 8.0
    pad_d = max(N - y.shape[2], 0)
    pad_h = max(N - y.shape[3], 0)
    pad_w = max(N - y.shape[4], 0)
    if pad_d or pad_h or pad_w:
        y = F.pad(y, (0, pad_w, 0, pad_h, 0, pad_d))
    y = y[:, :, :N, :N, :N]
    return y.view(B, C, N, N, N).contiguous()


def _restrict_transfer3d(x: torch.Tensor, transfer_mode: str = "endpoint") -> torch.Tensor:
    if transfer_mode == "endpoint":
        return _restrict_avg3d(x)
    if transfer_mode == "full_weighting":
        return _restrict_full_weighting3d(x)
    raise ValueError(f"Unknown transfer_mode={transfer_mode!r}")


def _prolong_transfer3d(xc: torch.Tensor, size_fine: int, transfer_mode: str = "endpoint") -> torch.Tensor:
    if transfer_mode == "endpoint":
        return _prolong_trilinear3d(xc, size_fine=size_fine)
    if transfer_mode == "full_weighting":
        return _prolong_full_weighting3d(xc, size_fine=size_fine)
    raise ValueError(f"Unknown transfer_mode={transfer_mode!r}")


def _inject_bc_to_coarse3d(bc_mask: torch.Tensor, bc_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    N = int(bc_mask.shape[-1])
    Nc = _coarse_size_from_fine(N)
    idx = torch.linspace(0, N - 1, Nc, device=bc_mask.device).round().to(torch.int64)
    mask_c = bc_mask.index_select(-3, idx).index_select(-2, idx).index_select(-1, idx)
    val_c = bc_val.index_select(-3, idx).index_select(-2, idx).index_select(-1, idx)
    return mask_c.contiguous(), val_c.contiguous()


@torch.no_grad()
def build_dense_from_level_3d(level: "LinearElasticityLevel3D", block: int = 64) -> torch.Tensor:
    ndof = int(3 * level.N * level.N * level.N)
    K = torch.empty((ndof, ndof), device=level.Ktab.device, dtype=level.Ktab.dtype)
    eye = torch.eye(ndof, device=level.Ktab.device, dtype=level.Ktab.dtype)
    for start in range(0, ndof, int(block)):
        cols = eye[start : start + int(block)]
        basis = _vec_to_field_node_major_3d(cols, level.N)
        out = _field_to_vec_node_major_3d(level.apply(basis))
        K[:, start : start + int(block)] = out.t()
    return K


def build_dense_from_apply_3d(apply_fn, N: int, device: torch.device, dtype: torch.dtype, block: int = 32) -> torch.Tensor:
    ndof = int(3 * int(N) * int(N) * int(N))
    K = torch.empty((ndof, ndof), device=device, dtype=dtype)
    eye = torch.eye(ndof, device=device, dtype=dtype)
    for start in range(0, ndof, int(block)):
        cols = eye[start : start + int(block)]
        basis = _vec_to_field_node_major_3d(cols, int(N))
        out = _field_to_vec_node_major_3d(apply_fn(basis))
        K[:, start : start + int(block)] = out.t()
    return K


@torch.no_grad()
def build_block_inv_from_apply_3d(
    apply_fn,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    nodes_per_batch: int = 128,
    alpha: float = 0.0,
) -> torch.Tensor:
    N = int(N)
    nn = N ** 3
    D = torch.empty((nn, 3, 3), device=device, dtype=dtype)
    eye3 = torch.eye(3, device=device, dtype=dtype)
    for start in range(0, nn, int(nodes_per_batch)):
        end = min(start + int(nodes_per_batch), nn)
        nodes = torch.arange(start, end, device=device, dtype=torch.int64)
        n_batch = int(nodes.numel())
        basis = torch.zeros((n_batch * 3, 3, N, N, N), device=device, dtype=dtype)
        ii = nodes // (N * N)
        rem = nodes % (N * N)
        jj = rem // N
        kk = rem % N
        for comp in range(3):
            rows = torch.arange(n_batch, device=device, dtype=torch.int64) * 3 + comp
            basis[rows, comp, ii, jj, kk] = 1.0
        out = apply_fn(basis)
        block = out.view(n_batch, 3, 3, N, N, N)
        local = torch.arange(n_batch, device=device, dtype=torch.int64)
        D[nodes] = block[local, :, :, ii, jj, kk].permute(0, 2, 1).contiguous()
    if alpha != 0.0:
        D = D + float(alpha) * eye3.unsqueeze(0)
    try:
        inv = torch.linalg.inv(D)
    except RuntimeError:
        inv = torch.linalg.pinv(D)
    return inv.view(N, N, N, 3, 3).contiguous()


def _block3x3_inv_from_dense(K: torch.Tensor, N: int, alpha: float = 0.0) -> torch.Tensor:
    nn = int(N) ** 3
    device = K.device
    dtype = K.dtype
    base = (torch.arange(nn, device=device, dtype=torch.int64) * 3).view(nn, 1)
    offs = torch.arange(3, device=device, dtype=torch.int64).view(1, 3)
    idx = base + offs
    D = K[idx[:, :, None], idx[:, None, :]]
    if alpha != 0.0:
        D = D + float(alpha) * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    try:
        inv = torch.linalg.inv(D)
    except RuntimeError:
        inv = torch.linalg.pinv(D)
    return inv.view(int(N), int(N), int(N), 3, 3).contiguous()


@dataclass
class LinearElasticityLevel3D:
    N: int
    dx: float
    Ktab: torch.Tensor
    Dinv: torch.Tensor
    K: Optional[torch.Tensor] = None
    apply_op: Optional[object] = None

    @classmethod
    def build(
        cls,
        N: int,
        E: float = 1.0,
        nu: float = 0.3,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        dx = 1.0 / (int(N) - 1)
        Ke = element_stiffness_hex8(dx, E, nu)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        Ktab_np = build_fullfield_kernel_tab_3d(int(N), Ke, dtype=np_dtype)
        Ktab = torch.from_numpy(Ktab_np).to(device=device, dtype=dtype)
        Dinv = _block3x3_inv_from_center(Ktab, alpha=0.0)
        return cls(N=int(N), dx=dx, Ktab=Ktab, Dinv=Dinv)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        if self.K is not None:
            u_vec = _field_to_vec_node_major_3d(u)
            Ku_vec = u_vec @ self.K.t()
            return _vec_to_field_node_major_3d(Ku_vec, self.N)
        if self.apply_op is not None:
            return self.apply_op(u)
        return apply_Ku_fullfield_3d(u, self.Ktab)


def rhs_from_density(f_density: torch.Tensor, dx: float, bc_mask: torch.Tensor, bc_val: torch.Tensor) -> torch.Tensor:
    rhs = f_density * (float(dx) ** 3)
    return rhs * (1.0 - bc_mask) + bc_val * bc_mask


def jacobi_update(
    u: torch.Tensor,
    rhs: torch.Tensor,
    level: LinearElasticityLevel3D,
    bc_mask: torch.Tensor,
    bc_val: torch.Tensor,
    omega: float = 0.6,
    psi_net: Optional[PsiNet3D] = None,
    use_psi: bool = False,
) -> torch.Tensor:
    free = free_dof_mask(bc_mask)
    u = enforce_dirichlet(u, bc_mask, bc_val)
    residual = (rhs - level.apply(u)) * free
    du = _apply_block_mat(level.Dinv, residual) * float(omega)
    if use_psi and psi_net is not None:
        corr = psi_net(residual, free) * float(omega)
        u = u + du + corr
    else:
        u = u + du
    return enforce_dirichlet(u, bc_mask, bc_val)


def run_jacobi_iterations(
    f_density: torch.Tensor,
    bc_mask: torch.Tensor,
    bc_val: torch.Tensor,
    level: LinearElasticityLevel3D,
    iterations: int,
    omega: float = 0.6,
    psi_net: Optional[PsiNet3D] = None,
    use_psi: bool = False,
    init: str = "zeros",
) -> torch.Tensor:
    rhs = rhs_from_density(f_density, level.dx, bc_mask, bc_val)
    if init == "random":
        u = torch.randn_like(f_density)
    elif init == "zeros":
        u = torch.zeros_like(f_density)
    else:
        raise ValueError("init must be 'zeros' or 'random'")
    u = enforce_dirichlet(u, bc_mask, bc_val)
    for _ in range(int(iterations)):
        u = jacobi_update(
            u,
            rhs,
            level,
            bc_mask,
            bc_val,
            omega=omega,
            psi_net=psi_net,
            use_psi=use_psi,
        )
    return u


def build_vcycle_levels_3d(
    N: int,
    bc_mask: torch.Tensor,
    bc_val: torch.Tensor,
    E: float = 1.0,
    nu: float = 0.3,
    max_levels: int = 3,
    max_dense_dof: int = 3000,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    transfer_mode: str = "endpoint",
) -> tuple[list[LinearElasticityLevel3D], list[torch.Tensor], list[torch.Tensor]]:
    if bc_mask.dim() == 5:
        bc_mask = bc_mask[0]
    if bc_val.dim() == 5:
        bc_val = bc_val[0]
    bc_mask_l = bc_mask.to(device=device, dtype=dtype).contiguous()
    bc_val_l = bc_val.to(device=device, dtype=dtype).contiguous()

    levels: list[LinearElasticityLevel3D] = []
    mask_levels: list[torch.Tensor] = []
    val_levels: list[torch.Tensor] = []
    N_l = int(N)
    for level_idx in range(int(max_levels) + 1):
        level = LinearElasticityLevel3D.build(N=N_l, E=E, nu=nu, device=device, dtype=dtype)
        levels.append(level)
        mask_levels.append(bc_mask_l)
        val_levels.append(bc_val_l)
        if level_idx >= int(max_levels) or N_l <= 4:
            break
        bc_mask_l, bc_val_l = _inject_bc_to_coarse3d(bc_mask_l, bc_val_l)
        N_l = int(bc_mask_l.shape[-1])

    for l in range(len(levels) - 1):
        fine_level = levels[l]

        def apply_c(u_c, fine_level=fine_level, transfer_mode=transfer_mode):
            e_f = _prolong_transfer3d(u_c, size_fine=fine_level.N, transfer_mode=transfer_mode)
            Ke_f = fine_level.apply(e_f)
            return _restrict_transfer3d(Ke_f, transfer_mode=transfer_mode)

        levels[l + 1].apply_op = apply_c

    for l in range(1, len(levels)):
        level = levels[l]
        ndof = int(3 * level.N * level.N * level.N)
        if int(max_dense_dof) > 0 and ndof <= int(max_dense_dof):
            apply_fn = level.apply_op if level.apply_op is not None else level.apply
            level.K = build_dense_from_apply_3d(
                apply_fn,
                level.N,
                device=level.Ktab.device,
                dtype=level.Ktab.dtype,
                block=32,
            )
            level.Dinv = _block3x3_inv_from_dense(level.K, level.N, alpha=0.0)
        elif level.apply_op is not None:
            level.Dinv = build_block_inv_from_apply_3d(
                level.apply_op,
                level.N,
                device=level.Ktab.device,
                dtype=level.Ktab.dtype,
                nodes_per_batch=8,
                alpha=0.0,
            )
    return levels, mask_levels, val_levels


def smooth(
    u: torch.Tensor,
    rhs: torch.Tensor,
    level: LinearElasticityLevel3D,
    bc_mask: torch.Tensor,
    bc_val: torch.Tensor,
    omega: float,
    steps: int,
    psi_net: Optional[PsiNet3D] = None,
    use_psi: bool = False,
) -> torch.Tensor:
    for _ in range(int(steps)):
        u = jacobi_update(u, rhs, level, bc_mask, bc_val, omega=omega, psi_net=psi_net, use_psi=use_psi)
    return u


def coarse_direct_correction_3d(
    u: torch.Tensor,
    residual: torch.Tensor,
    level: LinearElasticityLevel3D,
    bc_mask: torch.Tensor,
    reg_eps: float = 1e-8,
) -> torch.Tensor:
    if level.K is None:
        level.K = build_dense_from_level_3d(level, block=64)
    K = level.K
    u_vec = _field_to_vec_node_major_3d(u)
    r_vec = _field_to_vec_node_major_3d(residual)
    free_vec = _field_to_vec_node_major_3d(free_dof_mask(bc_mask).unsqueeze(0)).squeeze(0) > 0.5
    free_idx = torch.nonzero(free_vec, as_tuple=False).squeeze(1)
    if free_idx.numel() == 0:
        return u
    Kff = K.index_select(0, free_idx).index_select(1, free_idx)
    trace = torch.trace(Kff)
    reg = float(reg_eps) * trace / max(1, Kff.shape[0])
    Kff = Kff + reg * torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
    for b in range(u_vec.shape[0]):
        bf = r_vec[b].index_select(0, free_idx)
        try:
            sol = torch.linalg.solve(Kff, bf.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            sol = torch.linalg.lstsq(Kff, bf.unsqueeze(-1)).solution.squeeze(-1)
        u_vec[b, free_idx] = u_vec[b, free_idx] + sol
    return _vec_to_field_node_major_3d(u_vec, level.N)


def vcycle(
    u: torch.Tensor,
    rhs: torch.Tensor,
    levels: list[LinearElasticityLevel3D],
    mask_levels: list[torch.Tensor],
    val_levels: list[torch.Tensor],
    lvl: int,
    pre: int,
    post: int,
    omega: float,
    psi_net: Optional[PsiNet3D] = None,
    use_psi: bool = False,
    transfer_mode: str = "endpoint",
) -> torch.Tensor:
    level = levels[lvl]
    bc_mask = mask_levels[lvl]
    bc_val = val_levels[lvl]
    u = smooth(u, rhs, level, bc_mask, bc_val, omega, pre, psi_net=psi_net, use_psi=use_psi)
    free = free_dof_mask(bc_mask)
    residual = (rhs - level.apply(u)) * free
    if lvl == len(levels) - 1:
        u = coarse_direct_correction_3d(u, residual, level, bc_mask)
        return enforce_dirichlet(u, bc_mask, bc_val)

    rhs_c = _restrict_transfer3d(residual, transfer_mode=transfer_mode)
    u_c = torch.zeros_like(rhs_c)
    u_c = enforce_dirichlet(u_c, mask_levels[lvl + 1], val_levels[lvl + 1])
    e_c = vcycle(
        u_c,
        rhs_c,
        levels,
        mask_levels,
        val_levels,
        lvl + 1,
        pre,
        post,
        omega,
        psi_net=psi_net,
        use_psi=use_psi,
        transfer_mode=transfer_mode,
    )
    e_f = _prolong_transfer3d(e_c, size_fine=level.N, transfer_mode=transfer_mode)
    u = enforce_dirichlet(u + e_f, bc_mask, bc_val)
    u = smooth(u, rhs, level, bc_mask, bc_val, omega, post, psi_net=psi_net, use_psi=use_psi)
    return u


def run_vcycles(
    f_density: torch.Tensor,
    bc_mask: torch.Tensor,
    bc_val: torch.Tensor,
    levels: list[LinearElasticityLevel3D],
    mask_levels: list[torch.Tensor],
    val_levels: list[torch.Tensor],
    cycles: int,
    pre: int = 1,
    post: int = 1,
    omega: float = 0.6,
    psi_net: Optional[PsiNet3D] = None,
    use_psi: bool = False,
    init: str = "zeros",
    transfer_mode: str = "endpoint",
) -> torch.Tensor:
    rhs = rhs_from_density(f_density, levels[0].dx, bc_mask, bc_val)
    if init == "random":
        u = torch.randn_like(f_density)
    elif init == "zeros":
        u = torch.zeros_like(f_density)
    else:
        raise ValueError("init must be 'zeros' or 'random'")
    u = enforce_dirichlet(u, bc_mask, bc_val)
    for _ in range(int(cycles)):
        u = vcycle(
            u,
            rhs,
            levels,
            mask_levels,
            val_levels,
            0,
            pre,
            post,
            omega,
            psi_net=psi_net,
            use_psi=use_psi,
            transfer_mode=transfer_mode,
        )
    return enforce_dirichlet(u, bc_mask, bc_val)
