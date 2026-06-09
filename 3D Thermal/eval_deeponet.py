import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from train_deeponet import (
    DeepONet3D,
    PackedTensorDataset3D,
    create_bc_mask_and_value_like_3d,
)

# ---------------------- helpers: upsample & solve to 64x64 ----------------------
def _apply_bc_rhs_like_generator(F_flat: np.ndarray, N: int):
    vertical_bc = np.linspace(1.0, 0.0, N)
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if i == 0:
                F_flat[idx] = 1.0
            elif i == N - 1:
                F_flat[idx] = 0.0
            elif j == 0 or j == N - 1:
                F_flat[idx] = vertical_bc[i]
    return F_flat

def _build_A_linear_with_bc(N: int, dx: float):
    nn = N * N
    diag = -4.0 * np.ones(nn)
    off1 = np.ones(nn - 1)
    off1[np.arange(1, nn) % N == 0] = 0
    offN = np.ones(nn - N)
    A = diags([diag, off1, off1, offN, offN], [0, -1, 1, -N, N], dtype=np.float64) / (dx * dx)
    A = A.tolil()
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                idx = i * N + j
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
    return A.tocsr()

def _trapezoid_total(f: np.ndarray, dx: float) -> float:
    w = np.ones_like(f, dtype=np.float64)
    w[[0, -1], :] *= 0.5
    w[:, [0, -1]] *= 0.5
    w[[0, -1]][:, [0, -1]] *= 0.5
    return float((w * f).sum() * dx * dx)

def _upsample_f_and_solve_u_64(f32: np.ndarray):
    N32 = f32.shape[0]
    dx32 = 1.0 / (N32 - 1)
    N64 = 64
    dx64 = 1.0 / (N64 - 1)

    with torch.no_grad():
        f64 = F.interpolate(
            torch.from_numpy(f32)[None, None].float(),
            size=(N64, N64),
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy().astype(np.float64)
    total32 = _trapezoid_total(f32, dx32)
    total64_before = _trapezoid_total(f64, dx64)
    if total64_before > 0:
        f64 *= (total32 / total64_before)

    A64 = _build_A_linear_with_bc(N64, dx64)
    F64 = f64.flatten().copy()
    F64 = _apply_bc_rhs_like_generator(F64, N64)
    u64 = spsolve(A64, F64).reshape(N64, N64)

    return f64.astype(np.float32), u64.astype(np.float32)

# ---------------------- evaluation ----------------------
@torch.no_grad()
def eval_deeponet(
    tra_pt: str,
    val_pt: str,
    ood_pt: str,
    best_model_path: str,
    ckpt_with_scale: str | None = None,
    batch_size: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tra_ds = PackedTensorDataset3D(tra_pt)  # f/u: [N,1,32,32]
    tra_loader = DataLoader(tra_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_ds = PackedTensorDataset3D(val_pt)  # f/u: [N,1,32,32]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_ds = PackedTensorDataset3D(ood_pt)  # f/u: [N,1,32,32]
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    _, _, D, H, W = val_ds.f.shape

    model = DeepONet3D(N=H, p=64, branch_width=80, trunk_hidden=80, trunk_freqs=12, in_ch=6).to(device)

    state = torch.load(best_model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        print(f"[Loaded checkpoint dict] {best_model_path}")
    else:
        model.load_state_dict(state)
        print(f"[Loaded state_dict] {best_model_path}")
    model.eval()

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for f, u in tra_loader:
        f = f.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
        f_s, u_s, bv_s = f, u, bc_val
        pred = model(f_s, bc_mask, bv_s)
        interior = (bc_mask < 0.5)
        diff2 = (pred - u_s).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u_s.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print(f"Train -- MSE: {mse:.6e} | Rel: {rel:.6e}")

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for f, u in val_loader:
        f = f.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
        f_s, u_s, bv_s = f, u, bc_val
        pred = model(f_s, bc_mask, bv_s)
        interior = (bc_mask < 0.5)
        diff2 = (pred - u_s).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u_s.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print(f"Val   -- MSE: {mse:.6e} | Rel: {rel:.6e}")

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for f, u in ood_loader:
        f = f.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
        f_s, u_s, bv_s = f, u, bc_val
        pred = model(f_s, bc_mask, bv_s)
        interior = (bc_mask < 0.5)
        diff2 = (pred - u_s).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u_s.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print(f"OOD   -- MSE: {mse:.6e} | Rel: {rel:.6e}")

if __name__ == "__main__":
    mode = "decrease"   # "linear" / "increase" / "decrease"
    tra_pt = f"data/training_data_N32_{mode}.pt"
    val_pt = f"data/validation_data_N32_{mode}.pt"
    ood_pt = f"data/dualforce_data_N32_{mode}.pt"
    for i in range(1, 6):
        best_model_path = f"model_{mode}/deeponet/best_deeponet_{mode}_{i}.pt"
        ckpt_with_scale = f"model_{mode}/deeponet/ckpt_deeponet_{mode}_{i}.pt"

        eval_deeponet(
            tra_pt=tra_pt,
            val_pt=val_pt,
            ood_pt=val_pt,
            best_model_path=best_model_path,
            ckpt_with_scale=ckpt_with_scale,
            batch_size=10,
        )
