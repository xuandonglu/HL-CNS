import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from benchmark_utils import (
    benchmark_tag_from_path,
    export_mat_batch,
    print_checkpoint_load,
    print_metrics,
    print_param_count,
    resolve_existing_path,
)
from model_hlcns import HLCNSSolver3D, smooth_hlcns_jacobi
from mfea_kernel_3d import build_vcycle_levels_3d, run_vcycles

# optional (elasticity-only helpers); guarded below
try:
    from plot_tools import compute_strain_tensor, plot_comparison_3d, plot_stress_divergence
    _HAS_PLOT_TOOLS = True
except Exception:
    _HAS_PLOT_TOOLS = False
    def compute_strain_tensor(*args, **kwargs): raise RuntimeError("plot_tools.compute_strain_tensor not available")
    def plot_comparison_3d(*args, **kwargs): print("plot_comparison_3d not available")
    def plot_stress_divergence(*args, **kwargs): print("plot_stress_divergence not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# ------------------------ Data ------------------------
class FieldDataset(Dataset):
    """
    Generic dataset that works for both problems.
    Expects torch files with keys: 'f', 'u', 'bc_mask', 'bc_val'.
    The channel count of 'u' determines problem type if not provided.
    """
    def __init__(self, data_dir, prefix="sample_", max_index=None):
        self.data_dir = data_dir
        self.prefix = prefix
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])
        if max_index is not None:
            self.file_list = [f for f in self.file_list
                              if int(f[len(prefix):len(prefix)+4]) < max_index]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.file_list[idx]))
        return (
            data['f'].float(),
            data['u'].float(),
            data['bc_mask'].float(),
            data['bc_val'].float()
        )

# ------------------------ Scaling ------------------------
class GlobalScaler:
    """Single scalar S applied to u, f, bc_val so K u = f stays consistent."""
    def __init__(self, eps=1e-12, quantile=0.99):
        self.eps = eps
        self.quantile = quantile
        self.S = None

    @torch.no_grad()
    def fit_from_loader(self, loader, max_batches=8, device="cpu"):
        mags = []
        for b, (f, u, _, _) in enumerate(loader):
            u = u.to(device, non_blocking=True)
            m = u.abs().reshape(u.shape[0], -1).amax(dim=1).cpu()
            mags.append(m)
            if max_batches is not None and (b + 1) >= max_batches:
                break
        mags = torch.cat(mags) if len(mags) else torch.tensor([1.0])
        q = torch.quantile(mags, torch.as_tensor(self.quantile))
        self.S = float(max(q.item(), self.eps))
        return self

    def scale_in(self, f, u, bc_val):
        S = self.S
        return f / S, u / S, bc_val / S

    def u_to_phys(self, u_scaled):
        return u_scaled * self.S

# ------------------------ Loss helpers ------------------------
def relative_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()

@torch.no_grad()
def _deriv_kernels(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 1, 3)
    Ky = k.view(1, 1, 1, 3, 1)
    Kz = k.view(1, 1, 3, 1, 1)
    return Kx, Ky, Kz

def grad_3d_scalar(T: torch.Tensor, dx: float):
    """T: [B,1,D,H,W] -> (Tx,Ty,Tz) each [B,1,D,H,W]"""
    Kx, Ky, Kz = _deriv_kernels(T.dtype, T.device)
    Tx = F.conv3d(F.pad(T, (1,1,0,0,0,0), mode='replicate'), Kx) / dx
    Ty = F.conv3d(F.pad(T, (0,0,1,1,0,0), mode='replicate'), Ky) / dx
    Tz = F.conv3d(F.pad(T, (0,0,0,0,1,1), mode='replicate'), Kz) / dx
    return Tx, Ty, Tz

# ------------------------ Trainer ------------------------
def eval_hlcns(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    problem_type: str = 'elasticity',  # 'elasticity' or 'thermal'
    batch_size: int = 8,
    max_eval_samples=None,
    n_cycles: int = 1,
    max_levels: int = 3,
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
    export_mat: bool = False,
    mat_output_dir: str = "mat_output",
    mat_tag: str | None = None,
    use_feanet_jacobi: bool = False,
    feanet_omega: float = 0.6,
    feanet_pre: int = 1,
    feanet_post: int = 1,
    feanet_transfer_mode: str = "full_weighting",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_feanet_jacobi and problem_type != 'elasticity':
        raise ValueError("use_feanet_jacobi=True is implemented here for the 3-D linear elasticity case.")
    if not use_feanet_jacobi:
        model_path = resolve_existing_path(model_path)
        best_model_path = resolve_existing_path(best_model_path)
    mat_tag = mat_tag or ("feanet_jacobi" if use_feanet_jacobi else benchmark_tag_from_path(model_path))

    # Datasets / loaders
    train_dataset = FieldDataset(data_dir, prefix="sample_", max_index=max_eval_samples)
    val_dataset   = FieldDataset(data_dir, prefix="validation_", max_index=max_eval_samples)
    ood_dataset   = FieldDataset(data_dir, prefix="dualforce_", max_index=max_eval_samples) if problem_type=='elasticity' else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader   = DataLoader(ood_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) if ood_dataset else None

    # Model / solver
    f0, u0, _, _ = train_dataset[0]
    N = f0.shape[-1]
    model = None
    feanet_levels = feanet_mask_levels = feanet_val_levels = None
    dx_for_metrics = 1.0 / float(max(N - 1, 1))
    scale_S = None
    if use_feanet_jacobi:
        if "linear" not in os.path.basename(os.path.normpath(data_dir)).lower():
            raise ValueError(f"FEA-Net Jacobi path should use a linear data folder, got: {data_dir}")
        _, _, bc_mask0, bc_val0 = train_dataset[0]
        feanet_levels, feanet_mask_levels, feanet_val_levels = build_vcycle_levels_3d(
            N=N,
            bc_mask=bc_mask0,
            bc_val=bc_val0,
            E=1.0,
            nu=0.3,
            max_levels=max_levels,
            max_dense_dof=3000,
            device=device,
            dtype=torch.float32,
            transfer_mode=feanet_transfer_mode,
        )
        print(f"[FEA-Net Jacobi] Built Hex8 V-cycle levels: {[level.N for level in feanet_levels]}")
        print(f"[FEA-Net Jacobi] omega={feanet_omega}, pre={feanet_pre}, post={feanet_post}, cycles={n_cycles}, transfer={feanet_transfer_mode}")
    else:
        model = HLCNSSolver3D(
            N=N,
            problem_type=problem_type,
            n_cycles=n_cycles,
            max_levels=max_levels,
            E=E,
            nu=nu,
            kappa=kappa,
        ).to(device)
        model._smooth_impl = smooth_hlcns_jacobi
        dx_for_metrics = model.dx
        model.eval()

        # Load weights (checkpoint dict or state_dict)
        if os.path.exists(model_path):
            obj = torch.load(model_path, map_location=device)
            if isinstance(obj, dict) and 'model' in obj:
                model.load_state_dict(obj['model'])
                scale_S = float(obj.get('scale_S', 0.0)) or None
                print_checkpoint_load(model_path, True)
                print(f"Scale S: {scale_S}")
            else:
                model.load_state_dict(obj)
                print_checkpoint_load(model_path, False)
                if best_model_path and os.path.exists(best_model_path):
                    cp = torch.load(best_model_path, map_location='cpu')
                    scale_S = float(cp.get('scale_S', 0.0)) or None
                    print(f"Recovered S from checkpoint: {scale_S}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # total trainable parameters
    if use_feanet_jacobi:
        trainable_params = 0
        all_params = 0
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
    print_param_count(trainable_params, all_params)

    loss_fn = nn.MSELoss()

    # Scaler S: prefer checkpoint's saved S if available; else estimate with non-shuffled loader (stable)
    scaler = GlobalScaler(quantile=0.99)
    if use_feanet_jacobi:
        scaler.S = 1.0
    else:
        saved_S = None
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            if isinstance(ckpt, dict):
                saved_S = ckpt.get('scale_S', None)
        if saved_S is not None:
            scaler.S = float(saved_S)
        else:
            # robust estimation: do NOT use shuffled small subset; use non-shuffled, all batches
            scale_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            scaler.fit_from_loader(scale_loader, max_batches=None, device=device)

    def predict(f, u, m, bv):
        if use_feanet_jacobi:
            return run_vcycles(
                f,
                m,
                bv,
                feanet_levels,
                feanet_mask_levels,
                feanet_val_levels,
                cycles=n_cycles,
                pre=feanet_pre,
                post=feanet_post,
                omega=feanet_omega,
                init="zeros",
                transfer_mode=feanet_transfer_mode,
            )
        f_s, _, bv_s = scaler.scale_in(f, u, bv)
        return scaler.u_to_phys(model(f_s, m, bv_s))

    # ---- Validation ----
    if model is not None:
        model.eval()

    train_mse = 0.0
    train_rel = 0.0
    train_strain_mse = 0.0
    train_strain_rel = 0.0
    with torch.no_grad():
        for f, u, m, bv in train_loader:
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            pred = predict(f, u, m, bv)
            train_mse += loss_fn(pred, u).item()
            train_rel += relative_l2(pred, u)
            if problem_type == 'elasticity':
                eps_pred = compute_strain_tensor(pred, dx_for_metrics)
                eps_true = compute_strain_tensor(u, dx_for_metrics)
                train_strain_mse += loss_fn(eps_pred, eps_true).item()
                train_strain_rel += relative_l2(eps_pred, eps_true)

    avg_train_mse = train_mse / max(1, len(train_loader))
    avg_train_rel = train_rel / max(1, len(train_loader))
    avg_strain_train_mse = train_strain_mse / max(1, len(train_loader))
    avg_strain_train_rel = train_strain_rel / max(1, len(train_loader))

    val_mse = 0.0
    val_rel = 0.0
    val_strain_mse = 0.0
    val_strain_rel = 0.0
    with torch.no_grad():
        for batch_idx, (f, u, m, bv) in enumerate(val_loader):
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            pred = predict(f, u, m, bv)
            export_mat_batch(
                export_mat,
                mat_output_dir,
                "feanet" if use_feanet_jacobi else "hlcns",
                mat_tag,
                "val",
                batch_idx,
                {"f": f, "pred": pred, "exact": u},
            )
            val_mse += loss_fn(pred, u).item()
            val_rel += relative_l2(pred, u)
            if problem_type == 'elasticity':
                eps_pred = compute_strain_tensor(pred, dx_for_metrics)
                eps_true = compute_strain_tensor(u, dx_for_metrics)
                val_strain_mse += loss_fn(eps_pred, eps_true).item()
                val_strain_rel += relative_l2(eps_pred, eps_true)

    avg_val_mse = val_mse / max(1, len(val_loader))
    avg_val_rel = val_rel / max(1, len(val_loader))
    avg_strain_val_mse = val_strain_mse / max(1, len(val_loader))
    avg_strain_val_rel = val_strain_rel / max(1, len(val_loader))

    # OOD (elasticity only)
    avg_ood_mse = None
    avg_ood_rel = None
    if ood_loader is not None:
        ood_mse = 0.0
        ood_rel = 0.0
        ood_strain_mse = 0.0
        ood_strain_rel = 0.0
        with torch.no_grad():
            for f, u, m, bv in ood_loader:
                f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                pred = predict(f, u, m, bv)
                ood_mse += loss_fn(pred, u).item()
                ood_rel += relative_l2(pred, u)
                if problem_type == 'elasticity':
                    eps_pred = compute_strain_tensor(pred, dx_for_metrics)
                    eps_true = compute_strain_tensor(u, dx_for_metrics)
                    ood_strain_mse += loss_fn(eps_pred, eps_true).item()
                    ood_strain_rel += relative_l2(eps_pred, eps_true)

        avg_ood_mse = ood_mse / max(1, len(ood_loader))
        avg_ood_rel = ood_rel / max(1, len(ood_loader))
        avg_strain_ood_mse = ood_strain_mse / max(1, len(ood_loader))
        avg_strain_ood_rel = ood_strain_rel / max(1, len(ood_loader))

    print("Displacement")
    print_metrics("Train", avg_train_mse, avg_train_rel)
    print_metrics("Val", avg_val_mse, avg_val_rel)
    if avg_ood_mse is not None:
        print_metrics("OOD", avg_ood_mse, avg_ood_rel)

    # print("Strain")
    # print(f"Train_MSE: {avg_strain_train_mse:.6e}")
    # print(f"Val_MSE: {avg_strain_val_mse:.6e}")
    # print(f"OOD_MSE: {avg_strain_ood_mse:.6e}")
    # print(f"Train_Rel: {avg_strain_train_rel:.6e}")
    # print(f"Val_Rel: {avg_strain_val_rel:.6e}")
    # print(f"OOD_Rel: {avg_strain_ood_rel:.6e}")

# ---------------- Script entry ----------------
if __name__ == "__main__":
    n = 1
    l = 3
    mode = 'linear'
    for i in range(1, 2):
        data_dir = f"./data_{mode}_twopeak"
        model_path = f"model_{mode}_twopeak/hlcns/checkpoint_hlcns_n{n}_l{l}_{i}.pt"
        best_model_path = f"model_{mode}/hlcns/best_hlcns_n{n}_l{l}_{i}.pt"
        eval_hlcns(
            data_dir, model_path, best_model_path,
            problem_type='elasticity', batch_size=1, max_eval_samples=1,
            E=1.0, nu=0.3, n_cycles=5, max_levels=3,
            export_mat=False,
            use_feanet_jacobi=True,
        )
