import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from timeit import default_timer

from benchmark_utils import print_checkpoint_load, print_param_count, print_time, resolve_existing_path
from model_hlcns import HLCNSSolver3D

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
def eval_hlcns_time(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    problem_type: str = 'elasticity',  # 'elasticity' or 'thermal'
    batch_size: int = 8,
    n_cycles: int = 1,
    max_levels: int = 3,
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = resolve_existing_path(model_path)
    best_model_path = resolve_existing_path(best_model_path)

    # Datasets / loaders
    val_dataset   = FieldDataset(data_dir, prefix="validation_")
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    f0, u0, _, _ = val_dataset[0]
    N = f0.shape[-1]
    model = HLCNSSolver3D(
        N=N,
        problem_type=problem_type,
        n_cycles=n_cycles,
        max_levels=max_levels,
        E=E,
        nu=nu,
        kappa=kappa,
    ).to(device)

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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print_param_count(trainable_params, all_params)

    # Scaler S: prefer checkpoint's saved S if available; else estimate with non-shuffled loader (stable)
    scaler = GlobalScaler(quantile=0.99)
    saved_S = None
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            saved_S = ckpt.get('scale_S', None)
    if saved_S is not None:
        scaler.S = float(saved_S)
    else:
        scaler.fit_from_loader(val_loader, max_batches=None, device=device)

    # ---- Validation ----
    model.eval()
    t_val = 0.0
    with torch.no_grad():
        for f, u, m, bv in val_loader:
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            t1 = default_timer()
            pred = model(f_s, m, bv_s)
            t2 = default_timer()
            t_val += t2 - t1
        print_time("Val", t_val)

# ---------------- Script entry ----------------
if __name__ == "__main__":
    mode = 'nonlinear'
    data_dir = f"./data_{mode}_twopeak"
    for n in range(1, 2):
        for l in range(3, 4):
            model_path = f"model_{mode}_twopeak/hlcns/checkpoint_hlcns_n{n}_l{l}_1.pt"
            best_model_path = f"model_{mode}/hlcns/{mode}_n{n}_l{l}_1.pt"
            eval_hlcns_time(
                data_dir, model_path, best_model_path,
                problem_type='elasticity', batch_size=10,
                E=1.0, nu=0.3, n_cycles=n, max_levels=l
            )
