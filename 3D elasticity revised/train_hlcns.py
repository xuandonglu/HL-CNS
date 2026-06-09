import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
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
def train_hlcns(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    problem_type: str = 'elasticity',  # 'elasticity' or 'thermal'
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_cycles: int = 1,
    max_levels: int = 3,
    # physics params
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
    log_csv_path: str = 'training_log.csv',
):
    assert problem_type in ('elasticity','thermal')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets / loaders
    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset   = FieldDataset(data_dir, prefix="validation_")
    ood_dataset   = FieldDataset(data_dir, prefix="dualforce_") if problem_type=='elasticity' else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader   = DataLoader(ood_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) if ood_dataset else None

    # Model
    f0, u0, _, _ = train_dataset[0]
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

    # total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    loss_fn = nn.MSELoss()

    # Global scale S
    scaler = GlobalScaler(quantile=0.99)
    start_epoch = 0
    best_val = float('inf')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        scaler.S = float(checkpoint.get('scale_S', 0.0)) or None
        if scaler.S is None:
            scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"Resuming from epoch {start_epoch}; S = {scaler.S:.3e}")
    else:
        print("Starting fresh.")
        scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"[Scaler] S = {scaler.S:.3e} (u,f,bc_val divided by S)")

    # Reg weights (tune as needed)
    l1_lambda = 0.0
    l2_lambda = 0.0
    l1_lambda_corr = 0.0
    l2_lambda_corr = 0.0

    ensure_parent_dir(model_path)
    ensure_parent_dir(best_model_path)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ["Epoch", "Train(MSE)", "RelL2", "Reg", "Val(MSE)"] + (["OOD(MSE)"] if ood_loader else [])
        writer.writerow(header)

        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_mse = total_reg = total_rel = 0.0

            for f, u, m, bv in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                f_s, u_s, bv_s = scaler.scale_in(f, u, bv)

                pred = model(f_s, m, bv_s)

                # ---- Loss ----
                if problem_type == 'elasticity':
                    eps_pred = compute_strain_tensor(pred, model.dx)
                    eps_true = compute_strain_tensor(u_s, model.dx)
                    mse_disp = loss_fn(pred, u_s)
                    mse_strain = loss_fn(eps_pred, eps_true)
                    mse_loss = 0.2 * mse_disp + 0.8 * mse_strain
                else:
                    # scalar thermal: field MSE + gradient MSE
                    Txp, Typ, Tzp = grad_3d_scalar(pred, model.dx)
                    Txt, Tyt, Tzt = grad_3d_scalar(u_s, model.dx)
                    mse_field = loss_fn(pred, u_s)
                    mse_grad = (loss_fn(Txp, Txt) + loss_fn(Typ, Tyt) + loss_fn(Tzp, Tzt)) / 3.0
                    mse_loss = 0.5 * mse_field + 0.5 * mse_grad
                    mse_disp = mse_field

                # Regularizers (optional)
                noise_kernel = model.get_noise_kernel(pred, f_s, m, bv_s)
                corr_tensor = getattr(model, '_last_correction', torch.zeros_like(pred))
                l1_reg = l1_lambda * torch.mean(torch.abs(noise_kernel))
                l2_reg = l2_lambda * torch.mean(noise_kernel**2)
                l1_corr= l1_lambda_corr * torch.mean(torch.abs(corr_tensor))
                l2_corr= l2_lambda_corr * torch.mean(corr_tensor**2)
                reg_loss = l1_reg + l2_reg + l1_corr + l2_corr

                loss = mse_loss + reg_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # NOTE: be sure to pass max_norm (this line is correct)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_mse += mse_disp.item()
                total_reg += reg_loss.item()
                total_rel += relative_l2(pred, u_s)

            # ---- Validation ----
            model.eval()
            val_mse = 0.0
            val_rel = 0.0
            with torch.no_grad():
                for f, u, m, bv in val_loader:
                    f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                    f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                    pred = model(f_s, m, bv_s)
                    val_mse += loss_fn(pred, u_s).item()
                    val_rel += relative_l2(pred, u_s)

            avg_train_mse = total_mse / max(1, len(train_loader))
            avg_reg_loss  = total_reg / max(1, len(train_loader))
            avg_rel_l2    = total_rel / max(1, len(train_loader))
            avg_val_mse   = val_mse   / max(1, len(val_loader))
            avg_val_rel   = val_rel   / max(1, len(val_loader))

            # OOD (elasticity only)
            avg_ood_mse = None
            if ood_loader is not None:
                ood_mse = 0.0
                ood_rel = 0.0
                with torch.no_grad():
                    for f, u, m, bv in ood_loader:
                        f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                        f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                        pred = model(f_s, m, bv_s)
                        ood_mse += loss_fn(pred, u_s).item()
                        ood_rel += relative_l2(pred, u_s)
                avg_ood_mse = ood_mse / max(1, len(ood_loader))
                avg_ood_rel = ood_rel / max(1, len(ood_loader))

            if scheduler:
                scheduler.step(avg_val_mse)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train MSE: {avg_train_mse:.6e} | RelL2: {avg_rel_l2:.6e}")
            print(f"  Reg Loss : {avg_reg_loss:.6e}")
            print(f"  Val  MSE : {avg_val_mse:.6e}")
            print(f"  Val  Rel : {avg_val_rel:.6e}")
            if avg_ood_mse is not None:
                print(f"  OOD  MSE : {avg_ood_mse:.6e}")
                print(f"  OOD  Rel : {avg_ood_rel:.6e}")

            # save checkpoint
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
                print(f"  New best model saved to {best_model_path}")

            # log
            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_mse, avg_val_mse, avg_ood_mse])

# ---------------- Script entry ----------------
if __name__ == "__main__":
    # import time
    # time.sleep(10500)

    n = 1
    for l in range(3, 4):
        for i in range(6, 7):
            mode = "nonlinear_twopeak"
            data_dir = f"./data_{mode}"

            model_path = f"model_{mode}/hlcns/checkpoint_hlcns_n{n}_l{l}_{i}1.pt"
            best_model_path = f"model_{mode}/hlcns/best_hlcns_n{n}_l{l}_{i}1.pt"
            log_path = f"model_{mode}/hlcns/{mode}_n{n}_l{l}_{i}.csv"

            # Example 1: Elasticity
            ensure_parent_dir(log_path)
            train_hlcns(
                data_dir, model_path, best_model_path,
                problem_type='elasticity',
                num_epochs=60, batch_size=10,
                E=1.0, nu=0.3,
                log_csv_path=log_path,
                n_cycles=n, max_levels=l
            )
