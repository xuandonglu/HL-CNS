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

# ===================== FCN Model =====================

class FCNElasticity(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(FCNElasticity, self).__init__()

        # Architecture with ~1.44M parameters
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(48, 72, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(72, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(96, 72, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(72, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(24, out_channels, kernel_size=1)
        )
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, f, bc_mask, bc_val):
        out = self.net(f)
        return out * (1 - bc_mask) + bc_val * bc_mask

# ===================== Dataset & Scaler (using your code) =====================

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

# ===================== Training Loop =====================

def train_elasticity(
    data_dir, model_path, best_model_path, num_epochs=20, batch_size=8,
    log_csv_path="training_log_fcn.csv"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets / loaders
    train_dataset = ElasticityDataset(data_dir, prefix="sample_")
    val_dataset = ElasticityDataset(data_dir, prefix="validation_")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = FCNElasticity().to(device)

    # total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

    loss_fn = nn.MSELoss()
    with open(log_csv_path, mode='w', newline='') as log_file:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_train_loss = 0.0

            for f, u, bc_mask, bc_val in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                f = f.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)
                m = bc_mask.to(device, non_blocking=True)
                bv = bc_val.to(device, non_blocking=True)

                # Apply scaling
                f_s, u_s, bv_s = scaler.scale_in(f, u, bv)

                # Forward pass
                pred = model(f_s, m, bv_s)

                # Compute losses
                mse_disp = loss_fn(pred, u_s)
                
                # Strain comparison
                eps_pred = compute_strain_tensor(pred, 1.0 / (u_s.shape[-1] - 1))
                eps_true = compute_strain_tensor(u_s, 1.0 / (u_s.shape[-1] - 1))
                mse_strain = loss_fn(eps_pred, eps_true)
                
                # Combined loss
                loss = 0.2 * mse_disp + 0.8 * mse_strain

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for f, u, bc_mask, bc_val in val_loader:
                    f = f.to(device)
                    u = u.to(device)
                    m = bc_mask.to(device)
                    bv = bc_val.to(device)
                    
                    f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                    pred = model(f_s, m, bv_s)
                    val_loss = loss_fn(pred, u_s)
                    total_val_loss += val_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            if scheduler:
                scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scale_S': scaler.S,
            }
            torch.save(state, model_path)
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ New best model saved")

    # ===== Final evaluation (physical units) =====
    model.eval()
    with torch.no_grad():
        f_sample, u_sample, bc_mask, bc_val = val_dataset[0]
        f_sample = f_sample.unsqueeze(0).to(device)
        u_sample = u_sample.unsqueeze(0).to(device)
        bc_mask = bc_mask.unsqueeze(0).to(device)
        bc_val = bc_val.unsqueeze(0).to(device)

        f_s, u_s, bv_s = scaler.scale_in(f_sample, u_sample, bc_val)
        u_pred_s = model(f_s, bc_mask, bv_s)

        u_pred_phys = scaler.u_to_phys(u_pred_s)
        u_true_phys = u_sample

        interior = ~bc_mask.bool()
        final_mse_phys = nn.MSELoss()(u_pred_phys[interior], u_true_phys[interior]).item()
        max_err_phys = (u_pred_phys - u_true_phys).abs().max().item()

        print("\nFinal Test (PHYSICAL):")
        print(f"  MSE: {final_mse_phys:.6f}  |  MaxErr: {max_err_phys:.6f}")

# ============================ run ============================
if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    data_dir = f"./data_{mode}"

    for i in range(1, 2):
        model_path = f"model_{mode}/fcn/checkpoint_fcn_{i}.pt"
        best_model_path = f"model_{mode}/fcn/best_fcn_{i}.pt"
        log_path = f"model_{mode}/fcn/best_fcn_{i}.csv"

        train_elasticity(
            data_dir=data_dir,
            model_path=model_path,
            best_model_path=best_model_path,
            num_epochs=100,
            batch_size=10,
            log_csv_path=log_path
        )