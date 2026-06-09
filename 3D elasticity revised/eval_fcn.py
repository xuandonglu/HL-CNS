import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot_tools import compute_strain_tensor, plot_comparison_3d, plot_stress_divergence
from train_fcn import (
    FCNElasticity,
    ElasticityDataset,
    relative_l2,
    GlobalScaler,
)

def eval_fcn(data_dir, model_path, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tra_dataset = ElasticityDataset(data_dir, prefix="sample_")
    val_dataset = ElasticityDataset(data_dir, prefix="validation_")
    ood_dataset = ElasticityDataset(data_dir, prefix="dualforce_")

    tra_loader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FCNElasticity().to(device)
    model.eval()
    loss_fn = nn.MSELoss()

    scaler = GlobalScaler(quantile=0.99)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        scaler.S = float(ckpt.get('scale_S', 0.0)) or None
        if scaler.S is None:
            scaler.fit_from_loader(tra_loader, max_batches=8, device=device)
    else:
        print("No saved model found. Starting training from scratch.")
        scaler.fit_from_loader(tra_loader, max_batches=8, device=device)
        print(f"[Scaler] Using global scale S = {scaler.S:.3e} (u,f,bc_val all divided by S)")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
        print(f"[Loaded checkpoint dict] {model_path}")
    else:
        model.load_state_dict(state)
        print(f"[Loaded state_dict] {model_path}")

    mse = 0
    rel = 0
    with torch.no_grad():
        for f, u, bc_mask, bc_val in tra_loader:
            f = f.to(device)
            u = u.to(device)
            m = bc_mask.to(device)
            bv = bc_val.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            mse += loss_fn(pred, u_s).item()
            rel += relative_l2(pred, u_s)
    avg_mse = mse / len(tra_loader)
    avg_rel = rel / len(tra_loader)
    print(f"Train -- MSE: {avg_mse:.6e} | Rel: {avg_rel:.6e}")

    mse = 0
    rel = 0
    with torch.no_grad():
        for f, u, bc_mask, bc_val in ood_loader:
            f = f.to(device)
            u = u.to(device)
            m = bc_mask.to(device)
            bv = bc_val.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            mse += loss_fn(pred, u_s).item()
            rel += relative_l2(pred, u_s)
    avg_mse = mse / len(ood_loader)
    avg_rel = rel / len(ood_loader)
    print(f"OOD   -- MSE: {avg_mse:.6e} | Rel: {avg_rel:.6e}")

    mse = 0
    rel = 0
    with torch.no_grad():
        for f, u, bc_mask, bc_val in val_loader:
            f = f.to(device)
            u = u.to(device)
            m = bc_mask.to(device)
            bv = bc_val.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            mse += loss_fn(pred, u_s).item()
            rel += relative_l2(pred, u_s)
    avg_mse = mse / len(val_loader)
    avg_rel = rel / len(val_loader)
    print(f"Val   -- MSE: {avg_mse:.6e} | Rel: {avg_rel:.6e}")

if __name__ == "__main__":
    mode = 'nonlinear'
    data_dir = f"./data_{mode}_twopeak"
    for i in range(1, 6):
        best_model_path = f"model_{mode}/fcn/best_fcn_{i}.pt"
        eval_fcn(
            data_dir=data_dir,
            model_path=best_model_path,
            batch_size=10
        )
