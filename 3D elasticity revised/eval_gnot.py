import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark_utils import print_checkpoint_load, print_metrics, print_param_count, resolve_existing_path
from train_gnot import GNOTElasticity3D
from train_hlcns import FieldDataset, GlobalScaler, relative_l2


@torch.no_grad()
def eval_gnot(data_dir: str, model_path: str, best_model_path: str | None = None, batch_size: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_existing_path(model_path)
    best_model_path = resolve_existing_path(best_model_path) if best_model_path else None

    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    ood_dataset = FieldDataset(data_dir, prefix="dualforce_")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GNOTElasticity3D().to(device)
    scaler = GlobalScaler(quantile=0.99)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        scaler.S = float(ckpt.get("scale_S", 0.0)) or None
        if best_model_path and os.path.exists(best_model_path):
            state = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
            print_checkpoint_load(best_model_path, isinstance(state, dict) and "model" in state)
        else:
            model.load_state_dict(ckpt["model"])
            print_checkpoint_load(model_path, True)
    else:
        model.load_state_dict(ckpt)
        print_checkpoint_load(model_path, False)
    if scaler.S is None:
        scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
    model.eval()
    print(f"Scale S: {scaler.S}")
    print_param_count(sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))

    loss_fn = nn.MSELoss()

    def run_split(name, loader):
        mse = 0.0
        rel = 0.0
        for f, u, m, bv in loader:
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            mse += loss_fn(pred, u_s).item()
            rel += relative_l2(pred, u_s)
        print_metrics(name, mse / max(1, len(loader)), rel / max(1, len(loader)))

    run_split("Train", train_loader)
    run_split("Val", val_loader)
    run_split("OOD", ood_loader)


if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    data_dir = f"./data_{mode}"
    for i in range(1, 6):
        eval_gnot(
            data_dir=data_dir,
            model_path=f"model_{mode}/gnot/checkpoint_gnot_{i}.pt",
            best_model_path=f"model_{mode}/gnot/best_gnot_{i}.pt",
            batch_size=10,
        )
