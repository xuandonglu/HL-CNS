import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark_utils import print_checkpoint_load, print_metrics, print_param_count, resolve_existing_path
from model_hlcns import HLCNSNoCorrection3D
from train_hlcns import FieldDataset, GlobalScaler, relative_l2


@torch.no_grad()
def eval_hlcns_no_correction(
    data_dir: str,
    model_path: str,
    best_model_path: str | None = None,
    model_cls=HLCNSNoCorrection3D,
    problem_type: str = "elasticity",
    batch_size: int = 10,
    n_cycles: int = 1,
    max_levels: int = 3,
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_existing_path(model_path)
    best_model_path = resolve_existing_path(best_model_path) if best_model_path else None

    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    ood_dataset = FieldDataset(data_dir, prefix="dualforce_") if problem_type == "elasticity" else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) if ood_dataset else None

    f0, _, _, _ = train_dataset[0]
    model = model_cls(
        N=f0.shape[-1],
        problem_type=problem_type,
        n_cycles=n_cycles,
        max_levels=max_levels,
        E=E,
        nu=nu,
        kappa=kappa,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(f"Expected checkpoint dict with scale_S and model: {model_path}")
    scaler = GlobalScaler(quantile=0.99)
    scaler.S = float(ckpt.get("scale_S", 0.0)) or None
    if scaler.S is None:
        scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)

    if best_model_path and os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print_checkpoint_load(best_model_path, isinstance(state, dict) and "model" in state)
    else:
        model.load_state_dict(ckpt["model"])
        print_checkpoint_load(model_path, True)

    model.eval()
    print(f"Scale S: {scaler.S}")
    print_param_count(sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
    loss_fn = nn.MSELoss()

    def run_split(name, loader):
        mse = 0.0
        rel = 0.0
        for f, u, m, bv in loader:
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            f_s, _, bv_s = scaler.scale_in(f, u, bv)
            pred = scaler.u_to_phys(model(f_s, m, bv_s))
            mse += loss_fn(pred, u).item()
            rel += relative_l2(pred, u)
        print_metrics(name, mse / max(1, len(loader)), rel / max(1, len(loader)))

    run_split("Train", train_loader)
    run_split("Val", val_loader)
    if ood_loader is not None:
        run_split("OOD", ood_loader)


if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    n = 1
    l = 3
    for i in range(1, 6):
        data_dir = f"./data_{mode}"
        model_dir = f"model_{mode}/hlcns_no_correction"
        eval_hlcns_no_correction(
            data_dir=data_dir,
            model_path=f"{model_dir}/checkpoint_hlcns_no_correction_n{n}_l{l}_{i}.pt",
            best_model_path=f"{model_dir}/best_hlcns_no_correction_n{n}_l{l}_{i}.pt",
            problem_type="elasticity",
            batch_size=10,
            E=1.0,
            nu=0.3,
            n_cycles=n,
            max_levels=l,
        )
