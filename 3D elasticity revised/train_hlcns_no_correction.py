import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_utils import ensure_parent_dir
from model_hlcns import HLCNSNoCorrection3D
from plot_tools import compute_strain_tensor
from train_hlcns import FieldDataset, GlobalScaler, relative_l2


def train_hlcns_no_correction(
    data_dir: str,
    model_path: str,
    best_model_path: str,
    model_cls=HLCNSNoCorrection3D,
    problem_type: str = "elasticity",
    num_epochs: int = 60,
    batch_size: int = 10,
    lr: float = 1e-3,
    n_cycles: int = 1,
    max_levels: int = 3,
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
    log_csv_path: str = "training_log_no_correction.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  |  Total: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    loss_fn = nn.MSELoss()
    scaler = GlobalScaler(quantile=0.99)
    start_epoch = 0
    best_val = float("inf")

    import os
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        scaler.S = float(checkpoint.get("scale_S", 0.0)) or None
        if scaler.S is None:
            scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"Resuming from epoch {start_epoch}; S = {scaler.S:.3e}")
    else:
        print("Starting fresh.")
        scaler.fit_from_loader(train_loader, max_batches=len(train_loader), device=device)
        print(f"[Scaler] S = {scaler.S:.3e} (u,f,bc_val divided by S)")

    ensure_parent_dir(model_path)
    ensure_parent_dir(best_model_path)
    ensure_parent_dir(log_csv_path)

    with open(log_csv_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)", "OOD(MSE)", "OOD(RelL2)"])

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_mse = 0.0
        train_rel = 0.0
        for f, u, m, bv in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
            f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
            pred = model(f_s, m, bv_s)
            eps_pred = compute_strain_tensor(pred, model.dx)
            eps_true = compute_strain_tensor(u_s, model.dx)
            mse_disp = loss_fn(pred, u_s)
            mse_strain = loss_fn(eps_pred, eps_true)
            loss = 0.2 * mse_disp + 0.8 * mse_strain

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_mse += mse_disp.item()
            train_rel += relative_l2(pred, u_s)

        def evaluate(loader):
            mse = 0.0
            rel = 0.0
            model.eval()
            with torch.no_grad():
                for f, u, m, bv in loader:
                    f, u, m, bv = f.to(device), u.to(device), m.to(device), bv.to(device)
                    f_s, u_s, bv_s = scaler.scale_in(f, u, bv)
                    pred = model(f_s, m, bv_s)
                    mse += loss_fn(pred, u_s).item()
                    rel += relative_l2(pred, u_s)
            return mse / max(1, len(loader)), rel / max(1, len(loader))

        avg_train = train_mse / max(1, len(train_loader))
        avg_train_rel = train_rel / max(1, len(train_loader))
        avg_val, avg_val_rel = evaluate(val_loader)
        avg_ood, avg_ood_rel = evaluate(ood_loader) if ood_loader is not None else (None, None)
        scheduler.step(avg_val)

        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scale_S": scaler.S}
        torch.save(state, model_path)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train MSE: {avg_train:.6e} | RelL2: {avg_train_rel:.6e}")
        print(f"  Val   MSE: {avg_val:.6e} | RelL2: {avg_val_rel:.6e}")
        if avg_ood is not None:
            print(f"  OOD   MSE: {avg_ood:.6e} | RelL2: {avg_ood_rel:.6e}")

        with open(log_csv_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, avg_train, avg_train_rel, avg_val, avg_val_rel, avg_ood, avg_ood_rel])


if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    n = 1
    l = 3
    for i in range(1, 6):
        data_dir = f"./data_{mode}"
        model_dir = f"model_{mode}/hlcns_no_correction"
        train_hlcns_no_correction(
            data_dir=data_dir,
            model_path=f"{model_dir}/checkpoint_hlcns_no_correction_n{n}_l{l}_{i}.pt",
            best_model_path=f"{model_dir}/best_hlcns_no_correction_n{n}_l{l}_{i}.pt",
            problem_type="elasticity",
            num_epochs=60,
            batch_size=10,
            E=1.0,
            nu=0.3,
            n_cycles=n,
            max_levels=l,
            log_csv_path=f"{model_dir}/hlcns_no_correction_{mode}_n{n}_l{l}_{i}.csv",
        )
