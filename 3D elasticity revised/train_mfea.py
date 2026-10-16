from __future__ import annotations

import csv
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from mfea_kernel_3d import (
    GlobalScaler,
    LinearElasticityLevel3D,
    PsiNet3D,
    enforce_dirichlet,
    free_dof_mask,
    jacobi_update,
    relative_l2,
    rhs_from_density,
    set_seed,
)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _assert_linear_data_dir(data_dir: str) -> None:
    if "linear" not in os.path.basename(os.path.normpath(data_dir)).lower():
        raise ValueError(f"MFEA/FEA baseline for this request must use a linear data folder, got: {data_dir}")


class FieldDataset(Dataset):
    def __init__(self, data_dir: str, prefix: str = "sample_"):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])
        if not self.file_list:
            raise FileNotFoundError(f"No files with prefix {prefix!r} found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = torch.load(os.path.join(self.data_dir, self.file_list[idx]), map_location="cpu")
        return (
            data["f"].float(),
            data["u"].float(),
            data["bc_mask"].float(),
            data["bc_val"].float(),
        )


@torch.no_grad()
def evaluate_one_step(
    psi: PsiNet3D,
    loader: DataLoader,
    level: LinearElasticityLevel3D,
    scaler: GlobalScaler,
    omega: float,
    device: torch.device,
    init_scale: float = 1.0,
) -> tuple[float, float]:
    psi.eval()
    loss_fn = nn.MSELoss()
    mse_total = 0.0
    rel_total = 0.0
    for f, u, bc_mask, bc_val in loader:
        f = f.to(device)
        u = u.to(device)
        bc_mask = bc_mask.to(device)
        bc_val = bc_val.to(device)
        f_s, u_s, bc_val_s = scaler.scale_in(f, u, bc_val)
        rhs = rhs_from_density(f_s, level.dx, bc_mask, bc_val_s)
        u_init = init_scale * torch.randn_like(u_s)
        u_init = enforce_dirichlet(u_init, bc_mask, bc_val_s)
        pred = jacobi_update(
            u_init,
            rhs,
            level,
            bc_mask,
            bc_val_s,
            omega=omega,
            psi_net=psi,
            use_psi=True,
        )
        free = free_dof_mask(bc_mask)
        mse_total += loss_fn(pred * free, u_s * free).item()
        rel_total += relative_l2(pred, u_s, free)
    return mse_total / max(1, len(loader)), rel_total / max(1, len(loader))


def train_mfea(
    data_dir: str = "./data_linear_twopeak",
    model_path: str = "model_linear_twopeak/mfea/checkpoint_mfea_1.pt",
    best_model_path: str = "model_linear_twopeak/mfea/best_mfea_1.pt",
    log_csv_path: str = "model_linear_twopeak/mfea/mfea_1.csv",
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    omega: float = 0.6,
    nb_layers: int = 3,
    init_scale: float = 1.0,
    E: float = 1.0,
    nu: float = 0.3,
    seed: int = 0,
) -> None:
    _assert_linear_data_dir(data_dir)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    f0, _, _, _ = train_dataset[0]
    level = LinearElasticityLevel3D.build(N=f0.shape[-1], E=E, nu=nu, device=device, dtype=torch.float32)
    psi = PsiNet3D(nb_layers=nb_layers, channels=3).to(device)

    optimizer = torch.optim.Adam(psi.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    loss_fn = nn.MSELoss()

    scaler = GlobalScaler(quantile=0.99)
    start_epoch = 0
    best_val = float("inf")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        psi.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.S = float(ckpt["scale_S"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resuming MFEA psi_net from epoch {start_epoch}; S={scaler.S:.6e}")
    else:
        scaler.fit_from_loader(train_loader, max_batches=None, device=device)
        print(f"Starting fresh MFEA psi_net; S={scaler.S:.6e}")

    trainable = sum(p.numel() for p in psi.parameters() if p.requires_grad)
    total = sum(p.numel() for p in psi.parameters())
    print(f"PsiNet3D parameters -- trainable: {trainable:,} | total: {total:,}")

    ensure_parent_dir(model_path)
    ensure_parent_dir(best_model_path)
    ensure_parent_dir(log_csv_path)
    log_mode = "a" if start_epoch > 0 else "w"
    write_header = log_mode == "w" or not os.path.exists(log_csv_path)
    with open(log_csv_path, log_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Epoch", "Train(MSE)", "Train(RelL2)", "Val(MSE)", "Val(RelL2)", "LR"])

    for epoch in range(start_epoch, num_epochs):
        psi.train()
        train_mse = 0.0
        train_rel = 0.0
        for f, u, bc_mask, bc_val in tqdm(train_loader, desc=f"MFEA epoch {epoch + 1}/{num_epochs}"):
            f = f.to(device)
            u = u.to(device)
            bc_mask = bc_mask.to(device)
            bc_val = bc_val.to(device)
            f_s, u_s, bc_val_s = scaler.scale_in(f, u, bc_val)
            rhs = rhs_from_density(f_s, level.dx, bc_mask, bc_val_s)

            u_init = init_scale * torch.randn_like(u_s)
            u_init = enforce_dirichlet(u_init, bc_mask, bc_val_s)
            pred = jacobi_update(
                u_init,
                rhs,
                level,
                bc_mask,
                bc_val_s,
                omega=omega,
                psi_net=psi,
                use_psi=True,
            )
            free = free_dof_mask(bc_mask)
            loss = loss_fn(pred * free, u_s * free)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psi.parameters(), 1.0)
            optimizer.step()

            train_mse += loss.item()
            train_rel += relative_l2(pred.detach(), u_s, free)

        avg_train_mse = train_mse / max(1, len(train_loader))
        avg_train_rel = train_rel / max(1, len(train_loader))
        avg_val_mse, avg_val_rel = evaluate_one_step(
            psi, val_loader, level, scaler, omega, device, init_scale=init_scale
        )
        scheduler.step(avg_val_mse)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train MSE {avg_train_mse:.6e} Rel {avg_train_rel:.6e} | "
            f"Val MSE {avg_val_mse:.6e} Rel {avg_val_rel:.6e}"
        )

        state = {
            "epoch": epoch,
            "model": psi.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scale_S": scaler.S,
            "best_val": min(best_val, avg_val_mse),
            "omega": omega,
            "nb_layers": nb_layers,
            "E": E,
            "nu": nu,
        }
        torch.save(state, model_path)
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            torch.save(state, best_model_path)
            print(f"  New best psi_net saved to {best_model_path}")

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_mse, avg_train_rel, avg_val_mse, avg_val_rel, lr_now])


if __name__ == "__main__":
    mode = "linear_twopeak"
    data_dir = f"./data_{mode}"
    for trial in range(1, 6):
        train_mfea(
            data_dir=data_dir,
            model_path=f"model_{mode}/mfea/checkpoint_mfea_{trial}.pt",
            best_model_path=f"model_{mode}/mfea/best_mfea_{trial}.pt",
            log_csv_path=f"model_{mode}/mfea/mfea_{trial}.csv",
            num_epochs=50,
            batch_size=10,
            lr=1e-3,
            omega=0.6,
            nb_layers=3,
            seed=2026 + trial,
        )
