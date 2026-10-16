from __future__ import annotations

import csv
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mfea_kernel_3d import (
    GlobalScaler,
    PsiNet3D,
    build_vcycle_levels_3d,
    free_dof_mask,
    relative_l2,
    run_vcycles,
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


def _load_psi(
    psi_path: str,
    nb_layers: int,
    device: torch.device,
) -> tuple[PsiNet3D, dict]:
    ckpt = torch.load(psi_path, map_location=device)
    saved_layers = int(ckpt.get("nb_layers", nb_layers)) if isinstance(ckpt, dict) else nb_layers
    psi = PsiNet3D(nb_layers=saved_layers, channels=3).to(device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        psi.load_state_dict(ckpt["model"])
        meta = ckpt
    else:
        psi.load_state_dict(ckpt)
        meta = {}
    psi.eval()
    return psi, meta


@torch.no_grad()
def _eval_loader(
    name: str,
    loader: DataLoader,
    levels: list,
    mask_levels: list,
    val_levels: list,
    scaler: GlobalScaler,
    psi: PsiNet3D | None,
    use_psi: bool,
    cycles: int,
    pre: int,
    post: int,
    omega: float,
    init: str,
    device: torch.device,
) -> dict:
    loss_fn = nn.MSELoss()
    mse = 0.0
    rel = 0.0
    batches = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f, u, bc_mask, bc_val in loader:
        f = f.to(device)
        u = u.to(device)
        bc_mask = bc_mask.to(device)
        bc_val = bc_val.to(device)
        f_s, u_s, bc_val_s = scaler.scale_in(f, u, bc_val)
        pred_s = run_vcycles(
            f_s,
            bc_mask,
            bc_val_s,
            levels,
            mask_levels,
            val_levels,
            cycles=cycles,
            pre=pre,
            post=post,
            omega=omega,
            psi_net=psi,
            use_psi=use_psi,
            init=init,
        )
        pred = scaler.u_to_phys(pred_s)
        free = free_dof_mask(bc_mask)
        mse += loss_fn(pred * free, u * free).item()
        rel += relative_l2(pred, u, free)
        batches += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_mb = 0.0
    elapsed = time.perf_counter() - t0
    return {
        "Split": name,
        "MSE": mse / max(1, batches),
        "RelL2": rel / max(1, batches),
        "TimeSec": elapsed,
        "MaxMemoryMB": mem_mb,
    }


def eval_mfea(
    data_dir: str = "./data_linear_twopeak",
    psi_path: str | None = "model_linear_twopeak/mfea/best_mfea_1.pt",
    use_psi: bool = True,
    log_csv_path: str | None = "model_linear_twopeak/mfea/eval_mfea_1.csv",
    batch_size: int = 4,
    cycles: int = 4,
    pre: int = 1,
    post: int = 1,
    max_levels: int = 3,
    omega: float = 0.6,
    nb_layers: int = 3,
    init: str = "zeros",
    E: float = 1.0,
    nu: float = 0.3,
    seed: int = 0,
) -> list[dict]:
    _assert_linear_data_dir(data_dir)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FieldDataset(data_dir, prefix="sample_")
    val_dataset = FieldDataset(data_dir, prefix="validation_")
    ood_dataset = FieldDataset(data_dir, prefix="dualforce_")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    f0, _, _, _ = train_dataset[0]
    _, _, bc_mask0, bc_val0 = train_dataset[0]
    levels, mask_levels, val_levels = build_vcycle_levels_3d(
        N=f0.shape[-1],
        bc_mask=bc_mask0,
        bc_val=bc_val0,
        E=E,
        nu=nu,
        max_levels=max_levels,
        device=device,
        dtype=torch.float32,
    )

    psi = None
    meta = {}
    if use_psi:
        if not psi_path:
            raise ValueError("use_psi=True requires psi_path.")
        psi, meta = _load_psi(psi_path, nb_layers=nb_layers, device=device)
        omega = float(meta.get("omega", omega))
        print(f"[Loaded psi_net] {psi_path}")
    else:
        print("[FEA mode] psi_net disabled")

    scaler = GlobalScaler(quantile=0.99)
    if use_psi and meta.get("scale_S", None) is not None:
        scaler.S = float(meta["scale_S"])
    else:
        scaler.fit_from_loader(train_loader, max_batches=None, device=device)
    print(f"Scale S: {scaler.S:.6e}")

    trainable = sum(p.numel() for p in psi.parameters() if p.requires_grad) if psi is not None else 0
    total = sum(p.numel() for p in psi.parameters()) if psi is not None else 0
    print(f"Parameters -- trainable: {trainable:,} | total: {total:,}")
    print(f"V-cycles: {cycles} | pre/post: {pre}/{post} | omega: {omega} | use_psi: {use_psi} | init: {init}")

    rows = []
    for split, loader in [("Train", train_loader), ("Val", val_loader), ("OOD", ood_loader)]:
        metrics = _eval_loader(
            split,
            loader,
            levels,
            mask_levels,
            val_levels,
            scaler,
            psi,
            use_psi,
            cycles,
            pre,
            post,
            omega,
            init,
            device,
        )
        rows.append(metrics)
        print(
            f"{split:<5} -- MSE: {metrics['MSE']:.6e} | Rel: {metrics['RelL2']:.6e} | "
            f"Time: {metrics['TimeSec']:.2f}s | MaxMem: {metrics['MaxMemoryMB']:.1f} MB"
        )

    if log_csv_path:
        ensure_parent_dir(log_csv_path)
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Model",
                    "UsePsi",
                    "Cycles",
                    "PreSmooth",
                    "PostSmooth",
                    "Omega",
                    "Init",
                    "Split",
                    "MSE",
                    "RelL2",
                    "TimeSec",
                    "MaxMemoryMB",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Model": "MFEA-Net" if use_psi else "FEA-Net",
                        "UsePsi": int(use_psi),
                        "Cycles": cycles,
                        "PreSmooth": pre,
                        "PostSmooth": post,
                        "Omega": omega,
                        "Init": init,
                        **row,
                    }
                )
    return rows


if __name__ == "__main__":
    mode = "linear_twopeak"
    data_dir = f"./data_{mode}"
    cycles = 4

    eval_mfea(
        data_dir=data_dir,
        psi_path=None,
        use_psi=False,
        log_csv_path=f"model_{mode}/fea/eval_fea_vcycle{cycles}.csv",
        batch_size=10,
        cycles=cycles,
        omega=0.6,
        init="zeros",
        seed=2026,
    )

    for trial in range(1, 6):
        eval_mfea(
            data_dir=data_dir,
            psi_path=f"model_{mode}/mfea/best_mfea_{trial}.pt",
            use_psi=True,
            log_csv_path=f"model_{mode}/mfea/eval_mfea_{trial}_vcycle{cycles}.csv",
            batch_size=10,
            cycles=cycles,
            omega=0.6,
            nb_layers=3,
            init="zeros",
            seed=2026 + trial,
        )
