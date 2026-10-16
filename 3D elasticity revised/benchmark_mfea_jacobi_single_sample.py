from __future__ import annotations

import argparse
import csv
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
from torch.utils.data import DataLoader

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
from train_mfea import FieldDataset


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _norm_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)


def _load_psi(
    psi_path: str,
    nb_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[PsiNet3D, dict]:
    ckpt = torch.load(psi_path, map_location=device)
    saved_layers = int(ckpt.get("nb_layers", nb_layers)) if isinstance(ckpt, dict) else nb_layers
    psi = PsiNet3D(nb_layers=saved_layers, channels=3).to(device=device, dtype=dtype)
    psi.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    psi.eval()
    return psi, ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-sample MFEA-Net benchmark using Jacobi iterations plus the learned PsiNet smoother."
    )
    parser.add_argument("--data-dir", default="./data_linear_twopeak")
    parser.add_argument("--prefix", default="validation_")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--psi-path", default="model_linear_twopeak/mfea/best_mfea_1.pt")
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--nb-layers", type=int, default=3)
    parser.add_argument("--E", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="model_linear_twopeak/comparison/mfea_jacobi_single_sample.tsv")
    args = parser.parse_args()

    if "linear" not in os.path.basename(os.path.normpath(args.data_dir)).lower():
        raise ValueError(f"MFEA-Net Jacobi benchmark must use a linear data folder, got: {args.data_dir}")
    if not os.path.exists(args.psi_path):
        raise FileNotFoundError(args.psi_path)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dataset = FieldDataset(args.data_dir, prefix=args.prefix)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} is outside [0, {len(dataset) - 1}]")

    psi, meta = _load_psi(args.psi_path, args.nb_layers, device=device, dtype=dtype)
    omega = float(meta.get("omega", args.omega))

    scale_loader = DataLoader(FieldDataset(args.data_dir, prefix="sample_"), batch_size=10, shuffle=False, num_workers=0)
    scaler = GlobalScaler(quantile=0.99)
    if meta.get("scale_S", None) is not None:
        scaler.S = float(meta["scale_S"])
    else:
        scaler.fit_from_loader(scale_loader, max_batches=None, device=device)

    f, u_true, bc_mask, bc_val = dataset[args.sample_index]
    f = f.unsqueeze(0).to(device=device, dtype=dtype)
    u_true = u_true.unsqueeze(0).to(device=device, dtype=dtype)
    bc_mask = bc_mask.unsqueeze(0).to(device=device, dtype=dtype)
    bc_val = bc_val.unsqueeze(0).to(device=device, dtype=dtype)

    f_s, _, bc_val_s = scaler.scale_in(f, u_true, bc_val)
    level = LinearElasticityLevel3D.build(
        N=f.shape[-1],
        E=args.E,
        nu=args.nu,
        device=device,
        dtype=dtype,
    )

    free = free_dof_mask(bc_mask)
    rhs = rhs_from_density(f_s, level.dx, bc_mask, bc_val_s)
    rhs_norm = _norm_batch(rhs * free).clamp_min(1e-30)
    u = enforce_dirichlet(torch.zeros_like(f_s), bc_mask, bc_val_s)

    # Warm up one MFEA step so timing excludes first-call overhead.
    _ = jacobi_update(u, rhs, level, bc_mask, bc_val_s, omega=omega, psi_net=psi, use_psi=True)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict] = []
    cumulative = 0.0
    for it in range(1, int(args.iterations) + 1):
        _sync(device)
        t0 = time.perf_counter()
        u = jacobi_update(u, rhs, level, bc_mask, bc_val_s, omega=omega, psi_net=psi, use_psi=True)
        _sync(device)
        dt = time.perf_counter() - t0
        cumulative += dt

        pred = scaler.u_to_phys(u)
        mse = ((pred - u_true) * free).pow(2).reshape(1, -1).mean().item()
        rel = relative_l2(pred, u_true, free)
        residual = (rhs - level.apply(u)) * free
        residual_rel = (_norm_batch(residual) / rhs_norm).mean().item()
        rows.append(
            {
                "Method": "MFEA-Net-Jacobi",
                "PsiPath": args.psi_path,
                "SampleIndex": args.sample_index,
                "Iteration": it,
                "Val MSE": mse,
                "Val RelL2 (%)": rel * 100.0,
                "ResidualRel": residual_rel,
                "StepTimeMs": dt * 1000.0,
                "CumulativeTimeMs": cumulative * 1000.0,
            }
        )

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0**2) if device.type == "cuda" else 0.0
    params = sum(p.numel() for p in psi.parameters())

    ensure_parent_dir(args.out)
    fields = [
        "Method",
        "PsiPath",
        "SampleIndex",
        "Iteration",
        "Val MSE",
        "Val RelL2 (%)",
        "ResidualRel",
        "StepTimeMs",
        "CumulativeTimeMs",
    ]
    with open(args.out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Method": row["Method"],
                    "PsiPath": row["PsiPath"],
                    "SampleIndex": row["SampleIndex"],
                    "Iteration": row["Iteration"],
                    "Val MSE": f"{row['Val MSE']:.6e}",
                    "Val RelL2 (%)": f"{row['Val RelL2 (%)']:.6f}",
                    "ResidualRel": f"{row['ResidualRel']:.6e}",
                    "StepTimeMs": f"{row['StepTimeMs']:.6f}",
                    "CumulativeTimeMs": f"{row['CumulativeTimeMs']:.6f}",
                }
            )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Scale S: {scaler.S:.6e}")
    print(f"PsiNet: {args.psi_path} | params: {params:,}")
    print("Operator: LinearElasticityLevel3D field Ktab, Jacobi plus PsiNet smoother, no multigrid levels")
    print(f"Sample: {args.prefix}{args.sample_index} | iterations: {args.iterations} | omega: {omega}")
    print(f"PeakMemoryMB: {peak_memory:.2f}")
    print(f"Saved: {args.out}")
    print("\t".join(fields))
    for row in rows:
        print(
            "\t".join(
                [
                    row["Method"],
                    row["PsiPath"],
                    str(row["SampleIndex"]),
                    str(row["Iteration"]),
                    f"{row['Val MSE']:.6e}",
                    f"{row['Val RelL2 (%)']:.6f}",
                    f"{row['ResidualRel']:.6e}",
                    f"{row['StepTimeMs']:.6f}",
                    f"{row['CumulativeTimeMs']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
