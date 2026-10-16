from __future__ import annotations

import argparse
import os
import time

import torch

from mfea_kernel_3d import (
    LinearElasticityLevel3D,
    PsiNet3D,
    build_vcycle_levels_3d,
    free_dof_mask,
    relative_l2,
    rhs_from_density,
    run_vcycles,
)


def _pick_file(data_dir: str, prefix: str, index: int) -> str:
    files = sorted(f for f in os.listdir(data_dir) if f.startswith(prefix))
    if not files:
        raise FileNotFoundError(f"No files with prefix {prefix!r} found in {data_dir}")
    if index < 0 or index >= len(files):
        raise IndexError(f"Sample index {index} out of range for {len(files)} files")
    return os.path.join(data_dir, files[index])


def _norm(x: torch.Tensor) -> float:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean().item()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-sample FEA-Net/MG-Jacobi V-cycle diagnostics.")
    parser.add_argument("--data-dir", default="./data_linear_twopeak")
    parser.add_argument("--prefix", default="validation_")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--cycles", default="0,1,2,4,8,16")
    parser.add_argument("--max-levels", type=int, default=3)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--psi-path", default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    sample_path = _pick_file(args.data_dir, args.prefix, args.index)
    data = torch.load(sample_path, map_location="cpu")

    f = data["f"].to(device=device, dtype=dtype).unsqueeze(0)
    u_true = data["u"].to(device=device, dtype=dtype).unsqueeze(0)
    bc_mask = data["bc_mask"].to(device=device, dtype=dtype).unsqueeze(0)
    bc_val = data["bc_val"].to(device=device, dtype=dtype).unsqueeze(0)

    N = int(f.shape[-1])
    free = free_dof_mask(bc_mask)
    rhs_level = LinearElasticityLevel3D.build(N=N, E=1.0, nu=0.3, device=device, dtype=dtype)
    rhs = rhs_from_density(f, rhs_level.dx, bc_mask, bc_val)
    rhs_norm = _norm(rhs * free)
    true_res = (rhs - rhs_level.apply(u_true)) * free
    true_res_rel = _norm(true_res) / max(rhs_norm, 1e-30)

    print(f"Device: {device}")
    print(f"Sample: {sample_path}")
    print(f"N: {N} | dtype: {dtype} | max_levels: {args.max_levels}")
    print(f"Truth check -- ||F-Ku_true||/||F||: {true_res_rel:.6e}")
    print(f"Truth norm -- ||u_true||: {_norm(u_true * free):.6e} | ||F||: {rhs_norm:.6e}")

    psi = None
    use_psi = False
    omega = float(args.omega)
    if args.psi_path:
        ckpt = torch.load(args.psi_path, map_location=device)
        nb_layers = int(ckpt.get("nb_layers", 3)) if isinstance(ckpt, dict) else 3
        omega = float(ckpt.get("omega", omega)) if isinstance(ckpt, dict) else omega
        psi = PsiNet3D(nb_layers=nb_layers, channels=3).to(device=device, dtype=dtype)
        psi.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        psi.eval()
        use_psi = True
        params = sum(p.numel() for p in psi.parameters())
        print(f"Loaded psi_net: {args.psi_path}")
        print(f"PsiNet params: {params:,} | omega: {omega}")

    t0 = time.perf_counter()
    levels, mask_levels, val_levels = build_vcycle_levels_3d(
        N=N,
        bc_mask=bc_mask[0],
        bc_val=bc_val[0],
        E=1.0,
        nu=0.3,
        max_levels=args.max_levels,
        device=device,
        dtype=dtype,
    )
    print(f"Built levels: {[level.N for level in levels]} | build time: {time.perf_counter() - t0:.3f} s")

    print("Cycles\tRelL2(%)\tMSE\tResidualRel\tMaxAbsPred\tTime(s)")
    for cycles_text in args.cycles.split(","):
        cycles = int(cycles_text.strip())
        t1 = time.perf_counter()
        if cycles == 0:
            pred = torch.zeros_like(u_true)
            pred = pred * (1.0 - bc_mask) + bc_val * bc_mask
        else:
            pred = run_vcycles(
                f,
                bc_mask,
                bc_val,
                levels,
                mask_levels,
                val_levels,
                cycles=cycles,
                pre=args.pre,
                post=args.post,
                omega=omega,
                psi_net=psi,
                use_psi=use_psi,
                init="zeros",
            )
        elapsed = time.perf_counter() - t1
        rel = relative_l2(pred, u_true, free)
        mse = ((pred - u_true) * free).pow(2).mean().item()
        residual = (rhs - levels[0].apply(pred)) * free
        residual_rel = _norm(residual) / max(rhs_norm, 1e-30)
        max_abs_pred = (pred * free).abs().max().item()
        print(f"{cycles}\t{rel * 100:.6f}\t{mse:.6e}\t{residual_rel:.6e}\t{max_abs_pred:.6e}\t{elapsed:.3f}")


if __name__ == "__main__":
    main()
