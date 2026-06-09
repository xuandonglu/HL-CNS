import os
import torch
from torch.utils.data import DataLoader

from benchmark_utils import (
    benchmark_tag_from_path,
    export_mat_batch,
    print_checkpoint_load,
    print_metrics,
    resolve_existing_path,
)
from train_hlcns import (
    HLCNSSolver2D,
    PackedTensorDataset2D,
    create_bc_mask_and_value_like_2d,
)

@torch.no_grad()
def eval_2d_hlcns(
    tra_pt: str,
    val_pt: str,
    ood_pt: str,
    best_ckpt: str,
    batch_size: int = 16,
    n_cycles: int = 1,
    max_levels: int = 3,
    kappa: float = 1.0,
    omega0: float = 0.99,
    device: str | None = None,
    export_mat: bool = False,
    mat_output_dir: str = "mat_output",
    mat_tag: str | None = None,
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    best_ckpt = resolve_existing_path(best_ckpt)
    mat_tag = mat_tag or benchmark_tag_from_path(best_ckpt)

    tra_ds = PackedTensorDataset2D(tra_pt)
    tra_loader = DataLoader(tra_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    val_ds = PackedTensorDataset2D(val_pt)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ood_ds = PackedTensorDataset2D(ood_pt)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    _, _, H, W = val_ds.f.shape

    model = HLCNSSolver2D(
        N=H,
        n_cycles=n_cycles,
        max_levels=max_levels,
        omega=omega0,
        kappa=kappa,
        prolong_mode='bilinear',
    ).to(device)
    model.eval()

    state = torch.load(best_ckpt, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
        print_checkpoint_load(best_ckpt, True)
    else:
        model.load_state_dict(state)
        print_checkpoint_load(best_ckpt, False)

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for f, u in tra_loader:
        f = f.to(device, non_blocking=True)   # [B,1,H,W]
        u = u.to(device, non_blocking=True)   # [B,1,H,W]
        bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
        pred = model(f, bc_mask, bc_val)     # [B,1,H,W]
        interior = (bc_mask < 0.5)
        diff2 = (pred - u).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print_metrics("Train", mse, rel)

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for batch_idx, (f, u) in enumerate(val_loader):
        f = f.to(device, non_blocking=True)   # [B,1,H,W]
        u = u.to(device, non_blocking=True)   # [B,1,H,W]
        bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)  # [B,1,H,W], [B,1,H,W]
        pred = model(f, bc_mask, bc_val)      # [B,1,H,W]
        export_mat_batch(
            export_mat,
            mat_output_dir,
            "hlcns",
            mat_tag,
            "val",
            batch_idx,
            {"f": f, "pred": pred, "exact": u},
        )
        interior = (bc_mask < 0.5)
        diff2 = (pred - u).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print_metrics("Val", mse, rel)

    mse_num = 0.0
    mse_den = 0
    rel_num2 = 0.0
    rel_den2 = 0.0
    for f, u in ood_loader:
        f = f.to(device, non_blocking=True)   # [B,1,H,W]
        u = u.to(device, non_blocking=True)   # [B,1,H,W]
        bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)  # [B,1,H,W], [B,1,H,W]
        pred = model(f, bc_mask, bc_val)      # [B,1,H,W]
        interior = (bc_mask < 0.5)
        diff2 = (pred - u).pow(2)
        mse_num += diff2[interior].sum().item()
        mse_den += int(interior.sum().item())
        rel_num2 += diff2[interior].sum().item()
        rel_den2 += (u.pow(2))[interior].sum().item()
    eps = 1e-12
    mse = mse_num / max(1, mse_den)
    rel = (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + eps)
    print_metrics("OOD", mse, rel)

if __name__ == "__main__":
    mode = "decrease"   # "linear" / "increase" / "decrease"
    tra_pt   = f"data/training_data_N32_{mode}.pt"
    val_pt   = f"data/validation_data_N32_{mode}.pt"
    ood_pt   = f"data/dualforce_data_N32_{mode}.pt"

    for i in range(1, 2):
        best_ckpt = f"model_{mode}/hlcns/best_2d_{mode}_l3_n1_{i}.pt"
        eval_2d_hlcns(
            tra_pt=tra_pt,
            val_pt=val_pt,
            ood_pt=ood_pt,
            best_ckpt=best_ckpt,
            batch_size=10,
            n_cycles=1,
            max_levels=3,
            kappa=1.0,
            omega0=0.99,
            export_mat=False)
