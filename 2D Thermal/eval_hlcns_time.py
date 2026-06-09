import os
import torch
from torch.utils.data import DataLoader
from timeit import default_timer

from benchmark_utils import print_checkpoint_load, print_time, resolve_existing_path
from train_hlcns import (
    HLCNSSolver2D,
    PackedTensorDataset2D,
    create_bc_mask_and_value_like_2d,
)

@torch.no_grad()
def eval_2d_hlcns_time(
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
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    best_ckpt = resolve_existing_path(best_ckpt)

    tra_ds = PackedTensorDataset2D(tra_pt)
    tra_loader = DataLoader(tra_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    val_ds = PackedTensorDataset2D(val_pt)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

    t_val = 0.0
    for f, u in val_loader:
        f = f.to(device, non_blocking=True)   # [B,1,H,W]
        bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
        pred = model(f, bc_mask, bc_val)     # [B,1,H,W]
        t1 = default_timer()
        pred = model(f, bc_mask, bc_val)     # [B,1,H,W]
        t2 = default_timer()
        t_val += t2 - t1
    print_time("Val", t_val)

if __name__ == "__main__":
    mode = "increase"   # "linear" / "increase" / "decrease"
    tra_pt   = f"data/training_data_N32_{mode}.pt"
    val_pt   = f"data/validation_data_N32_{mode}.pt"
    ood_pt   = f"data/dualforce_data_N32_{mode}.pt"

    for n in range(1, 3):
        for l in range(0, 4):
            best_ckpt = f"model_{mode}/hlcns/best_2d_{mode}_l{l}_n{n}_1.pt"
            eval_2d_hlcns_time(
                tra_pt=tra_pt,
                val_pt=val_pt,
                ood_pt=ood_pt,
                best_ckpt=best_ckpt,
                batch_size=10,
                n_cycles=n,
                max_levels=l,
                kappa=1.0,
                omega0=0.99)
