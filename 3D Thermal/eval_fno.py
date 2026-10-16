import torch
from torch.utils.data import DataLoader

from benchmark_utils import benchmark_tag_from_path, export_mat_batch, print_checkpoint_load, print_metrics, print_param_count, resolve_existing_path
from train_fno import FNOThermal3D
from train_hlcns import PackedTensorDataset3D, create_bc_mask_and_value_like_3d


@torch.no_grad()
def eval_fno(
    tra_pt: str,
    val_pt: str,
    ood_pt: str,
    best_model_path: str,
    batch_size: int = 10,
    device: str | None = None,
    export_mat: bool = False,
    mat_output_dir: str = "mat_output",
    mat_tag: str | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    best_model_path = resolve_existing_path(best_model_path)
    mat_tag = mat_tag or benchmark_tag_from_path(best_model_path)
    train_loader = DataLoader(PackedTensorDataset3D(tra_pt), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(PackedTensorDataset3D(val_pt), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(PackedTensorDataset3D(ood_pt), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = FNOThermal3D().to(device)
    state = torch.load(best_model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        print_checkpoint_load(best_model_path, True)
    else:
        model.load_state_dict(state)
        print_checkpoint_load(best_model_path, False)
    model.eval()
    print_param_count(sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))

    def run_split(name, loader):
        mse_num = 0.0
        mse_den = 0
        rel_num2 = 0.0
        rel_den2 = 0.0
        for batch_idx, (f, u) in enumerate(loader):
            f = f.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            bc_mask, bc_val = create_bc_mask_and_value_like_3d(f)
            pred = model(f, bc_mask, bc_val)
            if name == "Val":
                export_mat_batch(export_mat, mat_output_dir, "fno", mat_tag, "val", batch_idx, {"f": f, "pred": pred, "exact": u})
            interior = bc_mask < 0.5
            diff2 = (pred - u).pow(2)
            mse_num += diff2[interior].sum().item()
            mse_den += int(interior.sum().item())
            rel_num2 += diff2[interior].sum().item()
            rel_den2 += u.pow(2)[interior].sum().item()
        print_metrics(name, mse_num / max(1, mse_den), (rel_num2 ** 0.5) / ((rel_den2 ** 0.5) + 1e-12))

    run_split("Train", train_loader)
    run_split("Val", val_loader)
    run_split("OOD", ood_loader)


if __name__ == "__main__":
    mode = "linear"
    for i in range(1, 6):
        eval_fno(
            tra_pt=f"data/training_data_N32_{mode}.pt",
            val_pt=f"data/validation_data_N32_{mode}.pt",
            ood_pt=f"data/dualforce_data_N32_{mode}.pt",
            best_model_path=f"model_{mode}/fno/best_fno_{mode}_{i}.pt",
            batch_size=10,
            export_mat=False,
        )
