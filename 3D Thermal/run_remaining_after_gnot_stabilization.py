from train_fno import train_3d_fno
from train_gnot import train_3d_gnot


FNO_MODES = ("decrease", "linear")
GNOT_MODES = ("increase", "decrease", "linear")
TRIALS = range(1, 6)


def run_fno(mode: str, trial: int) -> None:
    model_dir = f"model_{mode}/fno"
    train_3d_fno(
        train_pt=f"data/training_data_N32_{mode}.pt",
        val_pt=f"data/validation_data_N32_{mode}.pt",
        ood_pt=f"data/dualforce_data_N32_{mode}.pt",
        model_ckpt=f"{model_dir}/ckpt_fno_{mode}_{trial}.pt",
        best_ckpt=f"{model_dir}/best_fno_{mode}_{trial}.pt",
        num_epochs=100,
        batch_size=10,
        lr=1e-3,
        use_grad_loss=True,
        log_csv_path=f"{model_dir}/fno_{mode}_{trial}.csv",
    )


def run_gnot(mode: str, trial: int) -> None:
    model_dir = f"model_{mode}/gnot"
    train_3d_gnot(
        train_pt=f"data/training_data_N32_{mode}.pt",
        val_pt=f"data/validation_data_N32_{mode}.pt",
        ood_pt=f"data/dualforce_data_N32_{mode}.pt",
        model_ckpt=f"{model_dir}/ckpt_gnot_{mode}_{trial}.pt",
        best_ckpt=f"{model_dir}/best_gnot_{mode}_{trial}.pt",
        num_epochs=100,
        batch_size=10,
        lr=5e-4,
        use_grad_loss=True,
        grad_weight=0.02,
        seed=3000 + 100 * GNOT_MODES.index(mode) + trial,
        log_csv_path=f"{model_dir}/gnot_{mode}_{trial}.csv",
    )


def main() -> None:
    for mode in FNO_MODES:
        for trial in TRIALS:
            print(f"\n=== 3D Thermal FNO | mode={mode} | trial={trial} ===", flush=True)
            run_fno(mode, trial)
    for mode in GNOT_MODES:
        for trial in TRIALS:
            print(f"\n=== 3D Thermal GNOT Stable | mode={mode} | trial={trial} ===", flush=True)
            run_gnot(mode, trial)


if __name__ == "__main__":
    main()
