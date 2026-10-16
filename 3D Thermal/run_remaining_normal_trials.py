from train_fno import train_3d_fno
from train_gnot import train_3d_gnot


FNO_TASKS = (
    ("decrease", range(3, 6)),
    ("linear", range(1, 6)),
)
GNOT_TASKS = (
    ("increase", range(2, 6)),
    ("decrease", range(1, 6)),
    ("linear", range(1, 6)),
)


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
        lr=1e-3,
        use_grad_loss=True,
        log_csv_path=f"{model_dir}/gnot_{mode}_{trial}.csv",
    )


def main() -> None:
    for mode, trials in FNO_TASKS:
        for trial in trials:
            print(f"\n=== 3D Thermal FNO | mode={mode} | trial={trial} ===", flush=True)
            run_fno(mode, trial)
    for mode, trials in GNOT_TASKS:
        for trial in trials:
            print(f"\n=== 3D Thermal GNOT | mode={mode} | trial={trial} ===", flush=True)
            run_gnot(mode, trial)


if __name__ == "__main__":
    main()
