from train_fno import train_2d_fno
from train_gnot import train_2d_gnot


MODES = ("increase", "decrease", "linear")
TRIALS = range(1, 6)


def run_fno(mode: str, trial: int) -> None:
    model_dir = f"model_{mode}/fno"
    train_2d_fno(
        train_pt=f"data/training_data_N32_{mode}.pt",
        val_pt=f"data/validation_data_N32_{mode}.pt",
        model_ckpt=f"{model_dir}/ckpt_2d_fno_{mode}_{trial}.pt",
        best_ckpt=f"{model_dir}/best_2d_fno_{mode}_{trial}.pt",
        num_epochs=100,
        batch_size=10,
        lr=1e-3,
        use_grad_loss=True,
        log_csv_path=f"{model_dir}/fno_{mode}_{trial}.csv",
    )


def run_gnot(mode: str, trial: int) -> None:
    model_dir = f"model_{mode}/gnot"
    train_2d_gnot(
        train_pt=f"data/training_data_N32_{mode}.pt",
        val_pt=f"data/validation_data_N32_{mode}.pt",
        model_ckpt=f"{model_dir}/ckpt_2d_gnot_{mode}_{trial}.pt",
        best_ckpt=f"{model_dir}/best_2d_gnot_{mode}_{trial}.pt",
        num_epochs=100,
        batch_size=10,
        lr=1e-3,
        use_grad_loss=True,
        log_csv_path=f"{model_dir}/gnot_{mode}_{trial}.csv",
    )


def main() -> None:
    for mode in MODES:
        for trial in TRIALS:
            print(f"\n=== 2D Thermal FNO | mode={mode} | trial={trial} ===", flush=True)
            run_fno(mode, trial)
        for trial in TRIALS:
            print(f"\n=== 2D Thermal GNOT | mode={mode} | trial={trial} ===", flush=True)
            run_gnot(mode, trial)


if __name__ == "__main__":
    main()
