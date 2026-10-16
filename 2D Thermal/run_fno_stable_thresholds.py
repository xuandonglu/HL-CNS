import csv
from pathlib import Path
from statistics import mean, stdev

from eval_fno_stable import eval_fno_stable
from train_fno_stable import train_2d_fno_stable


MODES = ("linear", "increase", "decrease")
TRIALS = range(1, 6)
THRESHOLDS = {"linear": 0.05, "increase": 0.10, "decrease": 0.10}


def train_one(mode: str, trial: int) -> None:
    model_dir = f"model_{mode}/fno_stable"
    train_2d_fno_stable(
        train_pt=f"data/training_data_N32_{mode}.pt",
        val_pt=f"data/validation_data_N32_{mode}.pt",
        model_ckpt=f"{model_dir}/ckpt_2d_fno_stable_{mode}_{trial}.pt",
        best_ckpt=f"{model_dir}/best_2d_fno_stable_{mode}_{trial}.pt",
        num_epochs=100,
        batch_size=10,
        lr=5e-4,
        use_grad_loss=True,
        grad_weight=0.02,
        seed=2000 + 100 * MODES.index(mode) + trial,
        log_csv_path=f"{model_dir}/fno_stable_{mode}_{trial}.csv",
    )


def summarize_training(mode: str) -> tuple[float, float, list[dict]]:
    rows = []
    for trial in TRIALS:
        path = Path(f"model_{mode}/fno_stable/fno_stable_{mode}_{trial}.csv")
        with path.open(newline="") as f:
            log_rows = list(csv.DictReader(f))
        best = min(log_rows, key=lambda row: float(row["Val(RelL2)"]))
        rows.append(
            {
                "mode": mode,
                "trial": trial,
                "best_epoch": int(best["Epoch"]) + 1,
                "best_val_mse": float(best["Val(MSE)"]),
                "best_val_rel": float(best["Val(RelL2)"]),
            }
        )
    rels = [row["best_val_rel"] for row in rows]
    return mean(rels), stdev(rels), rows


def eval_mode(mode: str) -> list[dict]:
    rows = []
    for trial in TRIALS:
        print(f"\n=== Eval 2D Thermal FNO Stable | mode={mode} | trial={trial} ===", flush=True)
        metrics = eval_fno_stable(
            tra_pt=f"data/training_data_N32_{mode}.pt",
            val_pt=f"data/validation_data_N32_{mode}.pt",
            ood_pt=f"data/dualforce_data_N32_{mode}.pt",
            best_model_path=f"model_{mode}/fno_stable/best_2d_fno_stable_{mode}_{trial}.pt",
            batch_size=10,
            export_mat=False,
        )
        rows.append(
            {
                "mode": mode,
                "trial": trial,
                "train_mse": metrics["train"][0],
                "train_rel": metrics["train"][1],
                "val_mse": metrics["val"][0],
                "val_rel": metrics["val"][1],
                "ood_mse": metrics["ood"][0],
                "ood_rel": metrics["ood"][1],
            }
        )
    return rows


def write_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_train_rows = []
    all_eval_rows = []
    for mode in MODES:
        for trial in TRIALS:
            print(f"\n=== 2D Thermal FNO Stable | mode={mode} | trial={trial} ===", flush=True)
            train_one(mode, trial)
        rel_mean, rel_std, train_rows = summarize_training(mode)
        all_train_rows.extend(train_rows)
        print(
            f"\n>>> Training summary {mode}: best Val Rel mean={rel_mean:.6e}, "
            f"std={rel_std:.6e}, threshold={THRESHOLDS[mode]:.6e}",
            flush=True,
        )
        all_eval_rows.extend(eval_mode(mode))
        eval_rels = [row["val_rel"] for row in all_eval_rows if row["mode"] == mode]
        print(
            f"\n>>> Eval summary {mode}: Val Rel mean={mean(eval_rels):.6e}, "
            f"std={stdev(eval_rels):.6e}, threshold={THRESHOLDS[mode]:.6e}",
            flush=True,
        )

    write_rows("run_logs/summary_2d_fno_stable_training.csv", all_train_rows)
    write_rows("run_logs/summary_2d_fno_stable_eval.csv", all_eval_rows)


if __name__ == "__main__":
    main()
