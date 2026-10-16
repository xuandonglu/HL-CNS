# HL-CNS Benchmark Examples

This repository contains the code for the three numerical examples presented in the HL-CNS article.

Associated paper: [View the article on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231226022927).

## Case Layout

| Folder | Manuscript case | Tasks |
| --- | --- | --- |
| `2D Thermal` | 2-D diffusion | Constant, amplifying, and attenuating conductivity on `1 x 32 x 32` fields |
| `3D Thermal` | 3-D diffusion | Constant, amplifying, and attenuating conductivity on `1 x 32 x 32 x 32` fields |
| `3D elasticity revised` | 3-D elasticity / elasto-plasticity | Linear elasticity and bilinear elasto-plasticity on `3 x 32 x 32 x 32` fields |

Each case keeps the same benchmark model family used in the paper: ResNet, FCN,
U-Net, DeepONet, FNO, GNOT, and HL-CNS. The elasticity folder also contains
HL-CNS ablations and FEA/MFEA comparison scripts.

## Environment

Use Python 3.10 or newer. Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The scripts select CUDA when available and otherwise use the CPU. For GPU runs,
install a PyTorch build compatible with the local CUDA environment. CuPy is
optional for the nonlinear elasticity data generator, which includes a NumPy
fallback.

## Data and Checkpoints

Datasets, trained checkpoints, generated logs, and plotting exports are excluded
by `.gitignore` and are not distributed with the source code. Training and
evaluation require the corresponding local data; evaluation also requires trained
checkpoints.

Run scripts from inside their case folder because paths are relative to the
working directory. For either thermal case, the expected files are:

```text
data/training_data_N32_<mode>.pt
data/validation_data_N32_<mode>.pt
data/dualforce_data_N32_<mode>.pt
```

Here `<mode>` is `linear`, `increase`, or `decrease`. Each file is a tensor
dictionary containing `f` and `u`, with a leading sample dimension followed by
the field dimensions shown in the table above.

For elasticity, use `data_linear_twopeak/` or `data_nonlinear_twopeak/` inside
`3D elasticity revised/`. Samples are individual `.pt` dictionaries with `f`,
`u`, `bc_mask`, and `bc_val` tensors of shape `[3, 32, 32, 32]`. File prefixes
select the splits: `sample_` for training, `validation_` for validation, and
`dualforce_` for out-of-distribution evaluation. In these elasticity files,
`bc_mask = 1` marks a prescribed displacement.

Data generation scripts are provided in each case folder. Check their settings
before running them: the current entries generate selected splits and modes,
rather than automatically producing every dataset. Create the thermal `data/`
directory before generating thermal samples. Generator mode names `increasing`
and `decreasing` correspond to training filename suffixes `increase` and
`decrease`. For elasticity, check `output_dir` and the saved filename prefix
when generating each split.

## Training and Evaluation

For example, from the repository root:

```bash
cd "2D Thermal"
python train_fno.py
python eval_fno.py
```

For other models, use their corresponding `train_<model>.py` and
`eval_<model>.py` files. Check the `if __name__ == "__main__"` block before
running: mode, trial range, paths, hierarchy depth, and cycle count retain the
settings of the individual experiments and are not synchronized across scripts.
Match evaluation checkpoint names and model settings to the training run.

`run_fno_gnot_all_modes.py` in each thermal folder schedules five trials for
each of the three modes. Other `run_remaining_*.py` and
`run_fno_stable_thresholds.py` scripts retain experiment-specific continuation
or tuning schedules.

In the elasticity folder, `train_hlcns_no_correction.py` and
`train_hlcns_no_relaxation.py` have corresponding evaluation scripts.
`train_mfea.py` trains PsiNet. The standalone single-sample Jacobi benchmarks
are `benchmark_fea_jacobi_single_sample.py` and
`benchmark_mfea_jacobi_single_sample.py`; run either with `--help` for its
options. These scripts use Jacobi updates, whereas `eval_mfea.py` retains a
multigrid evaluation path. `debug_fea_vcycle_one_sample.py` is a diagnostic
multigrid script, and `mfea_kernel_2d_field.py` is the 2-D reference code.

## HL-CNS Naming

The proposed model is named `hlcns` throughout the code:

- `train_hlcns.py` trains the HL-CNS model.
- `eval_hlcns.py` evaluates HL-CNS.
- `eval_hlcns_time.py` reports inference timing when available.
- `model_hlcns.py` contains the shared 3-D HL-CNS implementation for the elasticity example.

HL-CNS checkpoints are written under `model_<mode>/hlcns/`.

## Evaluation Output

Benchmark evaluation scripts print metrics in a consistent format:

```text
Train -- MSE: ... | Rel: ...
Val   -- MSE: ... | Rel: ...
OOD   -- MSE: ... | Rel: ...
```

HL-CNS `.mat` export is disabled by default. To export a representative validation
batch for external plotting, call the relevant `eval_hlcns(...)` function with
`export_mat=True`; the file will be written to `mat_output/`.

Training histories are written to the CSV paths configured in the training
scripts, generally under `model_<mode>/<model>/`. Column layouts vary between
scripts. Benchmark TSV files contain the metrics and timing columns named in
their headers.

The ignore rules apply to Git-based uploads. When uploading through the GitHub
website, select only the source and documentation files; uploading the entire
folder manually does not apply `.gitignore` rules.
