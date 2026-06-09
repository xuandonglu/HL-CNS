# HL-CNS Benchmark Examples

This repository contains the three numerical examples used in the HL-CNS manuscript.

## Case Layout

| Folder | Manuscript case | Tasks |
| --- | --- | --- |
| `2D Thermal/Linear BC` | 2-D diffusion | Constant, amplifying, and attenuating conductivity on `1 x 32 x 32` fields |
| `3D Thermal/linearbc` | 3-D diffusion | Constant, amplifying, and attenuating conductivity on `1 x 32 x 32 x 32` fields |
| `3D elasticity revised` | 3-D elasticity / elasto-plasticity | Linear elasticity and bilinear elasto-plasticity on `3 x 32 x 32 x 32` fields |

Each case keeps the same benchmark model family used in the paper: ResNet, FCN,
U-Net, DeepONet, and HL-CNS.

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
