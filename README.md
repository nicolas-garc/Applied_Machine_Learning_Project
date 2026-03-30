# CAMELS MVP Prototype

> Predict a hydrodynamical CAMELS map from a matched N-body map, using simulation parameters as conditioning inputs.

The implementation is intentionally small and practical. It focuses on the emulator part of the proposal and leaves the Information-Ordered Bottleneck analysis out of scope for now.

## What This Prototype Does

The current prototype trains a small CNN/U-Net baseline to learn:

- **Input:** N-body total matter map (`Mtot`)
- **Target:** hydrodynamical gas mass map (`Mgas`)
- **Conditioning:** 6 CAMELS parameters

It uses a **real subset** of the CAMELS Multifield Dataset, trains end to end, evaluates on a held-out grouped test split, and supports standalone prediction from a saved checkpoint.

## Why This Scope

The proposal mentioned two separate goals:

1. An emulator for `N-body -> hydro`
2. An intrinsic-dimensionality study using an Information-Ordered Bottleneck

For a course MVP, the smallest defensible prototype is the emulator. That is what this repo implements.

## Dataset Choice

The proposal linked the CAMELS Multifield Dataset documentation:

- CAMELS CMD docs: <https://camels.readthedocs.io/en/latest/CMD.html>

From the CMD documentation and data layout, the relevant facts for this prototype are:

- 2D maps are stored as `.npy` arrays
- each map is `256 x 256`
- matched N-body/hydro data exists
- maps are aligned by index within a simulation
- each simulation contributes **15 maps**
- parameter files are stored separately

To keep the prototype fast and local, this repo uses a small `CV + EX` subset instead of the much larger `LH` set.

The files used are:

- `Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy`
- `Maps_Mtot_Nbody_IllustrisTNG_EX_z=0.00.npy`
- `params_CV_Nbody_IllustrisTNG.txt`
- `params_EX_Nbody_IllustrisTNG.txt`
- `Maps_Mgas_IllustrisTNG_CV_z=0.00.npy`
- `Maps_Mgas_IllustrisTNG_EX_z=0.00.npy`

## High-Level Pipeline

The repo follows this flow:

1. Load the real CAMELS `.npy` map files and parameter text files
2. Match each map to its 6-D parameter row using the CMD rule `map_number // 15`
3. Transform the maps into `log10(x + 1)` space
4. Downsample maps from `256 x 256` to `64 x 64` for speed
5. Split data by **simulation group**, not by individual map
6. Train a small parameter-conditioned U-Net
7. Evaluate on a held-out test split
8. Save metrics, plots, predictions, and the trained checkpoint

## How The Data Loader Works

The data code lives in `camels_mvp/data.py`.

Key choices:

- It reads `CV` and `EX` only.
- It expects each parameter row to correspond to 15 maps.
- It repeats each parameter row 15 times so every map has its own conditioning vector.
- It assigns a **group id per simulation** so train/val/test splits do not leak maps from the same simulation across splits.

This is important because random per-map splitting would overestimate performance.

## Preprocessing

The preprocessing is simple:

- `log10(x + 1)` on both inputs and targets
- average-pooling style downsampling from `256` to `64`
- standardization using train-split mean/std only

Why:

- CAMELS field values span a large dynamic range
- the log transform stabilizes training
- downsampling makes training fast on CPU

## Model

The model lives in `camels_mvp/model.py`.

It is a compact parameter-conditioned U-Net:

- 1 input channel
- 1 output channel
- skip connections
- 6-D parameter vector projected into the bottleneck
- circular padding in convolutions to better match the periodic structure of simulation maps

This is much smaller than a research-grade model, but it is enough to demonstrate the idea end to end.

## Training

The training script lives in `camels_mvp/train.py`.

It does the following:

- builds the dataset bundle
- creates grouped train/val/test splits
- computes normalization stats from the training split
- trains with Adam and L1 loss
- saves the best checkpoint by validation loss
- evaluates on the test split
- writes plots and JSON metrics

The main output checkpoint is:

- `artifacts/run/best_model.pt`

## Evaluation

The prototype reports metrics in normalized log-space / denormalized log-space:

- `MAE`
- `MSE`
- `RMSE`
- Pearson correlation
- `R²`

For the verified run in this repo, the held-out test metrics were:

- `RMSE(log10) = 0.0931`
- `MAE(log10) = 0.0579`
- `Pearson r = 0.9809`
- `R² = 0.9619`

These numbers are saved in:

- `artifacts/run/metrics.json`

## Prediction

The prediction script lives in `camels_mvp/predict.py`.

It:

- loads the saved checkpoint
- loads one input `.npy` map or one map from a stack
- normalizes the input map and parameter vector
- runs the model
- writes the predicted field to disk
- saves a small preview plot

This is the fastest way to test that the repo works.

## Repository Layout

```text
camels_mvp/
  data.py          data loading, normalization, grouped splits
  model.py         conditioned U-Net baseline
  train.py         training, evaluation, artifact export
  predict.py       inference with a saved checkpoint

scripts/
  download_camels_subset.py   helper script to download the small CAMELS subset

data/
  raw/             downloaded CAMELS subset

artifacts/
  run/             trained checkpoint, metrics, plots, saved predictions

tools/
  extract_pdf.jxa  helper used during proposal/PDF extraction
```

## Environment Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies are intentionally minimal:

- `numpy`
- `scikit-learn`
- `matplotlib`
- `torch`

## Download The Data

If `data/raw/` is empty, download the small subset with:

```bash
.venv/bin/python scripts/download_camels_subset.py
```

If the files are already present, the script skips them.

## Fastest Smoke Test

If you want the quickest check that everything works, run prediction with the already-trained checkpoint:

```bash
.venv/bin/python -m camels_mvp.predict \
  --checkpoint artifacts/run/best_model.pt \
  --input-map data/raw/Maps_Mtot_Nbody_IllustrisTNG_EX_z=0.00.npy \
  --map-index 0 \
  --params 0.3,0.8,1.0,1.0,1.0,1.0 \
  --output-npy artifacts/prediction.npy \
  --plot-path artifacts/prediction.png \
  --device cpu
```

Expected outputs:

- `artifacts/prediction.npy`
- `artifacts/prediction.png`

## Re-Train The Model

To run the verified training configuration:

```bash
.venv/bin/python -m camels_mvp.train \
  --raw-dir data/raw \
  --output-dir artifacts/run \
  --epochs 10 \
  --batch-size 8 \
  --image-size 64 \
  --device cpu
```

Outputs:

- `artifacts/run/best_model.pt`
- `artifacts/run/metrics.json`
- `artifacts/run/history.json`
- `artifacts/run/training_curve.png`
- `artifacts/run/prediction_grid.png`
- `artifacts/run/test_predictions.npz`

## How Long It Takes

Measured in this repo on local CPU:

- one `predict` run: about **6 seconds**
- one training epoch: about **18 seconds**
- full 10-epoch run: about **30 to 40 seconds**

Approximate storage:

- `.venv/`: about **591 MB**
- `data/`: about **255 MB**
- `artifacts/run/`: about **5.7 MB**

This prototype does **not** require a GPU.

## Interpreting The Outputs

The most useful files after training are:

- `metrics.json`: summary metrics and split sizes
- `training_curve.png`: train vs validation loss
- `prediction_grid.png`: input / target / prediction / error examples
- `best_model.pt`: checkpoint for later inference

If `prediction_grid.png` looks structurally similar to the target maps and the test metrics are close to the verified numbers above, the prototype is behaving as expected.

## Important Assumptions

- The prototype uses `Mtot -> Mgas` only.
- It uses `IllustrisTNG` as the hydro suite.
- It uses `CV + EX` instead of `LH` to keep runtime and download size small.
- Metrics are reported in `log10(field + 1)` space.
- The split is grouped by simulation.

These choices are deliberate and keep the repo aligned with the proposal while staying runnable on a normal laptop.

## Limitations

This is still only an MVP.

What it does **not** do:

- no Information-Ordered Bottleneck implementation
- no intrinsic-dimensionality estimation
- no large-scale CAMELS training on `LH`
- no benchmarking across multiple hydro target fields
- no hyperparameter search
- no uncertainty estimation

## Suggested Next Steps

If you want to extend this toward the full proposal:

1. Train on the `LH` set for a more realistic dataset size
2. Compare multiple hydro targets such as `T`, `P`, or `HI`
3. Add an autoencoder + IOB bottleneck experiment for the intrinsic-dimensionality part

## File References

Core implementation files:

- `camels_mvp/data.py`
- `camels_mvp/model.py`
- `camels_mvp/train.py`
- `camels_mvp/predict.py`
- `scripts/download_camels_subset.py`

Primary outputs:

- `artifacts/run/best_model.pt`
- `artifacts/run/metrics.json`
- `artifacts/run/training_curve.png`
- `artifacts/run/prediction_grid.png`

## Summary

This repo gives you a clean course-project prototype:

- real CAMELS data
- real training loop
- real evaluation
- real prediction command
- small enough to run locally

It is not the full research project, but it is a working, defensible baseline that demonstrates the core idea end to end.
