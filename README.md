# Applied Machine Learning Project: CAMELS MVP

This repo contains a minimal, working prototype for the course project:

> predict a hydrodynamical CAMELS map from a matched N-body map, using simulation parameters as conditioning inputs.

The implementation is intentionally small and practical. It covers the emulator part of the proposal and leaves the Information-Ordered Bottleneck analysis out of scope for now.

For a fresh-machine setup, see [TEAMMATE_SETUP.md](TEAMMATE_SETUP.md).

## What This Prototype Does

The current prototype trains a small parameter-conditioned CNN/U-Net to learn:

- **Input:** N-body total matter map (`Mtot`)
- **Target:** hydrodynamical gas mass map (`Mgas`)
- **Conditioning:** 6 CAMELS parameters

It uses a real subset of the CAMELS Multifield Dataset, trains end to end, evaluates on a grouped held-out split, and supports standalone prediction from a saved checkpoint.

## What Is And Is Not Stored In Git

This repo tracks:

- code
- documentation
- dependency list
- a script to download the small CAMELS subset

This repo does **not** track:

- `.venv/`
- downloaded `data/`
- generated `artifacts/`

Each teammate is expected to create those locally.

## Dataset Choice

The prototype uses a small `CV + EX` subset from the CAMELS Multifield Dataset instead of the much larger `LH` set, so it stays runnable on a normal laptop.

Files expected in `data/raw/`:

- `Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy`
- `Maps_Mtot_Nbody_IllustrisTNG_EX_z=0.00.npy`
- `params_CV_Nbody_IllustrisTNG.txt`
- `params_EX_Nbody_IllustrisTNG.txt`
- `Maps_Mgas_IllustrisTNG_CV_z=0.00.npy`
- `Maps_Mgas_IllustrisTNG_EX_z=0.00.npy`

Download helper:

```bash
.venv/bin/python scripts/download_camels_subset.py
```

## High-Level Pipeline

1. Load the CAMELS `.npy` map files and parameter text files
2. Match each map to its 6-D parameter row using the CMD `map_number // 15` rule
3. Transform maps into `log10(x + 1)` space
4. Downsample maps from `256 x 256` to `64 x 64`
5. Split by **simulation group**, not by individual map
6. Train a small parameter-conditioned U-Net
7. Evaluate on a held-out test split
8. Save metrics, plots, predictions, and a checkpoint

## Repository Layout

```text
camels_mvp/
  data.py          data loading, normalization, grouped splits
  model.py         conditioned U-Net baseline
  train.py         training, evaluation, artifact export
  predict.py       inference with a saved checkpoint

scripts/
  download_camels_subset.py   helper script to download the small CAMELS subset
```

## Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `scikit-learn`
- `matplotlib`
- `torch`

## Training

Verified training command:

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

## Prediction

After training, run:

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

Outputs:

- `artifacts/prediction.npy`
- `artifacts/prediction.png`

## Fastest Local Validation

If someone just wants to confirm the pipeline works on a fresh machine, use a 1-epoch run:

```bash
.venv/bin/python -m camels_mvp.train \
  --raw-dir data/raw \
  --output-dir artifacts/quick_test \
  --epochs 1 \
  --batch-size 8 \
  --image-size 64 \
  --device cpu
```

That should generate a checkpoint plus metrics and confirm the full path is wired correctly.

## Runtime And Storage

Measured locally for this prototype:

- one `predict` run: about **6 seconds**
- one training epoch: about **18 seconds**
- full 10-epoch run: about **30 to 40 seconds**

Approximate local storage:

- `.venv/`: about **591 MB**
- `data/`: about **255 MB**
- generated `artifacts/run/`: about **5.7 MB**

This prototype does **not** require a GPU.

## Evaluation

Metrics are reported in log-space:

- `MAE`
- `MSE`
- `RMSE`
- Pearson correlation
- `R²`

Verified test metrics from the local run:

- `RMSE(log10) = 0.0931`
- `MAE(log10) = 0.0579`
- `Pearson r = 0.9809`
- `R² = 0.9619`

These are written to `artifacts/run/metrics.json`.

## Important Assumptions

- The prototype uses `Mtot -> Mgas` only.
- It uses `IllustrisTNG` as the hydro suite.
- It uses `CV + EX` instead of `LH` to keep runtime and download size small.
- Metrics are reported in `log10(field + 1)` space.
- The split is grouped by simulation.

## Limitations

This is still an MVP. It does **not** include:

- Information-Ordered Bottleneck experiments
- intrinsic-dimensionality estimation
- large-scale `LH` training
- multi-target benchmarking
- hyperparameter search
- uncertainty estimation

## Suggested Next Steps

1. Train on the `LH` set for a more realistic dataset size
2. Compare multiple hydro targets such as `T`, `P`, or `HI`
3. Add an autoencoder + IOB bottleneck experiment for the intrinsic-dimensionality part

