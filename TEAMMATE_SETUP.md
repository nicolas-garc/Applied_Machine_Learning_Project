# Teammate Setup And Run Guide

These are the exact steps a teammate should follow on a fresh machine.

## 1. Prerequisites

Make sure you have:

- Python `3.9+`
- Git
- internet access to download the CAMELS subset
- about `1 GB` of free disk space for the environment, data, and generated outputs

## 2. Clone The Repo

```bash
git clone <repo-url>
cd Applied_Machine_Learning_Project
```

If you already have the repo:

```bash
git checkout main
git pull
```

## 3. Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not want to activate it, you can use `.venv/bin/python` directly in every command instead.

## 4. Download The CAMELS Subset

Run:

```bash
.venv/bin/python scripts/download_camels_subset.py
```

This downloads the small `CV + EX` subset into `data/raw/`.

## 5. Quick Sanity Check

Run a 1-epoch training job:

```bash
.venv/bin/python -m camels_mvp.train \
  --raw-dir data/raw \
  --output-dir artifacts/quick_test \
  --epochs 1 \
  --batch-size 8 \
  --image-size 64 \
  --device cpu
```

If this finishes successfully, the project is set up correctly.

You should see files such as:

- `artifacts/quick_test/best_model.pt`
- `artifacts/quick_test/metrics.json`
- `artifacts/quick_test/training_curve.png`
- `artifacts/quick_test/prediction_grid.png`

## 6. Full Local Run

Run the verified configuration:

```bash
.venv/bin/python -m camels_mvp.train \
  --raw-dir data/raw \
  --output-dir artifacts/run \
  --epochs 10 \
  --batch-size 8 \
  --image-size 64 \
  --device cpu
```

Expected outputs:

- `artifacts/run/best_model.pt`
- `artifacts/run/metrics.json`
- `artifacts/run/history.json`
- `artifacts/run/training_curve.png`
- `artifacts/run/prediction_grid.png`
- `artifacts/run/test_predictions.npz`

## 7. Run Prediction

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

Expected outputs:

- `artifacts/prediction.npy`
- `artifacts/prediction.png`

## 8. What To Look At

The most useful files are:

- `artifacts/run/metrics.json`
- `artifacts/run/training_curve.png`
- `artifacts/run/prediction_grid.png`
- `artifacts/prediction.png`

## 9. Expected Runtime

Approximate local timings for this prototype:

- prediction: about `6 seconds`
- 1 training epoch: about `18 seconds`
- full 10-epoch run: about `30 to 40 seconds`

## 10. Common Issues

If `data/raw/` is missing:

- run the downloader again

If `best_model.pt` is missing:

- you need to train first

If the machine has no GPU:

- that is fine, this version is designed to run on CPU

If the install fails:

- confirm you are using Python `3.9+`
- recreate the virtual environment and reinstall requirements

