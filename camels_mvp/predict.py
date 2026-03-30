from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "artifacts/.mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from camels_mvp.data import NormalizationStats, downsample_maps
from camels_mvp.model import ConditionedUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the CAMELS MVP model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--params", required=True, help="Comma-separated 6-value parameter vector.")
    parser.add_argument("--output-npy", type=Path, default=Path("artifacts/prediction.npy"))
    parser.add_argument("--plot-path", type=Path, default=Path("artifacts/prediction.png"))
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_map(input_map: np.ndarray, image_size: int, stats: NormalizationStats) -> np.ndarray:
    if input_map.ndim == 3:
        input_map = input_map[0]
    if input_map.ndim != 2:
        raise ValueError(f"Expected a 2D map or a stack of maps, received shape {input_map.shape}")

    input_map = np.log10(input_map.astype(np.float32) + 1.0)
    input_map = downsample_maps(input_map[None, ...], image_size)[0]
    return ((input_map - stats.input_mean) / stats.input_std).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    device = select_device(args.device)

    model = ConditionedUNet(param_dim=int(config["param_dim"]), base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    maps = np.load(args.input_map)
    input_map = maps[args.map_index] if maps.ndim == 3 else maps
    params = np.asarray([float(value) for value in args.params.split(",")], dtype=np.float32)
    if params.shape[0] != int(config["param_dim"]):
        raise ValueError(f"Expected {config['param_dim']} parameters, received {params.shape[0]}")

    input_norm = preprocess_map(input_map, int(config["image_size"]), stats)
    params_norm = ((params - stats.param_mean) / stats.param_std).astype(np.float32)

    with torch.no_grad():
        prediction_norm = model(
            torch.from_numpy(input_norm[None, None, ...]).to(device),
            torch.from_numpy(params_norm[None, ...]).to(device),
        )

    prediction_log = prediction_norm.cpu().numpy()[0, 0] * stats.target_std + stats.target_mean
    prediction_physical = np.power(10.0, prediction_log, dtype=np.float32) - 1.0
    np.save(args.output_npy, prediction_physical.astype(np.float32))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(input_map.astype(np.float32) + 1.0), cmap="magma")
    plt.title("Input Mtot")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction_log, cmap="magma")
    plt.title("Predicted target")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=160)
    plt.close()

    print(f"Saved prediction array to {args.output_npy}")
    print(f"Saved preview plot to {args.plot_path}")


if __name__ == "__main__":
    main()
