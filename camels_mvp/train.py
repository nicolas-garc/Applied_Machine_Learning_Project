from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "artifacts/.mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

from camels_mvp.data import (
    CAMELSMapDataset,
    build_bundle,
    compute_stats,
    denormalize_input,
    denormalize_target,
    normalize_bundle,
    split_by_group,
    summarize_split,
)
from camels_mvp.model import ConditionedUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal CAMELS N-body to hydro emulator.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--target-field", default="Mgas")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset: CAMELSMapDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        params = batch["params"].to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs, params)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect: bool = False,
) -> Tuple[float, Dict[str, np.ndarray]]:
    model.eval()
    losses: List[float] = []
    collected: Dict[str, List[np.ndarray]] = {
        "inputs": [],
        "targets": [],
        "predictions": [],
        "sample_ids": [],
    }

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        params = batch["params"].to(device)

        predictions = model(inputs, params)
        loss = criterion(predictions, targets)
        losses.append(float(loss.item()))

        if collect:
            collected["inputs"].append(inputs.cpu().numpy())
            collected["targets"].append(targets.cpu().numpy())
            collected["predictions"].append(predictions.cpu().numpy())
            collected["sample_ids"].append(batch["sample_id"].cpu().numpy())

    payload = {}
    if collect:
        payload = {
            key: np.concatenate(value, axis=0) if value else np.array([])
            for key, value in collected.items()
        }
    return float(np.mean(losses)), payload


def compute_metrics(
    inputs_norm: np.ndarray,
    targets_norm: np.ndarray,
    predictions_norm: np.ndarray,
    stats,
) -> Dict[str, float]:
    inputs_log = denormalize_input(inputs_norm.squeeze(1), stats)
    targets_log = denormalize_target(targets_norm.squeeze(1), stats)
    predictions_log = denormalize_target(predictions_norm.squeeze(1), stats)

    y_true = targets_log.reshape(-1)
    y_pred = predictions_log.reshape(-1)

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(math.sqrt(mse))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mse_log10": mse,
        "rmse_log10": rmse,
        "mae_log10": mae,
        "pearson_r_log10": pearson,
        "r2_log10": r2,
        "input_log10_mean": float(inputs_log.mean()),
        "target_log10_mean": float(targets_log.mean()),
        "prediction_log10_mean": float(predictions_log.mean()),
    }


def save_history_plot(output_dir: Path, history: List[Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close()


def save_prediction_plot(
    output_dir: Path,
    eval_payload: Dict[str, np.ndarray],
    stats,
    max_rows: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = denormalize_input(eval_payload["inputs"].squeeze(1), stats)
    targets = denormalize_target(eval_payload["targets"].squeeze(1), stats)
    predictions = denormalize_target(eval_payload["predictions"].squeeze(1), stats)
    errors = np.abs(predictions - targets)

    row_count = min(max_rows, inputs.shape[0])
    fig, axes = plt.subplots(row_count, 4, figsize=(12, 3 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(row_count):
        panels = [
            (inputs[row], "Input Mtot"),
            (targets[row], "Target Mgas"),
            (predictions[row], "Prediction"),
            (errors[row], "Abs. error"),
        ]
        for column, (image, title) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(image, cmap="magma")
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_grid.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    bundle = build_bundle(args.raw_dir, target_field=args.target_field, image_size=args.image_size)
    splits = split_by_group(bundle, seed=args.seed)
    stats = compute_stats(bundle, splits["train"])
    normalized_bundle = normalize_bundle(bundle, stats)

    train_dataset = CAMELSMapDataset(
        normalized_bundle.inputs, normalized_bundle.targets, normalized_bundle.params, splits["train"]
    )
    val_dataset = CAMELSMapDataset(
        normalized_bundle.inputs, normalized_bundle.targets, normalized_bundle.params, splits["val"]
    )
    test_dataset = CAMELSMapDataset(
        normalized_bundle.inputs, normalized_bundle.targets, normalized_bundle.params, splits["test"]
    )

    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConditionedUNet(param_dim=normalized_bundle.params.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _ = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "target_field": args.target_field,
                        "image_size": args.image_size,
                        "param_dim": int(normalized_bundle.params.shape[1]),
                        "base_channels": 16,
                    },
                    "normalization": stats.to_dict(),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, eval_payload = evaluate(model, test_loader, criterion, device, collect=True)
    metrics = compute_metrics(
        inputs_norm=eval_payload["inputs"],
        targets_norm=eval_payload["targets"],
        predictions_norm=eval_payload["predictions"],
        stats=stats,
    )
    metrics["test_l1_normalized"] = test_loss

    split_summary = {
        split_name: summarize_split(bundle, sample_ids) for split_name, sample_ids in splits.items()
    }
    summary = {
        "device": str(device),
        "config": vars(args),
        "splits": split_summary,
        "metrics": metrics,
    }

    save_history_plot(args.output_dir, history)
    save_prediction_plot(args.output_dir, eval_payload, stats)

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=eval_payload["sample_ids"],
        inputs=eval_payload["inputs"],
        targets=eval_payload["targets"],
        predictions=eval_payload["predictions"],
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
