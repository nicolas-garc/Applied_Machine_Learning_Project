from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

MAPS_PER_SIMULATION = 15
SUPPORTED_SETS = ("CV", "EX")


@dataclass
class DataBundle:
    inputs: np.ndarray
    targets: np.ndarray
    params: np.ndarray
    group_ids: np.ndarray
    set_names: np.ndarray
    map_indices: np.ndarray


@dataclass
class NormalizationStats:
    input_mean: float
    input_std: float
    target_mean: float
    target_std: float
    param_mean: np.ndarray
    param_std: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_mean": float(self.input_mean),
            "input_std": float(self.input_std),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "param_mean": self.param_mean.tolist(),
            "param_std": self.param_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NormalizationStats":
        return cls(
            input_mean=float(payload["input_mean"]),
            input_std=float(payload["input_std"]),
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
            param_mean=np.asarray(payload["param_mean"], dtype=np.float32),
            param_std=np.asarray(payload["param_std"], dtype=np.float32),
        )


class CAMELSMapDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        params: np.ndarray,
        sample_ids: np.ndarray,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.params = params
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_id = int(self.sample_ids[index])
        return {
            "inputs": torch.from_numpy(self.inputs[sample_id][None, ...]),
            "targets": torch.from_numpy(self.targets[sample_id][None, ...]),
            "params": torch.from_numpy(self.params[sample_id]),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
        }


def build_bundle(raw_dir: Path, target_field: str = "Mgas", image_size: int = 64) -> DataBundle:
    input_maps: List[np.ndarray] = []
    target_maps: List[np.ndarray] = []
    param_rows: List[np.ndarray] = []
    group_ids: List[np.ndarray] = []
    set_names: List[np.ndarray] = []
    map_indices: List[np.ndarray] = []

    group_offset = 0
    map_offset = 0
    for set_name in SUPPORTED_SETS:
        input_path = raw_dir / f"Maps_Mtot_Nbody_IllustrisTNG_{set_name}_z=0.00.npy"
        target_path = raw_dir / f"Maps_{target_field}_IllustrisTNG_{set_name}_z=0.00.npy"
        params_path = raw_dir / f"params_{set_name}_Nbody_IllustrisTNG.txt"

        inputs = np.load(input_path).astype(np.float32)
        targets = np.load(target_path).astype(np.float32)
        params = np.loadtxt(params_path).astype(np.float32)
        if params.ndim == 1:
            params = params.reshape(1, -1)

        expected_maps = params.shape[0] * MAPS_PER_SIMULATION
        if inputs.shape[0] != expected_maps or targets.shape[0] != expected_maps:
            raise ValueError(
                f"Unexpected map count for {set_name}: "
                f"{inputs.shape[0]=}, {targets.shape[0]=}, {expected_maps=}"
            )

        inputs = np.log10(inputs + 1.0, dtype=np.float32)
        targets = np.log10(targets + 1.0, dtype=np.float32)
        inputs = downsample_maps(inputs, image_size)
        targets = downsample_maps(targets, image_size)

        input_maps.append(inputs)
        target_maps.append(targets)
        param_rows.append(np.repeat(params, MAPS_PER_SIMULATION, axis=0))
        group_ids.append(np.repeat(np.arange(group_offset, group_offset + params.shape[0]), MAPS_PER_SIMULATION))
        set_names.append(np.full(expected_maps, set_name))
        map_indices.append(np.arange(map_offset, map_offset + expected_maps))

        group_offset += params.shape[0]
        map_offset += expected_maps

    return DataBundle(
        inputs=np.concatenate(input_maps, axis=0),
        targets=np.concatenate(target_maps, axis=0),
        params=np.concatenate(param_rows, axis=0),
        group_ids=np.concatenate(group_ids, axis=0),
        set_names=np.concatenate(set_names, axis=0),
        map_indices=np.concatenate(map_indices, axis=0),
    )


def downsample_maps(maps: np.ndarray, image_size: int) -> np.ndarray:
    current_size = maps.shape[-1]
    if current_size == image_size:
        return maps
    if current_size % image_size != 0:
        raise ValueError(f"Cannot downsample from {current_size} to {image_size}")

    factor = current_size // image_size
    return (
        maps.reshape(maps.shape[0], image_size, factor, image_size, factor)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )


def split_by_group(bundle: DataBundle, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    split_groups: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    for set_name in SUPPORTED_SETS:
        groups = np.unique(bundle.group_ids[bundle.set_names == set_name])
        groups = np.array(groups, copy=True)
        rng.shuffle(groups)

        if set_name == "CV":
            n_train, n_val = 20, 3
        elif set_name == "EX":
            n_train, n_val = 2, 1
        else:
            n_train = max(1, int(len(groups) * 0.7))
            n_val = max(1, int(len(groups) * 0.15))

        split_groups["train"].extend(groups[:n_train].tolist())
        split_groups["val"].extend(groups[n_train : n_train + n_val].tolist())
        split_groups["test"].extend(groups[n_train + n_val :].tolist())

    sample_splits = {}
    for split_name, groups in split_groups.items():
        mask = np.isin(bundle.group_ids, np.asarray(groups))
        sample_splits[split_name] = np.nonzero(mask)[0]

    return sample_splits


def compute_stats(bundle: DataBundle, train_ids: np.ndarray) -> NormalizationStats:
    input_train = bundle.inputs[train_ids]
    target_train = bundle.targets[train_ids]
    params_train = bundle.params[train_ids]

    input_std = float(np.maximum(input_train.std(), 1e-6))
    target_std = float(np.maximum(target_train.std(), 1e-6))
    param_std = params_train.std(axis=0)
    param_std[param_std < 1e-6] = 1.0

    return NormalizationStats(
        input_mean=float(input_train.mean()),
        input_std=input_std,
        target_mean=float(target_train.mean()),
        target_std=target_std,
        param_mean=params_train.mean(axis=0).astype(np.float32),
        param_std=param_std.astype(np.float32),
    )


def normalize_bundle(bundle: DataBundle, stats: NormalizationStats) -> DataBundle:
    return DataBundle(
        inputs=((bundle.inputs - stats.input_mean) / stats.input_std).astype(np.float32),
        targets=((bundle.targets - stats.target_mean) / stats.target_std).astype(np.float32),
        params=((bundle.params - stats.param_mean) / stats.param_std).astype(np.float32),
        group_ids=bundle.group_ids.copy(),
        set_names=bundle.set_names.copy(),
        map_indices=bundle.map_indices.copy(),
    )


def denormalize_target(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return values * stats.target_std + stats.target_mean


def denormalize_input(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return values * stats.input_std + stats.input_mean


def summarize_split(bundle: DataBundle, sample_ids: np.ndarray) -> Dict[str, object]:
    set_values, set_counts = np.unique(bundle.set_names[sample_ids], return_counts=True)
    group_count = int(np.unique(bundle.group_ids[sample_ids]).shape[0])
    return {
        "samples": int(sample_ids.shape[0]),
        "groups": group_count,
        "sets": {name: int(count) for name, count in zip(set_values.tolist(), set_counts.tolist())},
    }

