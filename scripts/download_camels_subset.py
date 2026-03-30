from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FILES = {
    "Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy",
    "Maps_Mtot_Nbody_IllustrisTNG_EX_z=0.00.npy": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/Maps_Mtot_Nbody_IllustrisTNG_EX_z=0.00.npy",
    "params_CV_Nbody_IllustrisTNG.txt": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/params_CV_Nbody_IllustrisTNG.txt",
    "params_EX_Nbody_IllustrisTNG.txt": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/params_EX_Nbody_IllustrisTNG.txt",
    "Maps_Mgas_IllustrisTNG_CV_z=0.00.npy": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG/Maps_Mgas_IllustrisTNG_CV_z=0.00.npy",
    "Maps_Mgas_IllustrisTNG_EX_z=0.00.npy": "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG/Maps_Mgas_IllustrisTNG_EX_z=0.00.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the small CAMELS subset used by the MVP.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        destination = args.output_dir / name
        if destination.exists():
            print(f"skip {name}")
            continue
        print(f"downloading {name}")
        urlretrieve(url, destination)


if __name__ == "__main__":
    main()

