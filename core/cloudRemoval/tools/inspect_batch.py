from __future__ import annotations

import argparse

from cloud_removal.config import apply_overrides, load_config
from cloud_removal.data import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one dataset sample")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    dataset = build_dataset(config, args.split)
    sample = dataset[args.index]
    print(f"dataset length: {len(dataset)}")
    print(f"sample id: {sample['sample_id']}")
    for key in ["input", "target", "observed_mask", "missing_mask"]:
        tensor = sample[key]
        print(
            f"{key:>13}: shape={tuple(tensor.shape)}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
            f"mean={tensor.mean().item():.6f}"
        )
    print("paths:")
    for path in sample["paths"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

