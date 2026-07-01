from __future__ import annotations

import argparse

from cloud_removal.config import apply_overrides, load_config
from cloud_removal.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refactored ConvLSTM cloud-removal model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set training.epochs=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    train(config)


if __name__ == "__main__":
    main()
