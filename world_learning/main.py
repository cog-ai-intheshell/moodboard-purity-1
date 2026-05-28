#!/usr/bin/env python3
"""World-learning command group.

Use this entrypoint for long-running data preparation, training, calibration
and evaluation jobs. The web app should only load produced artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_learning import calibrate, data_factory, evaluate, train_fusion_encoder, train_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="World-learning utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("data", help="Normalize local folders into the common sample schema.")
    subcommands.add_parser("train", help="Build current world-model artifacts.")
    subcommands.add_parser("train-fusion", help="Train the future fusion encoder.")
    subcommands.add_parser("evaluate", help="Evaluate artifacts on held-out splits.")
    subcommands.add_parser("calibrate", help="Calibrate thresholds on D_calib.")
    args, remaining = parser.parse_known_args()

    if args.command == "data":
        data_factory.main_with_args(remaining)
    elif args.command == "train":
        train_world_model.main_with_args(remaining)
    elif args.command == "train-fusion":
        train_fusion_encoder.main_with_args(remaining)
    elif args.command == "evaluate":
        evaluate.main_with_args(remaining)
    elif args.command == "calibrate":
        calibrate.main_with_args(remaining)


if __name__ == "__main__":
    main()
