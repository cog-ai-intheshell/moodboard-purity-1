#!/usr/bin/env python3
"""Calibration entrypoint for final thresholds."""

from __future__ import annotations

import argparse


def main_with_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate purity, outlier and spectral thresholds.")
    parser.add_argument("--config", default="configs/training/calibration.yaml")
    args = parser.parse_args(argv)
    print(f"Calibration scaffold ready with config: {args.config}")


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
