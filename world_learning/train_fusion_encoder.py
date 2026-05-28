#!/usr/bin/env python3
"""Future fusion-encoder training entrypoint."""

from __future__ import annotations

import argparse


def main_with_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the future aesthetic fusion encoder.")
    parser.add_argument("--config", default="configs/training/fusion_encoder_v1.yaml")
    args = parser.parse_args(argv)
    print(f"Fusion encoder training is not implemented yet. Config reserved: {args.config}")


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
