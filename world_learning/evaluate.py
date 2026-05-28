#!/usr/bin/env python3
"""Evaluation entrypoint for D_test and D_guard."""

from __future__ import annotations

import argparse


def main_with_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate world-model artifacts.")
    parser.add_argument("--split", default="datasets/splits/D_test")
    args = parser.parse_args(argv)
    print(f"Evaluation scaffold ready for split: {args.split}")


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
