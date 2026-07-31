#!/usr/bin/env python3
"""Generate a deterministic static matmul + bias + ReLU MLIR workload."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_dimensions(m: int, n: int, k: int, tile: int) -> None:
    if min(m, n, k, tile) <= 0:
        raise ValueError("M, N, K, and tile must be positive")
    if any(dimension % tile for dimension in (m, n, k)):
        raise ValueError(
            f"dimensions {m}x{n}x{k} must be divisible by tile {tile}"
        )


def generate_workload(m: int, n: int, k: int, dtype: str = "f32") -> str:
    if dtype not in {"f16", "f32"}:
        raise ValueError("dtype must be f16 or f32")
    lhs = f"tensor<{m}x{k}x{dtype}>"
    rhs = f"tensor<{k}x{n}x{dtype}>"
    output = f"tensor<{m}x{n}x{dtype}>"
    return f"""// Generated deterministically by scripts/generate_workload.py.
module {{
  func.func @generated_matmul_bias_relu(
      %lhs: {lhs}, %rhs: {rhs}, %bias: {output}) -> {output} {{
    %zero = arith.constant dense<0.0> : {output}
    %init0 = tensor.empty() : {output}
    %init1 = tensor.empty() : {output}
    %init2 = tensor.empty() : {output}
    %matmul = linalg.matmul
        ins(%lhs, %rhs : {lhs}, {rhs})
        outs(%init0 : {output}) -> {output}
    %biased = linalg.add
        ins(%matmul, %bias : {output}, {output})
        outs(%init1 : {output}) -> {output}
    %activated = linalg.max
        ins(%biased, %zero : {output}, {output})
        outs(%init2 : {output}) -> {output}
    return %activated : {output}
  }}
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f32")
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_dimensions(args.m, args.n, args.k, args.tile)
    text = generate_workload(args.m, args.n, args.k, args.dtype)
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
