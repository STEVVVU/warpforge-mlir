#!/usr/bin/env python3
"""Record deterministic WarpForge IR transformations; no runtime claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .generate_workload import generate_workload, validate_dimensions
except ImportError:
    from generate_workload import generate_workload, validate_dimensions

OPS = ("linalg.matmul", "linalg.add", "linalg.max", "linalg.generic")


def count_ops(text: str) -> dict[str, int]:
    return {
        operation: len(re.findall(rf"(?<![A-Za-z0-9_.]){re.escape(operation)}\b", text))
        for operation in OPS
    }


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warpforge-opt", type=Path, required=True)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized-output", type=Path)
    parser.add_argument("--expected", type=Path)
    return parser.parse_args()


def require_expected_subset(expected: object, actual: object, path: str = "") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path or 'root'}: expected an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path or 'root'}: missing key {key}")
            require_expected_subset(value, actual[key], f"{path}.{key}".strip("."))
        return
    if expected != actual:
        raise ValueError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> int:
    args = parse_args()
    validate_dimensions(args.m, args.n, args.k, args.tile)
    source = generate_workload(args.m, args.n, args.k, args.dtype)

    with tempfile.TemporaryDirectory(prefix="warpforge-") as directory:
        input_path = Path(directory) / "workload.mlir"
        input_path.write_text(source, encoding="utf-8", newline="\n")
        optimized = run(
            [
                str(args.warpforge_opt),
                str(input_path),
                "--warpforge-fuse-matmul-epilogue",
                "--warpforge-simplify",
            ]
        ).stdout
        roofline = run(
            [str(args.warpforge_opt), str(input_path), "--warpforge-roofline"]
        ).stderr.strip()

    before = count_ops(source)
    after = count_ops(optimized)
    element_bytes = 2 if args.dtype == "f16" else 4
    flops = 2 * args.m * args.n * args.k
    tensor_bytes = element_bytes * (
        args.m * args.k + args.k * args.n + 2 * args.m * args.n
    )
    version = run([str(args.warpforge_opt), "--version"]).stdout.splitlines()[0]
    record = {
        "schema": 1,
        "tool": version,
        "workload": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "tile": args.tile,
        },
        "model": {
            "flops": flops,
            "minimum_tensor_bytes": tensor_bytes,
            "arithmetic_intensity_flop_per_byte": round(flops / tensor_bytes, 3),
        },
        "before": {"sha256": digest(source), "operation_counts": before},
        "after": {"sha256": digest(optimized), "operation_counts": after},
        "operation_delta": {
            operation: after[operation] - before[operation] for operation in OPS
        },
        "roofline_report": roofline,
        "runtime_measurement": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.optimized_output:
        args.optimized_output.write_text(
            optimized, encoding="utf-8", newline="\n"
        )
    if args.expected:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        require_expected_subset(expected, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
