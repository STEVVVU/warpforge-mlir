# Pipeline evidence

## Sub-features

Generate a static matmul workload and compare observed operation counts with checked-in expectations.

## How to get to it (user POV)

Run `scripts/benchmark_pipeline.py` with the built optimizer.

## Driving it with the verification driver

Run the pipeline command in the skill. Inspect `pipeline.json` and `optimized.mlir` in `build/evidence/`.

## Gotchas

FLOPs and bytes are modeled; this does not measure runtime or numerical equivalence.
