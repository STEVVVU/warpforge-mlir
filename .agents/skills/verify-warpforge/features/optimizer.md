# Optimizer CLI

## Sub-features

Static shape legality, bias/ReLU fusion, tiling and vectorization, and rejected unsupported shapes.

## How to get to it (user POV)

Invoke `build/bin/warpforge-opt` on an MLIR file with a WarpForge pass flag.

## Driving it with the verification driver

Run `scripts/verify_driver.py` as documented in the skill. Successful outputs must pass their fixture checks and reparse; rejected inputs must exit nonzero and match their diagnostics.

## Gotchas

The fixtures exercise specific examples, not all shapes or numerical equivalence. The driver refuses an existing evidence directory; choose a new path for each run.
