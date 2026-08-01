# WarpForge MLIR

WarpForge is an out-of-tree LLVM/MLIR 19 compiler project for static AI
inference kernels. It uses the official C++ pass, rewrite, TilingInterface,
Linalg vectorization, LLVM conversion, and NVVM dialect APIs. It does not
implement a substitute IR or parser.

The project builds `warpforge-opt`, an `mlir-opt`-style driver that accepts
standard MLIR and adds:

- static matmul shape/type legality verification with actionable diagnostics;
- matmul bias/ReLU epilogue fusion into `linalg.generic`;
- upstream canonical folding and common-subexpression elimination;
- configurable SCF tiling followed by Linalg-to-Vector conversion;
- deterministic FLOP, minimum-byte, and arithmetic-intensity analysis.

## Toolchain

The supported configuration is Ubuntu 24.04 with the official apt.llvm.org
LLVM 19 repository and the versioned packages `clang-19`, `llvm-19-dev`,
`llvm-19-tools`, `libmlir-19-dev`, and `mlir-19-tools`. CMake reads:

```text
LLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
MLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
```

Configure and test:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang-19 \
  -DCMAKE_CXX_COMPILER=clang++-19 \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
cmake --build build --target warpforge-opt
cmake --build build --target check-warpforge
python3 -m unittest discover -s test/python -v
```

## Usage

Inspect or run individual stages:

```bash
build/bin/warpforge-opt input.mlir --warpforge-verify-inference
build/bin/warpforge-opt input.mlir --warpforge-fuse-matmul-epilogue
build/bin/warpforge-opt input.mlir --warpforge-roofline
build/bin/warpforge-opt input.mlir --warpforge-inference-opt
```

Generate a reproducible workload and evidence record:

```bash
python3 scripts/generate_workload.py -o workload.mlir
python3 scripts/benchmark_pipeline.py \
  --warpforge-opt build/bin/warpforge-opt \
  --expected evidence/expected-pipeline.json \
  --output build/evidence/pipeline.json \
  --optimized-output build/evidence/optimized.mlir
```

See [architecture](docs/ARCHITECTURE.md), [pass invariants](docs/PASSES.md), and
the [lit fixtures](test/) for lowering commands and diagnostics.

## Deterministic evidence

The default 64x64x64 f32 workload has:

- 524,288 modeled FLOPs;
- 65,536 idealized tensor bytes and 8.0 FLOP/byte;
- one `linalg.matmul`, one `linalg.add`, and one `linalg.max` before fusion;
- deltas of -1 `linalg.add`, -1 `linalg.max`, and +1 `linalg.generic`.

These expectations are checked in
[`evidence/expected-pipeline.json`](evidence/expected-pipeline.json). CI
regenerates a full record containing tool version, exact operation counts, and
pre/post IR SHA-256 hashes. No runtime measurement is included.

The test surface contains 10 lit/FileCheck fixtures and 4 Python unit tests.
On this checkout, the Python tests pass. A local CMake configure identified
Clang 19.1.7 but could not link its compiler probe against the extracted
Ubuntu sysroot, so C++ compilation and lit execution remain deferred to the
pinned Ubuntu CI job. See the
[local verification record](evidence/LOCAL_VERIFICATION.md).

## Lowering scope

Fixtures demonstrate Linalg to SCF to LLVM dialect lowering, Vector to LLVM
conversion, and GPU index operations to NVVM special-register IR. They do not
compile PTX, launch a CUDA kernel, or claim NVIDIA GPU performance.

Current limitations:

- rank-2 static matmuls only; no dynamic-shape specialization;
- exact tile divisibility; no padding, predication, or masked tails;
- same-shaped bias tensor; no broadcast epilogue matching;
- no convolution, attention, quantization, layout search, or cost-driven tile
  selection;
- no CUDA runtime integration, PTX assembly, or hardware benchmarking;
- roofline bytes are an ideal lower bound, not measured memory traffic.

## Resume bullets

- Built an out-of-tree LLVM/MLIR 19 optimizer using official Linalg,
  TilingInterface, SCF, Vector, LLVM, and NVVM APIs for static inference
  legality, graph fusion, GPU-oriented tiling, and progressive lowering.
- Implemented deterministic matmul analysis and transformation evidence: the
  64x64x64 f32 fixture models 524,288 FLOPs and 65,536 minimum tensor bytes,
  while fusion removes two named epilogue ops and introduces one generic op.
- Added 10 lit/FileCheck fixtures, 4 Python tests, warnings-as-errors builds,
  actionable failure diagnostics, and Ubuntu CI pinned to distro LLVM/MLIR 19.

## License

MIT
