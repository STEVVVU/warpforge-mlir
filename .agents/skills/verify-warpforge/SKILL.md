---
name: verify-warpforge
description: Build and drive the WarpForge optimizer CLI and retain transformation and rejection evidence.
---

# Verify WarpForge

Proven on Ubuntu 24.04 with LLVM/MLIR 19.1.7 in hosted run [41414021411](https://github.com/STEVVVU/warpforge-mlir/actions/runs/41414021411), source `11cc18d1b58f33f681346d255c31ac5e4472e283`. The built CLI, positive and rejected fixtures, FileCheck, successful-output reparsing, observed pipeline IR and identity receipts passed. The downloaded artifact retained those outputs after job completion. Use a clean checkout and follow the package installation in `.github/workflows/ci.yml`. This proves the mapped transformations, not numerical equivalence or runtime speed.

## Launch

From the repository root, set `LLVM_LIT` to the installed LLVM 19 `lit.py`, then run:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir -DLLVM_EXTERNAL_LIT="$LLVM_LIT"
cmake --build build --target warpforge-opt --parallel 2
```

## Doctor

Run `build/bin/warpforge-opt --version` and `git status --short`. Record the source commit, any local changes, tool version, and binary hash with the driver. A dirty checkout cannot be identified by commit alone.

## Drive

```sh
cmake --build build --target check-warpforge
python3 -m unittest discover -s test/python -v
python3 scripts/benchmark_pipeline.py --warpforge-opt build/bin/warpforge-opt \
  --expected evidence/expected-pipeline.json --output build/evidence/pipeline.json \
  --optimized-output build/evidence/optimized.mlir
python3 scripts/verify_driver.py --opt build/bin/warpforge-opt \
  --filecheck /usr/lib/llvm-19/bin/FileCheck --output build/evidence/verification
```

The driver invokes the actual CLI, runs fixture FileCheck assertions on its output and diagnostics, and reparses each successful transformation with the optimizer's verifier. See `features/README.md` for covered paths.

## Evidence

Preserve `build/evidence/` outside cleanup. Choose a new output directory for each run; the driver refuses to overwrite existing evidence. It records commands, exit codes, stdout, stderr, source commit, working-tree status, tracked and untracked source-file hashes, tool version, and binary SHA-256. Each CLI command has a 60-second deadline. CI uploads this directory even after failure when files exist. Inspect the relevant IR and diagnostics as well as exit codes. Operation counts and modeled FLOPs do not establish numerical correctness or speed.

## Cleanup

These commands launch no persistent process. Remove only scratch files created for the run; retain `build/evidence/` and confirm it still exists after cleanup. The build directory can be retained for reuse.
