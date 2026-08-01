# Local Verification

Recorded on 2026-07-17 from Windows.

## Completed locally

```text
python -m unittest discover -s test/python -v
Ran 4 tests
OK
```

`python -m py_compile` also accepted the generator, benchmark harness, and test
module. A generator smoke test emitted one each of `linalg.matmul`,
`linalg.add`, and `linalg.max` with the requested f16 tensor dimensions.
`git diff --check` reported no whitespace errors.

## Toolchain boundary

An official LLVM/MLIR 19.1.7 Ubuntu toolchain was extracted under the ignored
`.toolchain/` directory for API checking without changing the host system.
CMake identified Clang 19.1.7, but its compiler probe could not complete:
the extracted sysroot lacked the merged-`/usr` compatibility paths required by
the libc linker script. No WarpForge translation unit was compiled or linked.

Therefore the C++ build, lit/FileCheck suite, and generated optimization record
were not executed locally. `.github/workflows/ci.yml` defines the intended
verification environment: Ubuntu 24.04 with official apt.llvm.org Clang,
LLVM, and MLIR 19 packages under `/usr/lib/llvm-19`. CI was not run from this
local-only checkout.

The implementation's unstable API points were checked against official
`llvmorg-19.1.7` headers, including `MlirOptMain`,
`scf::SCFTilingOptions::setTileSizes`, `scf::tileUsingSCF`, and
`linalg::vectorize`.
