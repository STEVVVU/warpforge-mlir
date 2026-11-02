# Executed optimizer verification

[Hosted run 41414021411](https://github.com/STEVVVU/warpforge-mlir/actions/runs/41414021411) built and executed source `11cc18d1b58f33f681346d255c31ac5e4472e283` on Ubuntu 24.04 with LLVM/MLIR 19.1.7. Ten lit/FileCheck fixtures and seven Python tests passed. The maintained driver additionally retained each command, exit code, output IR, diagnostic and successful-output reparse.

The five mapped CLI cases checked static-shape acceptance, dynamic-shape rejection, bias/ReLU epilogue fusion, tiling/vectorization, and rejection of non-divisible tiles. All FileCheck invocations and successful-output reparses exited zero; both intentionally rejected inputs exited one with the required diagnostics. The latter path prints numeric tile sizes `16x16x16`. The previous binary printed control characters because the diagnostic streamed pass option objects directly; explicit integer values and a FileCheck assertion cover the correction.

The generated 64 by 64 f32 matmul fixture retained one `linalg.matmul`. Fusion replaced its separate `linalg.add` and `linalg.max` epilogues with one `linalg.generic`, changing three named operations to two operations. Actual transformed IR was retained in `optimized.mlir` with SHA-256 `ad6e94120120619dd6dde6f6d468a48dd90fd6a63472ea98d4f7b89919fdeeb3`. The matmul itself remains a separate operation.

The `warpforge-evidence` artifact contains the pipeline result, observed IR and diagnostics, clean source commit, per-source-file hashes, optimizer version and binary SHA-256. The main agent downloaded the complete artifact after job completion, inspected the outputs and reconciled their expected exit codes. The verifier launches no persistent service. Earlier run [41412396557](https://github.com/STEVVVU/warpforge-mlir/actions/runs/41412396557) is preserved and contains the unreadable diagnostic that motivated the regression fix.

The fixture's 524,288 FLOPs and 65,536 minimum tensor bytes are analytic model outputs. These runs do not measure execution speed, GPU code, numerical equivalence or memory traffic. Structural IR checks are not a substitute for those measurements.
