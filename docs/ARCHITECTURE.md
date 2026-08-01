# WarpForge Architecture

## Scope

WarpForge is an out-of-tree LLVM/MLIR 19 optimizer for static inference
matmuls. It adds no private IR dialect. The driver registers upstream MLIR
dialects, extensions, and passes, while `WarpForgePasses` operates on standard
`func`, `arith`, `tensor`, `linalg`, `scf`, `vector`, `LLVM`, `gpu`, and `NVVM`
operations.

```text
tensor + linalg input
  -> legality verification
  -> bias/ReLU epilogue fusion
  -> canonicalize + CSE
  -> TilingInterface / scf.for tiles
  -> Linalg vectorization
  -> upstream bufferization and conversion passes
  -> LLVM dialect

gpu module -> upstream GPU-to-NVVM conversion -> NVVM + LLVM dialect IR
```

## Pass invariants

### `warpforge-verify-inference`

- Examines every `linalg.matmul` in a `func.func`.
- Requires tensor semantics, rank two, fully static dimensions, and one
  uniform f16 or f32 element type.
- Verifies the `MxK * KxN -> MxN` relationship.
- Emits an operation-anchored diagnostic and fails the pass on violation.

Static shapes are a deliberate contract for deterministic tile selection.
Bufferization belongs after graph optimization.

### `warpforge-fuse-matmul-epilogue`

Matches a single-use `linalg.matmul -> linalg.add -> linalg.max` chain where
the max threshold is a floating-point zero splat. It replaces add and max with
one rank-two `linalg.generic` containing `arith.addf` and `arith.maximumf`.
The matmul remains explicit, so later tiling can target its reduction.

The pass currently requires a same-shaped bias tensor. Broadcast bias is not
silently accepted.

### `warpforge-tile-vectorize`

Uses the upstream `TilingInterface`, `scf::tileUsingSCF`, and
`linalg::vectorize` APIs. Defaults are M=16, N=16, K=16 and can be changed with
pass options. Shapes must be exactly divisible by all tiles; the pass rejects
tails with an actionable diagnostic instead of assuming padding or masking.

### `warpforge-roofline`

Reports `2*M*N*K` floating-point operations and a minimum tensor-traffic model:

```text
element_bytes * (M*K + K*N + 2*M*N)
```

The output term is counted once for reading the accumulator and once for
writing it. Arithmetic intensity is FLOPs divided by these idealized bytes.
This excludes cache behavior, packing, launch overhead, and device bandwidth.

## Pipelines

`warpforge-simplify` uses upstream canonicalization and CSE.

`warpforge-inference-opt` composes verification, epilogue fusion,
canonicalization, tiling/vectorization, canonicalization, and CSE. Lowering to
LLVM remains explicit so users can choose bufferization and ABI policy.

The lit fixtures demonstrate:

- Linalg buffer semantics to SCF loops and LLVM dialect;
- generated Vector dialect structure and Vector-to-LLVM conversion;
- GPU index operations to NVVM special-register reads.

## Failure and determinism

Passes fail rather than partially optimizing illegal shapes. Tile sizes are
explicit, operation traversal follows source order, and the benchmark harness
hashes both source and transformed textual IR. It records operation counts and
the analytical model but no wall-clock or GPU timing.
