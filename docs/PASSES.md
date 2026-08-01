# Pass Reference

| Pass or pipeline | Anchor | Purpose |
| --- | --- | --- |
| `warpforge-verify-inference` | `func.func` | Check static rank-2 f16/f32 matmul legality |
| `warpforge-fuse-matmul-epilogue` | `func.func` | Fuse same-shape bias add and zero-threshold ReLU |
| `warpforge-tile-vectorize` | `func.func` | Tile through `TilingInterface`, then vectorize |
| `warpforge-roofline` | `func.func` | Print FLOP, ideal byte, and intensity estimates |
| `warpforge-simplify` | module pipeline | Run canonicalization and CSE |
| `warpforge-inference-opt` | module pipeline | Compose the inference optimization sequence |

## Tile options

```text
--warpforge-tile-vectorize="tile-m=16 tile-n=16 tile-k=16"
```

Each value must be positive and divide its corresponding static matmul
dimension. Non-divisible shapes fail with a padding/masking recommendation.

## Analysis output

For `M=64`, `N=128`, `K=32`, and f16:

```text
[warpforge-roofline] function=encoder_projection op=linalg.matmul \
shape=64x128x32 flops=524288 bytes=45056 intensity=11.636 flop/byte
```

This is an analytical report, not a hardware roofline measurement.
