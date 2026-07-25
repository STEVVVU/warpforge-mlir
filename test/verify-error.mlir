// RUN: not warpforge-opt %s --warpforge-verify-inference 2>&1 | FileCheck %s

func.func @dynamic_shape_rejected(
    %lhs: tensor<?x64xf32>,
    %rhs: tensor<64x128xf32>,
    %init: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x128xf32>)
      outs(%init : tensor<?x128xf32>) -> tensor<?x128xf32>
  return %result : tensor<?x128xf32>
}

// CHECK: error: 'linalg.matmul' op requires fully static shapes for deterministic GPU tiling
// CHECK-SAME: specialize dynamic dimensions before this pipeline
