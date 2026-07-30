// RUN: not warpforge-opt %s --warpforge-tile-vectorize 2>&1 | FileCheck %s

func.func @tail_requires_policy(
    %lhs: tensor<31x16xf32>,
    %rhs: tensor<16x16xf32>,
    %init: tensor<31x16xf32>) -> tensor<31x16xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<31x16xf32>, tensor<16x16xf32>)
      outs(%init : tensor<31x16xf32>) -> tensor<31x16xf32>
  return %result : tensor<31x16xf32>
}

// CHECK: error: 'linalg.matmul' op shape 31x16 * 16x16 is not divisible
// CHECK-SAME: pad or select explicit divisors to avoid masked tails
