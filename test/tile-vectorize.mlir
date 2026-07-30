// RUN: warpforge-opt %s --warpforge-tile-vectorize | FileCheck %s

func.func @tile_and_vectorize(
    %lhs: tensor<32x16xf32>,
    %rhs: tensor<16x32xf32>,
    %init: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<32x16xf32>, tensor<16x32xf32>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @tile_and_vectorize
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.matmul
