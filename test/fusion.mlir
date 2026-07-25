// RUN: warpforge-opt %s --warpforge-fuse-matmul-epilogue | FileCheck %s

func.func @matmul_bias_relu(
    %lhs: tensor<16x32xf32>,
    %rhs: tensor<32x16xf32>,
    %bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %zero = arith.constant dense<0.0> : tensor<16x16xf32>
  %init0 = tensor.empty() : tensor<16x16xf32>
  %init1 = tensor.empty() : tensor<16x16xf32>
  %init2 = tensor.empty() : tensor<16x16xf32>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<16x32xf32>, tensor<32x16xf32>)
      outs(%init0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %biased = linalg.add
      ins(%matmul, %bias : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%init1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %activated = linalg.max
      ins(%biased, %zero : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%init2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %activated : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: warpforge.fused_epilogue = "bias_relu"
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK-NOT: linalg.add
// CHECK-NOT: linalg.max
