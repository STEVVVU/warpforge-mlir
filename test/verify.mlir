// RUN: warpforge-opt %s --warpforge-verify-inference | FileCheck %s

func.func @legal_matmul(
    %lhs: tensor<64x32xf16>,
    %rhs: tensor<32x128xf16>,
    %init: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<64x32xf16>, tensor<32x128xf16>)
      outs(%init : tensor<64x128xf16>) -> tensor<64x128xf16>
  return %result : tensor<64x128xf16>
}

// CHECK-LABEL: func.func @legal_matmul
// CHECK: linalg.matmul
