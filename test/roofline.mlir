// RUN: warpforge-opt %s --warpforge-roofline 2>&1 | FileCheck %s

func.func @encoder_projection(
    %lhs: tensor<64x32xf16>,
    %rhs: tensor<32x128xf16>,
    %init: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<64x32xf16>, tensor<32x128xf16>)
      outs(%init : tensor<64x128xf16>) -> tensor<64x128xf16>
  return %result : tensor<64x128xf16>
}

// CHECK: [warpforge-roofline] function=encoder_projection op=linalg.matmul
// CHECK-SAME: shape=64x128x32 flops=524288 bytes=45056
// CHECK-SAME: intensity=11.636 flop/byte
