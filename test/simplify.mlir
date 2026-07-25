// RUN: warpforge-opt %s --warpforge-simplify | FileCheck %s

func.func @fold_constants() -> f32 {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %sum = arith.addf %one, %two : f32
  return %sum : f32
}

// CHECK-LABEL: func.func @fold_constants
// CHECK: %[[THREE:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-NEXT: return %[[THREE]] : f32
