// RUN: warpforge-opt %s --convert-vector-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | FileCheck %s

func.func @reduce_vector(%input: vector<8xf32>) -> f32 {
  %result = vector.reduction <add>, %input : vector<8xf32> into f32
  return %result : f32
}

// CHECK-LABEL: llvm.func @reduce_vector
// CHECK: llvm.
// CHECK: llvm.return
// CHECK-NOT: vector.
// CHECK-NOT: func.
