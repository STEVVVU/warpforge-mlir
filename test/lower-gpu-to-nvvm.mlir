// RUN: warpforge-opt %s --convert-gpu-to-nvvm | FileCheck %s

gpu.module @kernels {
  func.func @thread_index() -> index {
    %thread = gpu.thread_id x
    func.return %thread : index
  }
}

// CHECK-LABEL: gpu.module @kernels
// CHECK: llvm.func @thread_index
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.sext
// CHECK: llvm.return
// CHECK-NOT: gpu.thread_id
