// RUN: warpforge-opt %s --convert-linalg-to-loops | FileCheck %s --check-prefix=SCF
// RUN: warpforge-opt %s --convert-linalg-to-loops --lower-affine \
// RUN:   --convert-scf-to-cf --convert-index-to-llvm --convert-arith-to-llvm \
// RUN:   --expand-strided-metadata --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

func.func @matmul_to_llvm(
    %lhs: memref<16x16xf32>,
    %rhs: memref<16x16xf32>,
    %out: memref<16x16xf32>) {
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xf32>, memref<16x16xf32>)
      outs(%out : memref<16x16xf32>)
  return
}

// SCF-LABEL: func.func @matmul_to_llvm
// SCF: scf.for
// SCF: memref.load
// SCF: arith.mulf
// SCF: arith.addf
// SCF: memref.store

// LLVM-LABEL: llvm.func @matmul_to_llvm
// LLVM: llvm.br
// LLVM: llvm.load
// LLVM: llvm.fmul
// LLVM: llvm.fadd
// LLVM: llvm.store
// LLVM: llvm.return
// LLVM-NOT: linalg.
// LLVM-NOT: scf.
