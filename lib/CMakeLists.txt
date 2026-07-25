add_mlir_library(WarpForgePasses
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
