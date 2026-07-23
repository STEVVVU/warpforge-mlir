add_mlir_library(WarpForgePasses
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
