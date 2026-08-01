add_mlir_library(WarpForgePasses
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTransforms
  MLIRVectorDialect
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(WarpForgePasses PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
  )
endif()
