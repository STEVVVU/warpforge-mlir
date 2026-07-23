get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(warpforge-opt warpforge-opt.cpp)
llvm_update_compile_flags(warpforge-opt)

target_link_libraries(warpforge-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  WarpForgePasses
)

mlir_check_all_link_libraries(warpforge-opt)
