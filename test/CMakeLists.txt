configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WARPFORGE_TEST_DEPENDS
  FileCheck
  count
  not
  warpforge-opt
)

add_lit_testsuite(check-warpforge "Running WarpForge regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${WARPFORGE_TEST_DEPENDS}
)
set_target_properties(check-warpforge PROPERTIES FOLDER "Tests")

add_lit_testsuites(WARPFORGE ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${WARPFORGE_TEST_DEPENDS}
)
