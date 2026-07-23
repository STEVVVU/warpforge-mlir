import os

import lit.formats
from lit.llvm import llvm_config

config.name = "WARPFORGE"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.warpforge_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]

llvm_config.use_default_substitutions()
tool_dirs = [config.warpforge_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["FileCheck", "count", "not", "warpforge-opt"], tool_dirs
)
