# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import tempfile
import unittest
from pathlib import Path


class CudaPluginArchitecturesTest(unittest.TestCase):
    def _check_llm_target(self, architectures, msvc, expected_architectures, expect_native_sm120):
        cmake_dir = Path(__file__).resolve().parents[2] / "cmake"
        plugin = (cmake_dir / "onnxruntime_providers_cuda_plugin.cmake").read_text(encoding="utf-8")
        # Execute the production LLM block without configuring the rest of the CUDA provider.
        start = plugin.index("  if(_cuda_plugin_llm_srcs)")
        end = plugin.index("  if(_cuda_plugin_llm_fp4_srcs)", start)
        llm_block = plugin[start:end]
        with tempfile.TemporaryDirectory(prefix="ort cuda plugin arch ") as directory:
            source = Path(directory)
            (source / "empty.cc").write_text("", encoding="utf-8")
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(TestCudaPluginArchitectures LANGUAGES CXX)
include("{cmake_dir.as_posix()}/external/cuda_configuration.cmake")
include("{cmake_dir.as_posix()}/onnxruntime_cuda_source_filters.cmake")
set(CMAKE_CUDA_COMPILER_VERSION 13.0)
set(CMAKE_CUDA_ARCHITECTURES "{architectures}")
setup_cuda_architectures()
set(MSVC {"ON" if msvc else "OFF"})
function(onnxruntime_add_object_library name)
  add_library("${{name}}" OBJECT ${{ARGN}})
endfunction()
add_library(onnxruntime_providers_cuda_plugin STATIC empty.cc)
add_custom_target(generated_headers)
set(onnxruntime_EXTERNAL_DEPENDENCIES generated_headers)
set(onnxruntime_plugin_nvcc_threads 1)
set(_cuda_plugin_llm_srcs "${{CMAKE_CURRENT_SOURCE_DIR}}/empty.cc")
set(_cuda_plugin_shared_compile_options "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>")
{llm_block}
get_target_property(archs onnxruntime_providers_cuda_plugin_llm CUDA_ARCHITECTURES)
if(NOT archs STREQUAL "{expected_architectures}")
  message(FATAL_ERROR "Unexpected LLM architectures: ${{archs}}")
endif()
get_target_property(options onnxruntime_providers_cuda_plugin_llm COMPILE_OPTIONS)
set(native_option "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_120,code=sm_120>")
if({"NOT " if expect_native_sm120 else ""}native_option IN_LIST options)
  message(FATAL_ERROR "Unexpected native SM120 compile options: ${{options}}")
endif()
if(NOT "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>" IN_LIST options)
  message(FATAL_ERROR "Shared CUDA compile options were lost: ${{options}}")
endif()
""",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["cmake", "-S", str(source), "-B", str(source / "build")],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_msvc_sm120_spellings_add_native_sass_and_preserve_virtual_ptx(self):
        for architectures in ("120", "120a", "120-real", "120a-real"):
            with self.subTest(architectures=architectures):
                self._check_llm_target(architectures, True, "120-virtual", True)
        for architectures in ("75;86;120", "75;86;120a"):
            with self.subTest(architectures=architectures):
                self._check_llm_target(architectures, True, "75-real;86-real;120-virtual", True)

    def test_other_architectures_and_non_msvc_builds_do_not_add_sm120_sass(self):
        cases = (
            ("75;86;90", True, "75-real;86-real;90a-real"),
            ("121", True, "121-virtual"),
            ("86;120-virtual", True, "86-real;120-virtual"),
            ("120", False, "120a-real"),
            ("120a", False, "120a-real"),
            ("86;120a", False, "86-real;120a-real"),
        )
        for architectures, msvc, expected_architectures in cases:
            with self.subTest(architectures=architectures, msvc=msvc):
                self._check_llm_target(architectures, msvc, expected_architectures, False)


if __name__ == "__main__":
    unittest.main()
