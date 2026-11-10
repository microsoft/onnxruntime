# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the CUDA architectures that the packages are built for.

They run the production CMake code (cuda_configuration.cmake, onnxruntime_cuda_source_filters.cmake and the
FlashAttention blocks of the CUDA provider and the CUDA plugin) in a throw-away CMake project that enables no
language, so neither a CUDA toolkit nor a compiler is needed.
"""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_DIR = REPO_ROOT / "cmake"
GITHUB_DIR = REPO_ROOT / "tools" / "ci_build" / "github"


class CudaPackageArchitecturesTest(unittest.TestCase):
    def _run_cmake(self, architectures, body, prelude="", before_setup=""):
        """Configure a project that runs `before_setup`, setup_cuda_architectures(), `prelude` and `body`."""
        with tempfile.TemporaryDirectory(prefix="ort cuda arch ") as directory:
            source = Path(directory)
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(TestCudaPackageArchitectures LANGUAGES NONE)
include("{CMAKE_DIR.as_posix()}/external/cuda_configuration.cmake")
include("{CMAKE_DIR.as_posix()}/onnxruntime_cuda_source_filters.cmake")
set(CMAKE_CUDA_COMPILER_VERSION 13.0)
set(CMAKE_CUDA_ARCHITECTURES "{architectures}")
{before_setup}
setup_cuda_architectures()
{prelude}
{body}
""",
                encoding="utf-8",
            )
            return subprocess.run(
                ["cmake", "-S", str(source), "-B", str(source / "build")],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def _configure(self, architectures, body, prelude=""):
        """Run `body` after setup_cuda_architectures() and return the values printed as `RESULT <name>=<value>`."""
        result = self._run_cmake(architectures, body, prelude)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(re.findall(r"RESULT (\w+)=(.*)", result.stdout))

    def _flash_architectures(self, architectures, reuse_sm80=None, plugin=False):
        """Return the CUDA_ARCHITECTURES of the FlashAttention library, as computed by the production CMake code.

        `reuse_sm80` sets onnxruntime_FLASH_ATTENTION_REUSE_SM80; None keeps the default from CMakeLists.txt.
        """
        if plugin:
            text = (CMAKE_DIR / "onnxruntime_providers_cuda_plugin.cmake").read_text(encoding="utf-8")
            start = text.index("if(_cuda_plugin_flash_attention_srcs)")
            block = text[start : text.index("if(_cuda_plugin_xqa_srcs)", start)]
            prelude = """
function(onnxruntime_add_cuda_plugin_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES;COMPILE_OPTIONS")
  message(STATUS "RESULT FLASH_ARCHS=${_ARG_CUDA_ARCHITECTURES}")
endfunction()
set(_cuda_plugin_flash_attention_srcs flash.cu)
set(_cuda_plugin_shared_compile_options "")
"""
        else:
            text = (CMAKE_DIR / "onnxruntime_providers_cuda.cmake").read_text(encoding="utf-8")
            start = text.index("    if(onnxruntime_cuda_flash_attention_srcs)")
            block = text[start : text.index("    # XQA kernels require SM80+.", start)]
            prelude = """
function(onnxruntime_add_cuda_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES")
  message(STATUS "RESULT FLASH_ARCHS=${_ARG_CUDA_ARCHITECTURES}")
endfunction()
set(onnxruntime_cuda_flash_attention_srcs flash.cu)
"""
        prelude += "set(onnxruntime_FLASH_NVCC_THREADS 1)\n"
        if reuse_sm80 is None:
            text = (CMAKE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
            option = re.search(r"^option\(onnxruntime_FLASH_ATTENTION_REUSE_SM80 .*\)$", text, re.MULTILINE)
            self.assertIsNotNone(option)
            prelude += option.group(0)
        else:
            prelude += f"set(onnxruntime_FLASH_ATTENTION_REUSE_SM80 {'ON' if reuse_sm80 else 'OFF'})"
        # Every case has an SM80+ entry, so the block never falls back to compiling the sources in the parent
        # target, which this project does not define.
        results = self._configure(architectures, block, prelude)
        return results.get("FLASH_ARCHS", "")

    def test_flash_attention_reuses_sm80_sass_on_sm86_and_sm89_only_when_requested(self):
        cases = (
            # (architectures, expected without the option, expected with the option)
            (
                "75-real;80-real;86-real;89-real;90-real;120-real",
                "80-real;86-real;89-real;90a-real;120a-real",
                "80-real;90a-real;120a-real",
            ),
            (
                "60-real;70-real;75-real;80-real;86-real;89-real;90-real;120-real",
                "80-real;86-real;89-real;90a-real;120a-real",
                "80-real;90a-real;120a-real",
            ),
            # Without SM80 SASS there is nothing to reuse.
            ("86-real;89-real", "86-real;89-real", "86-real;89-real"),
            ("75;86;89", "86-real;89-real", "86-real;89-real"),
            # SM87 (Tegra) is never dropped, and PTX entries are kept.
            ("80-real;87-real;89-real", "80-real;87-real;89-real", "80-real;87-real"),
            ("80-real;86-real;89-virtual", "80-real;86-real;89-virtual", "80-real;89-virtual"),
            ("86-real;89-real;80-virtual", "86-real;89-real;80-virtual", "86-real;89-real;80-virtual"),
        )
        for plugin in (False, True):
            for architectures, without_option, with_option in cases:
                with self.subTest(architectures=architectures, plugin=plugin):
                    self.assertEqual(self._flash_architectures(architectures, False, plugin), without_option)
                    self.assertEqual(self._flash_architectures(architectures, True, plugin), with_option)

    def test_x64_packages_reuse_sm80_flash_attention_by_default(self):
        text = (GITHUB_DIR / "azure-pipelines" / "py-cuda13-packaging-pipeline.yml").read_text(encoding="utf-8")
        for platform, expected in (("linux", "80-real;90a-real;120a-real"), ("windows", "80-real;120a-real")):
            match = re.search(rf"cmake_{platform}_x64_cuda_archs:\s*'([^']+)'", text)
            self.assertIsNotNone(match)
            for plugin in (False, True):
                with self.subTest(platform=platform, plugin=plugin):
                    self.assertEqual(self._flash_architectures(match.group(1), plugin=plugin), expected)

    def test_reuse_sm80_sass_keeps_sm86_and_sm89_when_min_sm_excludes_sm80(self):
        results = self._configure(
            "80-real;86-real;89-real",
            """
onnxruntime_filter_cuda_archs(min80 MIN_SM 80 REUSE_SM80_SASS)
onnxruntime_filter_cuda_archs(min86 MIN_SM 86 REUSE_SM80_SASS)
message(STATUS "RESULT MIN80=${min80}")
message(STATUS "RESULT MIN86=${min86}")
""",
        )
        self.assertEqual(results["MIN80"], "80-real")
        # MIN_SM drops the SM80 entry, so there is no SM80 cubin for SM86/SM89 GPUs to run.
        self.assertEqual(results["MIN86"], "86-real;89-real")


if __name__ == "__main__":
    unittest.main()
