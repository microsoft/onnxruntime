# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The standalone target must not enter normal or plugin CUDA source lists."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SourceFiltersTest(unittest.TestCase):
    def test_standalone_cuda_source_is_excluded_from_shared_provider_list(self):
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("CMake is not installed")
        root = Path(__file__).resolve().parents[5]
        standalone = Path(__file__).with_name("source_compat_cuda.cu")
        ordinary = root / "onnxruntime/contrib_ops/cuda/bert/attention_impl.cu"
        self.assertTrue(standalone.is_file() and ordinary.is_file())
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "source_filters.cmake"
            script.write_text(
                f'include("{root.as_posix()}/cmake/onnxruntime_cuda_source_filters.cmake")\n'
                f'set(sources "{standalone.as_posix()}" "{ordinary.as_posix()}")\n'
                "onnxruntime_filter_cuda_cu_sources(sources)\n"
                f'if(NOT sources STREQUAL "{ordinary.as_posix()}")\n'
                '  message(FATAL_ERROR "Standalone source leaked into provider list: ${sources}")\n'
                "endif()\n"
            )
            result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
