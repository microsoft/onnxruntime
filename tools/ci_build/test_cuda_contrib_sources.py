# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class CudaContribSourcesTest(unittest.TestCase):
    CPU_DEPENDENT_KERNELS = (
        "attention.cc",
        "decoder_masked_self_attention.cc",
        "embed_layer_norm.cc",
        "fast_gelu.cc",
        "longformer_attention.cc",
    )

    def _get_sources(self, disable_contrib_ops, cuda_minimal=False, flash_attention=False):
        repo_root = Path(__file__).resolve().parents[2]
        ort_root = repo_root / "onnxruntime"
        cmake_dir = repo_root / "cmake"
        provider = (cmake_dir / "onnxruntime_providers_cuda.cmake").read_text(encoding="utf-8")
        # Execute the production source selection without requiring a CUDA toolchain.
        source_selection = provider[: provider.index("  if (onnxruntime_ENABLE_TRAINING_OPS)")]
        with tempfile.TemporaryDirectory(prefix="ort cuda contrib sources ") as directory:
            source = Path(directory)
            for module in ("onnxruntime_cuda_source_filters.cmake", "onnxruntime_cuda_cccl.cmake"):
                shutil.copyfile(cmake_dir / module, source / module)
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(TestCudaContribSources LANGUAGES NONE)
set(ONNXRUNTIME_ROOT "{ort_root.as_posix()}")
set(onnxruntime_DISABLE_CONTRIB_OPS {"ON" if disable_contrib_ops else "OFF"})
set(onnxruntime_CUDA_MINIMAL {"ON" if cuda_minimal else "OFF"})
set(onnxruntime_USE_FLASH_ATTENTION {"ON" if flash_attention else "OFF"})
{source_selection}
set(sources ${{onnxruntime_providers_cuda_src}} ${{onnxruntime_cuda_flash_attention_srcs}})
list(JOIN sources "\\n" sources)
file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt" "${{sources}}")
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
            return {
                Path(path).relative_to(ort_root).as_posix()
                for path in (source / "build" / "sources.txt").read_text(encoding="utf-8").splitlines()
            }

    def test_disabled_contrib_excludes_cpu_dependent_kernels_and_preserves_attention(self):
        for flash_attention in (False, True):
            with self.subTest(flash_attention=flash_attention):
                sources = self._get_sources(disable_contrib_ops=True, flash_attention=flash_attention)
                for kernel in self.CPU_DEPENDENT_KERNELS:
                    self.assertNotIn(f"contrib_ops/cuda/bert/{kernel}", sources)
                for helper in (
                    "attention_kernel_options.cc",
                    "attention_impl.cu",
                    "attention_kv_cache.cu",
                    "group_query_attention_impl.cu",
                    "unfused_attention.cu",
                    "cutlass_fmha/memory_efficient_attention.cu",
                    "flash_attention/flash_api.cc",
                    "cudnn_fmha/cudnn_flash_attention.cc",
                ):
                    self.assertIn(f"contrib_ops/cuda/bert/{helper}", sources)
                self.assertIn("core/providers/cuda/llm/attention.cc", sources)
                self.assertEqual(
                    any("/flash_attention/" in path and path.endswith(".cu") for path in sources),
                    flash_attention,
                )

    def test_enabled_contrib_preserves_cpu_dependent_kernels(self):
        sources = self._get_sources(disable_contrib_ops=False)
        for kernel in self.CPU_DEPENDENT_KERNELS:
            self.assertIn(f"contrib_ops/cuda/bert/{kernel}", sources)

    def test_cuda_minimal_excludes_bert_sources(self):
        for disable_contrib_ops in (False, True):
            with self.subTest(disable_contrib_ops=disable_contrib_ops):
                sources = self._get_sources(disable_contrib_ops=disable_contrib_ops, cuda_minimal=True)
                self.assertFalse(any(path.startswith("contrib_ops/") for path in sources))
                self.assertNotIn("core/providers/cuda/llm/attention.cc", sources)


if __name__ == "__main__":
    unittest.main()
