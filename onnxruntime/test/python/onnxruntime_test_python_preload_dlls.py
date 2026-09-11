# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0114,C0115,C0116,W0212
import unittest

import onnxruntime


class TestGetNvidiaDllPaths(unittest.TestCase):
    """Unit tests for the private _get_nvidia_dll_paths helper that locates CUDA/cuDNN
    libraries inside the NVIDIA site-packages folders.

    NVIDIA restructured the CUDA Python wheels starting with CUDA 13: the per-component
    packages (cublas, cufft, cuda_runtime, ...) were consolidated into a single
    "nvidia/cu{major}" tree. These tests pin down the expected relative paths for the
    old (CUDA 12) and new (CUDA 13) layouts on both Windows and Linux.
    """

    def _paths(self, **kwargs):
        return onnxruntime._get_nvidia_dll_paths(**kwargs)

    # ---- CUDA 12 (legacy per-component layout) --------------------------------------
    def test_cuda12_windows(self):
        paths = self._paths(is_windows=True, build_cuda_version="12.4", cudnn=False)
        self.assertIn(("nvidia", "cublas", "bin", "cublasLt64_12.dll"), paths)
        self.assertIn(("nvidia", "cublas", "bin", "cublas64_12.dll"), paths)
        self.assertIn(("nvidia", "cufft", "bin", "cufft64_11.dll"), paths)
        self.assertIn(("nvidia", "cuda_runtime", "bin", "cudart64_12.dll"), paths)

    def test_cuda12_linux(self):
        paths = self._paths(is_windows=False, build_cuda_version="12.4", cudnn=False)
        self.assertIn(("nvidia", "cublas", "lib", "libcublasLt.so.12"), paths)
        self.assertIn(("nvidia", "cublas", "lib", "libcublas.so.12"), paths)
        self.assertIn(("nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"), paths)
        self.assertIn(("nvidia", "curand", "lib", "libcurand.so.10"), paths)
        self.assertIn(("nvidia", "cufft", "lib", "libcufft.so.11"), paths)
        self.assertIn(("nvidia", "cuda_runtime", "lib", "libcudart.so.12"), paths)

    # ---- CUDA 13 (consolidated "cu13" layout) ---------------------------------------
    def test_cuda13_windows_x86_64(self):
        paths = self._paths(is_windows=True, build_cuda_version="13.2", cudnn=False, arch="x86_64")
        self.assertIn(("nvidia", "cu13", "bin", "x86_64", "cublasLt64_13.dll"), paths)
        self.assertIn(("nvidia", "cu13", "bin", "x86_64", "cublas64_13.dll"), paths)
        self.assertIn(("nvidia", "cu13", "bin", "x86_64", "cufft64_12.dll"), paths)
        self.assertIn(("nvidia", "cu13", "bin", "x86_64", "cudart64_13.dll"), paths)

    def test_cuda13_windows_arch_override(self):
        paths = self._paths(is_windows=True, build_cuda_version="13.2", cudnn=False, arch="arm64")
        self.assertIn(("nvidia", "cu13", "bin", "arm64", "cudart64_13.dll"), paths)

    def test_cuda13_linux_is_flat(self):
        paths = self._paths(is_windows=False, build_cuda_version="13.2", cudnn=False)
        # Linux consolidated layout has no architecture sub-folder (flat "lib").
        self.assertIn(("nvidia", "cu13", "lib", "libcublasLt.so.13"), paths)
        self.assertIn(("nvidia", "cu13", "lib", "libcublas.so.13"), paths)
        self.assertIn(("nvidia", "cu13", "lib", "libnvrtc.so.13"), paths)
        self.assertIn(("nvidia", "cu13", "lib", "libcurand.so.10"), paths)
        self.assertIn(("nvidia", "cu13", "lib", "libcufft.so.12"), paths)
        self.assertIn(("nvidia", "cu13", "lib", "libcudart.so.13"), paths)

    # ---- cuDNN keeps its own package/layout in both schemes -------------------------
    def test_cudnn_layout_unchanged(self):
        for build_cuda_version in ("12.4", "13.2"):
            win = self._paths(is_windows=True, build_cuda_version=build_cuda_version, cuda=False)
            self.assertIn(("nvidia", "cudnn", "bin", "cudnn64_9.dll"), win)

            linux = self._paths(is_windows=False, build_cuda_version=build_cuda_version, cuda=False)
            self.assertEqual(linux, [("nvidia", "cudnn", "lib", "libcudnn.so.9")])

    # ---- toggles --------------------------------------------------------------------
    def test_cuda_and_cudnn_toggles(self):
        self.assertEqual(self._paths(is_windows=False, build_cuda_version="13.2", cuda=False, cudnn=False), [])

        cuda_only = self._paths(is_windows=False, build_cuda_version="13.2", cuda=True, cudnn=False)
        self.assertTrue(all(p[1] == "cu13" for p in cuda_only))


if __name__ == "__main__":
    unittest.main()
