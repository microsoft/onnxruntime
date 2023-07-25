# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
import sys
from glob import glob
from shutil import copyfile

import torch
from packaging import version

from onnxruntime.training import ortmodule


def _list_extensions(path):
    extensions = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower() == "setup.py":
                extensions.append(os.path.join(root, name))  # noqa: PERF401
    return extensions


def _list_cpu_extensions():
    return _list_extensions(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, "cpu"))


def _list_cuda_extensions():
    return _list_extensions(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, "cuda"))


def _install_extension(ext_name, ext_path, cwd):
    ret_code = subprocess.call((sys.executable, ext_path, "build"), cwd=cwd)
    if ret_code != 0:
        print(f"There was an error compiling '{ext_name}' PyTorch CPP extension")
        sys.exit(ret_code)


def _get_cuda_extra_build_params():
    nvcc_extra_args = ["-lineinfo", "-O3", "--use_fast_math"]
    cuda_version = torch.version.cuda
    if cuda_version is not None and version.parse(cuda_version) > version.parse("11.2"):
        # If number is 0, the number of threads used is the number of CPUs on the machine.
        nvcc_extra_args += ["--threads", "0"]

    os.environ["ONNXRUNTIME_CUDA_NVCC_EXTRA_ARGS"] = ",".join(nvcc_extra_args)


def build_torch_cpp_extensions():
    """Builds PyTorch CPP extensions and returns metadata."""
    # Run this from within onnxruntime package folder
    is_gpu_available = (torch.version.cuda is not None or torch.version.hip is not None) and (
        ortmodule.ONNXRUNTIME_CUDA_VERSION is not None or ortmodule.ONNXRUNTIME_ROCM_VERSION is not None
    )

    # Docker build don't have CUDA support, but Torch C++ extensions with CUDA may be forced
    force_cuda = bool(os.environ.get("ONNXRUNTIME_FORCE_CUDA", False))

    os.chdir(ortmodule.ORTMODULE_TORCH_CPP_DIR)

    # Extensions might leverage CUDA/ROCM versions internally
    os.environ["ONNXRUNTIME_CUDA_VERSION"] = (
        ortmodule.ONNXRUNTIME_CUDA_VERSION if ortmodule.ONNXRUNTIME_CUDA_VERSION is not None else ""
    )
    os.environ["ONNXRUNTIME_ROCM_VERSION"] = (
        ortmodule.ONNXRUNTIME_ROCM_VERSION if ortmodule.ONNXRUNTIME_ROCM_VERSION is not None else ""
    )

    if torch.version.cuda is not None and ortmodule.ONNXRUNTIME_CUDA_VERSION is not None:
        _get_cuda_extra_build_params()

    ############################################################################
    # Pytorch CPP Extensions that DO require CUDA/ROCM
    ############################################################################
    if is_gpu_available or force_cuda:
        for ext_setup in _list_cuda_extensions():
            _install_extension(ext_setup.split(os.sep)[-2], ext_setup, ortmodule.ORTMODULE_TORCH_CPP_DIR)

    ############################################################################
    # Pytorch CPP Extensions that DO NOT require CUDA/ROCM
    ############################################################################
    for ext_setup in _list_cpu_extensions():
        _install_extension(ext_setup.split(os.sep)[-2], ext_setup, ortmodule.ORTMODULE_TORCH_CPP_DIR)

    ############################################################################
    # Install Pytorch CPP Extensions into local onnxruntime package folder
    ############################################################################
    torch_cpp_exts = glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, "build", "lib.*", "*.so"))
    torch_cpp_exts.extend(glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, "build", "lib.*", "*.dll")))
    torch_cpp_exts.extend(glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, "build", "lib.*", "*.dylib")))
    for ext in torch_cpp_exts:
        dest_ext = os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, os.path.basename(ext))
        print(f"Installing {ext} -> {dest_ext}")
        copyfile(ext, dest_ext)

    # Tear down
    os.environ.pop("ONNXRUNTIME_CUDA_VERSION")
    os.environ.pop("ONNXRUNTIME_ROCM_VERSION")


if __name__ == "__main__":
    build_torch_cpp_extensions()
