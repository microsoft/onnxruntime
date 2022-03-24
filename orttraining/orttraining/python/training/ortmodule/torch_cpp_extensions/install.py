# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.training import ortmodule

from glob import glob
from shutil import copyfile
import os
import subprocess
import sys


def _list_extensions(path):
    extensions = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower() == 'setup.py':
                extensions.append(os.path.join(root, name))
    return extensions


def _list_cpu_extensions():
    return _list_extensions(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, 'cpu'))


def _list_cuda_extensions():
    return _list_extensions(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, 'cuda'))


def _install_extension(ext_name, ext_path, cwd):
    ret_code = subprocess.call(f"{sys.executable} {ext_path} build",
                               cwd=cwd,
                               shell=True)
    if ret_code != 0:
        print(f'There was an error compiling "{ext_name}" PyTorch CPP extension')
        sys.exit(ret_code)


def build_torch_cpp_extensions():
    '''Builds PyTorch CPP extensions and returns metadata'''

    # Run this from within onnxruntime package folder
    is_gpu_available = ortmodule.ONNXRUNTIME_CUDA_VERSION is not None or\
        ortmodule.ONNXRUNTIME_ROCM_VERSION is not None
    os.chdir(ortmodule.ORTMODULE_TORCH_CPP_DIR)

    # Extensions might leverage CUDA/ROCM versions internally
    os.environ["ONNXRUNTIME_CUDA_VERSION"] = ortmodule.ONNXRUNTIME_CUDA_VERSION \
        if not ortmodule.ONNXRUNTIME_CUDA_VERSION is None else ''
    os.environ["ONNXRUNTIME_ROCM_VERSION"] = ortmodule.ONNXRUNTIME_ROCM_VERSION \
        if not ortmodule.ONNXRUNTIME_ROCM_VERSION is None else ''

    ############################################################################
    # Pytorch CPP Extensions that DO require CUDA/ROCM
    ############################################################################
    if is_gpu_available:
        for ext_setup in _list_cuda_extensions():
            _install_extension(ext_setup.split(
                os.sep)[-2], ext_setup, ortmodule.ORTMODULE_TORCH_CPP_DIR)

    ############################################################################
    # Pytorch CPP Extensions that DO NOT require CUDA/ROCM
    ############################################################################
    for ext_setup in _list_cpu_extensions():
        _install_extension(ext_setup.split(
            os.sep)[-2], ext_setup, ortmodule.ORTMODULE_TORCH_CPP_DIR)

    ############################################################################
    # Install Pytorch CPP Extensions into local onnxruntime package folder
    ############################################################################
    torch_cpp_exts = glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR,
                                       'build',
                                       'lib.*',
                                       '*.so'))
    torch_cpp_exts.extend(glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR,
                                            'build',
                                            'lib.*',
                                            '*.dll')))
    torch_cpp_exts.extend(glob(os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR,
                                            'build',
                                            'lib.*',
                                            '*.dylib')))
    for ext in torch_cpp_exts:
        dest_ext = os.path.join(ortmodule.ORTMODULE_TORCH_CPP_DIR, os.path.basename(ext))
        print(f'Installing {ext} -> {dest_ext}')
        copyfile(ext, dest_ext)

    # Tear down
    os.environ.pop("ONNXRUNTIME_CUDA_VERSION")
    os.environ.pop("ONNXRUNTIME_ROCM_VERSION")


if __name__ == '__main__':
    build_torch_cpp_extensions()
