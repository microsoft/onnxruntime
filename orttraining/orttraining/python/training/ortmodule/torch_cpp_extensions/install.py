# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi import build_and_package_info as ort_info

import os
import subprocess
import sys

from glob import glob
from shutil import copyfile


def build_torch_cpp_extensions():
    '''Builds PyTorch CPP extensions and returns metadata'''

    cuda_version = ort_info.cuda_version if hasattr(ort_info, 'cuda_version') else None
    rocm_version = ort_info.rocm_version if hasattr(ort_info, 'rocm_version') else None

    # Run this from within onnxruntime package folder
    is_gpu_available = cuda_version is not None or rocm_version is not None
    cpp_ext_dir = os.path.join(os.path.dirname(__file__))
    os.chdir(cpp_ext_dir)

    ############################################################################
    # Pytorch CPP Extensions that DO require CUDA/ROCM
    ############################################################################
    if is_gpu_available:
        setup_script = os.path.join(cpp_ext_dir,
                                    'torch_gpu_allocator',
                                    'setup.py')
        version = '--use_rocm' if rocm_version else ''
        ret_code = subprocess.call(f"{sys.executable} {setup_script} build {version}",
                                   cwd=cpp_ext_dir,
                                   shell=True)
        if ret_code != 0:
            print('There was an error compiling "torch_gpu_allocator" PyTorch CPP extension')
            sys.exit(ret_code)

    ############################################################################
    # Pytorch CPP Extensions that DO NOT require CUDA/ROCM
    ############################################################################
    setup_script = os.path.join(cpp_ext_dir,
                                'aten_op_executor',
                                'setup.py')
    ret_code = subprocess.call(f"{sys.executable} {setup_script} build",
                               cwd=cpp_ext_dir,
                               shell=True)
    if ret_code != 0:
        print('There was an error compiling "aten_op_executor" PyTorch CPP extension')
        sys.exit(ret_code)

    setup_script = os.path.join(cpp_ext_dir,
                                'torch_interop_utils',
                                'setup.py')
    ret_code = subprocess.call(f"{sys.executable} {setup_script} build",
                               cwd=cpp_ext_dir,
                               shell=True)
    if ret_code != 0:
        print('There was an error compiling "torch_interop_utils" PyTorch CPP extension')
        sys.exit(ret_code)

    ############################################################################
    # Copy Pytorch CPP Extensions to the local onnxruntime package folder
    ############################################################################
    torch_cpp_exts = glob(os.path.join(cpp_ext_dir,
                                       'build',
                                       'lib.*',
                                       '*.so'))
    torch_cpp_exts.extend(glob(os.path.join(cpp_ext_dir,
                                            'build',
                                            'lib.*',
                                            '*.dll')))
    torch_cpp_exts.extend(glob(os.path.join(cpp_ext_dir,
                                            'build',
                                            'lib.*',
                                            '*.dylib')))
    for ext in torch_cpp_exts:
        dest_ext = os.path.join(cpp_ext_dir, os.path.basename(ext))
        print(f'Installing {ext} -> {dest_ext}')
        copyfile(ext, dest_ext)


if __name__ == '__main__':
    build_torch_cpp_extensions()
