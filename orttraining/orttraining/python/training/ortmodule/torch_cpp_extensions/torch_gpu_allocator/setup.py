from setuptools import setup, Extension
import sys
from torch.utils import cpp_extension


def parse_arg_remove_boolean(argv, arg_name):
    arg_value = False
    if arg_name in sys.argv:
        arg_value = True
        argv.remove(arg_name)

    return arg_value

use_rocm = True if parse_arg_remove_boolean(sys.argv, '--use_rocm') else False
gpu_identifier = "hip" if use_rocm else "cuda"
gpu_allocator_header = "HIPCachingAllocator" if use_rocm else "CUDACachingAllocator"

torch_gpu_allocator_addresses_cpp_source = f'''
#include <torch/extension.h>
#include <c10/{gpu_identifier}/{gpu_allocator_header}.h>

size_t gpu_caching_allocator_raw_alloc_address() {{
    return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_alloc);
}}

size_t gpu_caching_allocator_raw_delete_address() {{
    return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_delete);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("gpu_caching_allocator_raw_alloc_address", &gpu_caching_allocator_raw_alloc_address, "LLTM forward");
    m.def("gpu_caching_allocator_raw_delete_address", &gpu_caching_allocator_raw_delete_address, "LLTM backward");
}}

'''

with open('torch_gpu_allocator.cpp', 'w') as f:
    f.write(f"{torch_gpu_allocator_addresses_cpp_source}\n")

setup(name='torch_gpu_allocator',
      ext_modules=[cpp_extension.CppExtension('torch_gpu_allocator', ['torch_gpu_allocator.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
