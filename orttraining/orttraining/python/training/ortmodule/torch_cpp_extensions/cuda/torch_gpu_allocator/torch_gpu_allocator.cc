// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <c10/___gpu_identifier___/___gpu_allocator_header___.h>

size_t gpu_caching_allocator_raw_alloc_address() {
  return reinterpret_cast<size_t>(&c10::___gpu_identifier___::___gpu_allocator_header___::raw_alloc);
}

size_t gpu_caching_allocator_raw_delete_address() {
  return reinterpret_cast<size_t>(&c10::___gpu_identifier___::___gpu_allocator_header___::raw_delete);
}

size_t gpu_caching_allocator_empty_cache_address() {
  return reinterpret_cast<size_t>(&c10::___gpu_identifier___::___gpu_allocator_header___::emptyCache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_caching_allocator_raw_alloc_address", &gpu_caching_allocator_raw_alloc_address,
        "Address of PyTorch GPU allocator");
  m.def("gpu_caching_allocator_raw_delete_address", &gpu_caching_allocator_raw_delete_address,
        "Address of PyTorch GPU deallocator");
  m.def("gpu_caching_allocator_empty_cache_address", &gpu_caching_allocator_empty_cache_address,
        "Address of PyTorch GPU empty cache");
}
