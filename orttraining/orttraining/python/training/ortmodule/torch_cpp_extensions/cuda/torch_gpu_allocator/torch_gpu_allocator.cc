// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <c10/___gpu_identifier___/___gpu_allocator_header___.h>
#include <torch/extension.h>

void* delegate_raw_alloc(size_t nbytes) {
  auto allocator = c10::___gpu_identifier___::___gpu_allocator_header___::get();
  return allocator->raw_allocate(nbytes);
}

void delegate_raw_delete(void* ptr) {
  auto allocator = c10::___gpu_identifier___::___gpu_allocator_header___::get();
  allocator->raw_deallocate(ptr);
}

size_t gpu_caching_allocator_raw_alloc_address() {
  return reinterpret_cast<size_t>(&delegate_raw_alloc);
}

size_t gpu_caching_allocator_raw_delete_address() {
  return reinterpret_cast<size_t>(&delegate_raw_delete);
}

size_t gpu_caching_allocator_empty_cache_address() {
  // This is useful only if PYTORCH_NO_CUDA_MEMORY_CACHING=1 is not set.
  return reinterpret_cast<size_t>(&c10::cuda::CUDACachingAllocator::emptyCache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_caching_allocator_raw_alloc_address", &gpu_caching_allocator_raw_alloc_address,
        "Address of PyTorch GPU allocator");
  m.def("gpu_caching_allocator_raw_delete_address", &gpu_caching_allocator_raw_delete_address,
        "Address of PyTorch GPU deallocator");
  m.def("gpu_caching_allocator_empty_cache_address", &gpu_caching_allocator_empty_cache_address,
        "Address of PyTorch GPU empty cache");
}
