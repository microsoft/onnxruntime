// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "opencl_allocator.h"

#include <iostream>

namespace onnxruntime {
namespace opencl {

OpenCLAllocator::OpenCLAllocator(const cl::Context& ctx)
    : IAllocator(OrtMemoryInfo(DeviceAllocatorName, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtMemType::OrtMemTypeDefault, /*TODO: used true id here*/ 0))),
      ctx_(ctx) {
}

OpenCLAllocator::~OpenCLAllocator() {
  for (auto& [k, v] : cache_) {
    for (auto* p : v) {
      if (p != nullptr) {
        delete reinterpret_cast<cl::Buffer*>(p);
      }
    }
  }
}

void* OpenCLAllocator::Alloc(size_t size) {
  auto it = cache_.find(size);

  if (it == cache_.end() || it->second.empty()) {
    cl_int err{};
    auto* ptr = new cl::Buffer(ctx_, CL_MEM_READ_WRITE, size, nullptr, &err);
    OPENCL_CHECK_ERROR(err);
    std::cerr << "OpenCLAllocator allocated " << ptr << " --> cl::Buffer(" << ptr->operator()() << ") in cl::Context(" << ctx_() << ")" << std::endl;
    meta_[ptr] = {size, MemoryKind::Buffer};
    return ptr;
  }

  auto* ptr = it->second.front();
  std::cerr << "OpenCLAllocator reused " << ptr << std::endl;
  it->second.pop_front();
  return ptr;
}

void OpenCLAllocator::Free(void* p) {
  auto meta = meta_[p];
  auto it = cache_.find(meta.size);
  if (it == cache_.end()) {
    it = cache_.insert({meta.size, {}}).first;
  }
  it->second.push_front(p);
}

}  // namespace opencl
}  // namespace onnxruntime
