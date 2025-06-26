// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

#include <unordered_map>
#include <list>
#include <functional>
#include <variant>
#include "core/providers/opencl/opencl_execution_provider.h"
#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {

constexpr auto BufferAllocatorName = "OpenCL_Buffer";
constexpr auto CPUAllocatorName = "OpenCL_CPU";
constexpr auto CPUInputAllocatorName = "OpenCL_CPU_Input";

class OpenCLAllocator : public IAllocator {
 public:
  OpenCLAllocator(cl_context context)
      : IAllocator(
            OrtMemoryInfo( BufferAllocatorName, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
                          0, OrtMemTypeDefault)),
        context_(context) {}

  void* Alloc(size_t size) override {
    cl_int err;
    void* p = nullptr;
    if (size > 0) {
      p = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);
      if (err != CL_SUCCESS || p == nullptr) {
        throw std::runtime_error("OpenCL memory allocation failed");
      }
    }
    return p;
  }

  void Free(void* p) override {
    if (p != nullptr) {
      cl_mem buffer = static_cast<cl_mem>(p);
      clReleaseMemObject(buffer);
    }
  }

 private:
  cl_context context_;
};




}  // namespace opencl
}  // namespace onnxruntime
