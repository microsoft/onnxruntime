// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "opencl_utils.h"

#include <unordered_map>
#include <list>
#include <variant>

namespace onnxruntime {
namespace opencl {

#define ALLOCATOR_NAME "OpenCL"

enum MemoryKind: uint8_t {
  Buffer = 0,
  Image2D = 1,
};

struct OpenCLFatPtr {
  std::variant<cl::Buffer, cl::Image2D> ptr;
  size_t size;
  MemoryKind kind;
};

class OpenCLAllocator : public IAllocator {
 public:


  explicit OpenCLAllocator(const cl::Context& ctx);
  ~OpenCLAllocator() override;

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;

 private:
  cl::Context ctx_;
  // FIXME: better caching, cache for kernel benchmark at the moment
  std::unordered_map<void*, size_t> ptr_to_size_;
  std::unordered_map<size_t, std::list<void*>> cache_;
};

}  // namespace opencl
}  // namespace onnxruntime
