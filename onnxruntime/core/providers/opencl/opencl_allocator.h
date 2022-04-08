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
constexpr auto Image2DAllocatorName = "OpenCL_Image2D";
constexpr auto CPUAllocatorName = "OpenCL_CPU";
constexpr auto CPUInputAllocatorName = "OpenCL_CPU_Input";

template <typename InfoT, typename CreatorT>
class CachingPolicy {
  using InfoType = InfoT;
  using CreatorType = CreatorT;

 public:
  virtual ~CachingPolicy() = default;
  virtual void* CreateOrGetFromCache(OpenCLExecutionProvider* ctx, InfoType info) = 0;
  virtual void DestroyOrReturnToCache(void*) = 0;
  virtual void EvictAllCache() = 0;
};

struct BufferCreator;
struct Image2DCreator;

class OpenCLBufferAllocator : public IAllocator {
 public:
  explicit OpenCLBufferAllocator(OpenCLExecutionProvider& exec);
  ~OpenCLBufferAllocator() = default;

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  OpenCLExecutionProvider* exec_;
  std::unique_ptr<CachingPolicy<size_t, BufferCreator>> caching_;
};

class OpenCLImage2DAllocator : public IAllocator {
 public:
  explicit OpenCLImage2DAllocator(OpenCLExecutionProvider& exec);
  ~OpenCLImage2DAllocator() = default;

  void* Alloc(size_t size) override;
  void* Alloc(const TensorShape& shape) override;
  void* Alloc(Image2DDesc desc);
  void Free(void* p) override;

 private:
  OpenCLExecutionProvider* exec_;
  std::unique_ptr<CachingPolicy<Image2DDesc, Image2DCreator>> caching_;
};

}  // namespace opencl
}  // namespace onnxruntime
