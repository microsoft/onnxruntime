// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/allocator.h"

namespace onnxruntime::qnn {

class RpcMemLibrary;

class RpcMemAllocator : public IAllocator {
 public:
  // Gets the single OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo MemoryInfo();

  RpcMemAllocator(std::shared_ptr<RpcMemLibrary> rpc_mem_lib);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  // void GetStats(AllocatorStats* stats) override;

 private:
  std::shared_ptr<RpcMemLibrary> rpc_mem_lib_;
};

}  // namespace onnxruntime::qnn
