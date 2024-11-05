// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime::qnn {

struct RpcMemApi;

class RpcMemAllocator : public IAllocator {
 public:
  RpcMemAllocator(const RpcMemApi& rpc_mem_api);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  // void GetStats(AllocatorStats* stats) override;

 private:
  const RpcMemApi& rpc_mem_api_;
};

}  // namespace onnxruntime::qnn
