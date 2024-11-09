// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <limits>

#include "core/common/common.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

OrtMemoryInfo RpcMemAllocator::MemoryInfo() {
  return OrtMemoryInfo{QNN_HTP_SHARED, OrtAllocatorType::OrtDeviceAllocator,
                       OrtDevice{OrtDevice::CPU, OrtDevice::MemType::QNN_HTP_SHARED, /* device_id */ 0},
                       /* id */ 0, OrtMemTypeDefault};
}

RpcMemAllocator::RpcMemAllocator(std::shared_ptr<RpcMemLibrary> rpc_mem_lib)
    : IAllocator{MemoryInfo()},
      rpc_mem_lib_{std::move(rpc_mem_lib)} {
  ORT_ENFORCE(rpc_mem_lib_ != nullptr, "rpc_mem_lib_ must not be nullptr");
}

void* RpcMemAllocator::Alloc(size_t size) {
  // rpcmem_alloc() has an int size parameter.
  constexpr size_t max_size = std::numeric_limits<int>::max();
  if (size > max_size) {
    return nullptr;
  }

  return rpc_mem_lib_->Api().alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM, rpcmem::RPCMEM_DEFAULT_FLAGS,
                                   static_cast<int>(size));
}

void RpcMemAllocator::Free(void* p) {
  rpc_mem_lib_->Api().free(p);
}

}  // namespace onnxruntime::qnn
