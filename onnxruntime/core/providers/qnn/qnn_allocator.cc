// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <limits>

#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

RpcMemAllocator::RpcMemAllocator(const RpcMemApi& rpc_mem_api)
    : IAllocator{OrtMemoryInfo{"TODO name the allocator", OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice{OrtDevice::CPU, OrtDevice::MemType::QNN_HTP_SHARED, /* device id */ 0},
                               0, OrtMemTypeCPUOutput}},
      rpc_mem_api_{rpc_mem_api} {
}

void* RpcMemAllocator::Alloc(size_t size) {
  // rpcmem_alloc() has an int size parameter.
  constexpr size_t max_size = std::numeric_limits<int>::max();
  if (size > max_size) {
    return nullptr;
  }

  return rpc_mem_api_.alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM, rpcmem::RPCMEM_DEFAULT_FLAGS,
                            static_cast<int>(size));
}

void RpcMemAllocator::Free(void* p) {
  rpc_mem_api_.free(p);
}

}  // namespace onnxruntime::qnn
