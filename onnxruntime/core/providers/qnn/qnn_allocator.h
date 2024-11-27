// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/allocator.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

class QnnBackendManager;
class RpcMemLibrary;

class HtpSharedMemoryAllocator : public IAllocator {
 public:
  // Gets the single OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo MemoryInfo();

  HtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                           std::shared_ptr<QnnBackendManager> qnn_backend_manager);

  void* Alloc(size_t size) override;
  void* TensorAlloc(MLDataType element_data_type, const TensorShape& shape) override;
  void Free(void* p) override;
  // void GetStats(AllocatorStats* stats) override;

 private:
  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;
  std::shared_ptr<QnnBackendManager> qnn_backend_manager_;
};

}  // namespace onnxruntime::qnn
