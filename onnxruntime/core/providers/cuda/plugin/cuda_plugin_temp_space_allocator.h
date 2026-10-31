// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda_plugin {

// Allocator that kernels use for temporary buffers. It has to share the session's budget account: without one the
// allocator creates its own unlimited account, so temporary tensors would not count against the session's local or
// group limit.
inline AllocatorPtr CreateCudaPluginTempSpaceAllocator(int device_id, std::shared_ptr<SessionBudgetAccount> account) {
  return std::make_shared<::onnxruntime::CUDAAllocator>(device_id, ::onnxruntime::CUDA, std::move(account));
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
