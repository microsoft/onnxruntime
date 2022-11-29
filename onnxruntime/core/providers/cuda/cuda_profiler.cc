// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include <map>
#include <string>
#include <iostream>

#include "cuda_profiler.h"

namespace onnxruntime {
namespace profiling {

CudaProfiler::CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

CudaProfiler::~CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

} // namespace onnxruntime
} // namespace profiling

#endif // defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000
