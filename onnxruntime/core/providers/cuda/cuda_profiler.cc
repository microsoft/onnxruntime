// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_profiler.h"

#include <iostream>
#include <map>
#include <string>

namespace onnxruntime {
namespace profiling {

#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

CudaProfiler::CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

CudaProfiler::~CudaProfiler() {
  auto& manager = CUPTIManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

#endif /* #if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */

}  // namespace profiling
}  // namespace onnxruntime
