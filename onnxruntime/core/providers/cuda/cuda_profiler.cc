// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <map>
#include <string>
#include <iostream>

#include "cuda_profiler.h"
#include "core/common/logging/logging.h"

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

void CudaProfiler::StartRunProfiling() {
  LOGS_DEFAULT(WARNING) << "Run-level profiling is not implemented for this EP profiler, so GPU kernel events for this EP cannot be captured.";
}

#else

void CudaProfiler::StartRunProfiling() {
  LOGS_DEFAULT(WARNING) << "Run-level profiling is not implemented for this EP profiler, so GPU kernel events for this EP cannot be captured.";
}

#endif /* #if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */

}  // namespace profiling
}  // namespace onnxruntime
