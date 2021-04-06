// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Initialize the singleton instance
HalfGemmOptions HalfGemmOptions::instance;

const HalfGemmOptions* HalfGemmOptions::GetInstance() {
  if (!instance.initialized_) {
    // We do not use critical section here since it is fine to initialize multiple times by different threads.

    // The environment variable is for testing purpose only, and it might be removed in the future.
    constexpr const char* kCudaGemmOptions = "ORT_CUDA_GEMM_OPTIONS";
    int value = ParseEnvironmentVariableWithDefault<int>(kCudaGemmOptions, 0);
    instance.compute_16f_ = (value & 0x01) > 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    instance.disallow_reduced_precision_reduction_ = (value & 0x02) > 0;
    instance.pedantic_ = (value & 0x04) > 0;
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_
                       << " disallow_reduced_precision_reduction=" << instance.disallow_reduced_precision_reduction_
                       << " pedantic=" << instance.pedantic_;
#else
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_;
#endif
    instance.initialized_ = true;
  }

  return &instance;
}

}  // namespace cuda
}  // namespace onnxruntime
