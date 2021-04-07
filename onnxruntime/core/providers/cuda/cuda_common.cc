// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"

namespace onnxruntime {
namespace cuda {

// Initialize the singleton instance
HalfGemmOptions HalfGemmOptions::instance;

const HalfGemmOptions* HalfGemmOptions::GetInstance() {
  if (!instance.initialized_) {
    // We do not use critical section here since it is fine to initialize multiple times by different threads.
    int value = ParseEnvironmentVariableWithDefault<int>(kCudaGemmOptions, 0);
    instance.Initialize(value);
  }

  return &instance;
}

void HalfGemmOptions::Initialize(int value)
{
    compute_16f_ = (value & 0x01) > 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    disallow_reduced_precision_reduction_ = (value & 0x02) > 0;
    pedantic_ = (value & 0x04) > 0;
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_
                       << " disallow_reduced_precision_reduction=" << instance.disallow_reduced_precision_reduction_
                       << " pedantic=" << instance.pedantic_;
#else
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_;
#endif
    initialized_ = true;
}

}  // namespace cuda
}  // namespace onnxruntime
