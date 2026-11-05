// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace onnxruntime::contrib::cuda {
// Out-of-range or duplicate IDs invalidate the entire token row: Y/hidden/expert_output are NaN.
cudaError_t LaunchMoEPackedIQ2XXS(cudaStream_t stream, int rows, int topk, int experts, int intermediate_size,
                                  const void* input, const int32_t* indices, const float* weights, const uint8_t* fc1,
                                  const void* scale1, const uint8_t* fc2, const void* scale2, void* output,
                                  void* hidden, float* expert_output, int32_t* metadata);
}  // namespace onnxruntime::contrib::cuda
