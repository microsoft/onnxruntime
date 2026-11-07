// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace onnxruntime::contrib::cuda {

cudaError_t LaunchMoEPacked(cudaStream_t stream, int rows, int topk, int experts, int intermediate,
                            const void* input, const int32_t* indices, const float* weights,
                            const uint8_t* fc1, const uint8_t* fc2, void* output, void* hidden,
                            float* expert_output, int32_t* metadata);

cudaError_t LaunchMoEPackedRouter(cudaStream_t stream, int rows, int width, int experts, int topk,
                                  float scale, const void* input, const void* weight, const float* bias,
                                  int32_t* indices, float* weights, float* scores);

}  // namespace onnxruntime::contrib::cuda
