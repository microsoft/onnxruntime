// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/matmul_ggml_iq2_xxs.h"

namespace onnxruntime::contrib::cuda {

constexpr int kIQ2XXSMaxSelectedExperts = 32;
constexpr size_t kIQ2XXSScratchBytes = size_t{16} << 20;

Status ValidateQMoEIQ2XXSRoutes(const int64_t* ids, const float* weights,
                                int64_t tokens, int64_t experts, int k,
                                int* error, cudaStream_t stream);

Status LaunchQMoEIQ2XXS(const void* input, const uint8_t* up, const uint8_t* down,
                        const int64_t* ids, const float* weights, void* output,
                        void* intermediate, int* order, int64_t tokens,
                        int64_t hidden, int64_t inter, int k,
                        IQ2XXSActivationType type, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
