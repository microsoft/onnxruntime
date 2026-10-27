// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchGatedAddKernel(cudaStream_t stream,
                            T* output,
                            const T* x,
                            const T* y,
                            const T* gate,
                            int64_t count,
                            int64_t hidden_size);

Status LaunchProjectedSigmoidGateMul(cudaStream_t stream,
                                     const uint16_t* attention,
                                     const uint16_t* activation,
                                     const uint16_t* weight,
                                     const uint16_t* residual,
                                     uint16_t* output,
                                     int64_t rows,
                                     int64_t input_width,
                                     int64_t output_width);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime