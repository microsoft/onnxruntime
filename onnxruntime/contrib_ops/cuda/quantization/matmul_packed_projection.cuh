// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace onnxruntime::contrib::cuda {

enum class PackedProjectionFormat { NVFP4,
                                    Q8,
                                    IQ2XXS };

// BF16 activations/output; scales retain their source representation.
cudaError_t LaunchPackedProjection(PackedProjectionFormat format, const void* a, const uint8_t* b, const void* scales,
                                   const float* global_scale, bool fp16_scales, int rows, int n, int k, void* y,
                                   cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
