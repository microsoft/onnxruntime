// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchRelPosAttnBiasKernel(
    cudaStream_t stream,
    T* output,
    const T* bias_table,
    const int num_heads,
    const int seq_len,
    const int num_bucket,
    const int max_distance,
    const bool is_bidirectional
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
