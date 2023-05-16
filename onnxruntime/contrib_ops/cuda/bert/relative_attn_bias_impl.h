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
    const bool is_bidirectional,
    const int max_threads_per_block);

template <typename T>
Status LaunchGatedRelativePositionBiasKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    T* output,
    const T* rel_pos,
    const T* qw,  // from query * weight
    const T* bias,
    const T* eco_a,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int D,
    const int ldqw);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
