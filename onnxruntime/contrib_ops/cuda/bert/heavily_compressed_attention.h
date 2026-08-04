// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchHcaBlockBiasKernel(cudaStream_t stream, T* block_bias, const int64_t* position_ids,
                                int batch_size, int sequence_length, int entry_count,
                                int compress_rate, int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
