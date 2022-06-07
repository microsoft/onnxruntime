// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

int NonZeroCalcBlockCount(int64_t x_size);

cudaError_t NonZeroCalcPrefixSumTempStorageBytes(cudaStream_t stream, int* prefix_counts, int number_of_blocks, size_t& );

cudaError_t NonZeroInclusivePrefixSum(cudaStream_t stream, void* d_temp_storage, size_t temp_storage_bytes, int* prefix_counts, int number_of_blocks);

// count nonzero elements in each block into counts_in_blocks, 
// the counts_in_blocks buffer is pre-allocated on gpu first.
template<typename InputT>
cudaError_t NonZeroCountEachBlock(cudaStream_t stream, const InputT* x, int64_t x_size, int* counts_in_blocks);

// output nonzero positions using input x and prefix_counts for each blocks
template<typename InputT>
cudaError_t NonZeroOutputPositions(
    cudaStream_t stream, const InputT *x, int64_t x_size, int x_rank, const TArray<fast_divmod>& x_strides,
    const int* prefix_counts, int nonzero_elements, int64_t* results);

}  // namespace cuda
}  // namespace onnxruntime

