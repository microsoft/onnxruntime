// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status TransposeImpl(size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int64_t N);
template <typename T>
void BatchTranspose2DImpl(
    int64_t N,
    int64_t H,
    int64_t W,
    const T* X,
    T* Y);

// If perm_1 is true, it will perform transpose with perm (0, 2, 1, 3)
// otherwise it will perform transpose with perm (0, 2, 3, 1)
template <typename T>
void BatchTranspose3DImpl(
    int64_t N,
    int64_t M,
    int64_t L,
    int64_t K,
    const T* X,
    T* Y,
    bool perm_1);
}  // namespace cuda
}  // namespace onnxruntime
