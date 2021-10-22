// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void EyeLikeImpl(
    cudaStream_t stream,
    size_t offset,                  // offset of first element in diagnal
    size_t stripe,                  // stripe, here it's width + 1
    T* output_data,                 // output buffer
    size_t diag_count               // total number of elements in diagnal
);

}  // namespace cuda
}  // namespace onnxruntime
