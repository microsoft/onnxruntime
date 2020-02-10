// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/hip_utils.h"

namespace onnxruntime {
namespace hip {

template <typename T, typename Tin>
void GatherImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod* output_block_size,
    const fast_divmod* block_size,
    const Tin* indices_data,
    const T* input_data,
    T* output_data,
    const size_t N);

}  // namespace hip
}  // namespace onnxruntime
