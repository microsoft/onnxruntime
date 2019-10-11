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
    const int64_t k,
    const fast_divmod& fdm_x,
    T* output_data,
    size_t count
);

}  // namespace cuda
}  // namespace onnxruntime
