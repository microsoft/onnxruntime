// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U>
void ScalerImpl(
    const T* input_data,
    const U* scale,
    U* output_data,
    size_t N);

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
