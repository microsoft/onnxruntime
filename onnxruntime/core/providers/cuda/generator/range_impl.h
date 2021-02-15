// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {


template <typename T>
bool RangeImpl(cudaStream_t stream, const T start, const T delta, const int count, T* output);

}  // namespace cuda
}  // namespace onnxruntime
