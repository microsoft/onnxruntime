// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>

namespace onnxruntime {
namespace cuda {

// specifies the auxiliary type to use for accumulation of the given type
template <typename T>
struct AccumulationType;
template <>
struct AccumulationType<half> { using type = float; };
template <>
struct AccumulationType<float> { using type = float; };
template <>
struct AccumulationType<double> { using type = double; };
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
struct AccumulationType<nv_bfloat16> { using type = float; };
#endif

template <typename T>
using AccumulationType_t = typename AccumulationType<T>::type;

}  // namespace cuda
}  // namespace onnxruntime
