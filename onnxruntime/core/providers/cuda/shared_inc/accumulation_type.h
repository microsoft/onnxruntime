// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/framework/float16.h"

namespace onnxruntime {
namespace cuda {

// specifies the auxiliary type to use for accumulation of the given type
template <typename T>
struct AccumulationType;
template <>
struct AccumulationType<half> {
  using type = float;
};
template <>
struct AccumulationType<float> {
  using type = float;
};
template <>
struct AccumulationType<double> {
  using type = double;
};
template <>
struct AccumulationType<BFloat16> {
  using type = float;
};

template <typename T>
using AccumulationType_t = typename AccumulationType<T>::type;

}  // namespace cuda
}  // namespace onnxruntime
