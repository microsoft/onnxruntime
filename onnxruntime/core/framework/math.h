// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/narrow.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
auto EigenMap(Tensor& t) -> EigenVectorMap<T> {
  return EigenVectorMap<T>(t.MutableData<T>(), narrow<ptrdiff_t>(t.Shape().Size()));
}

template <typename T>
auto EigenMap(const Tensor& t) -> ConstEigenVectorMap<T> {
  return ConstEigenVectorMap<T>(t.Data<T>(), narrow<ptrdiff_t>(t.Shape().Size()));
}

}  // namespace onnxruntime
