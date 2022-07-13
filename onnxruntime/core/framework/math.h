// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
auto EigenMap(Tensor& t) -> EigenVectorMap<T> {
  return EigenVectorMap<T>(t.template MutableData<T>(), gsl::narrow<ptrdiff_t>(t.Shape().Size()));
}

template <typename T>
auto EigenMap(const Tensor& t) -> ConstEigenVectorMap<T> {
  return ConstEigenVectorMap<T>(t.template Data<T>(), gsl::narrow<ptrdiff_t>(t.Shape().Size()));
}

}  // namespace onnxruntime
