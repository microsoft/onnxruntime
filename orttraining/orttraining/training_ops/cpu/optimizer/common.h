// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <cmath>

namespace onnxruntime {
namespace contrib {

template <typename TC, typename TS>
TC compute_bias_correction_coefficient(const TC momentum_update_coefficient, const TS step) {
  if (step > 0) {
    return TC(1.0 - std::pow(static_cast<double>(momentum_update_coefficient), static_cast<double>(step)));
  } else {
    return TC(1.f);
  }
}

Status CopyIfNotSameCPUBuffer(OpKernelContext* ctx, size_t number_of_values, const TensorSeq* src_values,
                              TensorSeq* dest_values);

}  // namespace contrib
}  // namespace onnxruntime
