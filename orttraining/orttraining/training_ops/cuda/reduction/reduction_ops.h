// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ReduceSumTraining final : public onnxruntime::contrib::ATenOp {
 public:
  ReduceSumTraining(const OpKernelInfo& info) : ATenOp(info) {
    // fast_reduction_ = true;
  }
};

}  // namespace cuda
}  // namespace onnxruntime
