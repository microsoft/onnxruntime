// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
class DivGrad : public ReduceKernel<true> {  // TODO: not to derive from ReduceKernel.
                                             // Use a cudnn reduce sum simple helper instead.
 public:
  DivGrad(const OpKernelInfo& info) : ReduceKernel<true>(info, /*keep_dims_override*/ int64_t(0)) {}
  Status ComputeInternal(OpKernelContext*) const override;
};
}  // namespace cuda
}  // namespace onnxruntime
