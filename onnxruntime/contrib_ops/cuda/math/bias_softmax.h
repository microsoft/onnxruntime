// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// BiasSoftmax follows the OpSet-11 definision of Softmax Op, that is, the input will be coerced to a 2D tensor
// using axis attribute, all dims after axis (included) are in the same batch. This is different from definition
// since OpSet-13. To use BiasSoftmax, during the fusion, if Softmax is OpSet-13 or newer, you can only fuse it
// when axis attribute is the last dim, othewise, the computation result may be wrong.
class BiasSoftmax final : public onnxruntime::cuda::CudaKernel {
 public:
  BiasSoftmax(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
    int64_t is_inner_broadcast_value;
    ORT_ENFORCE(info.GetAttr<int64_t>("is_inner_broadcast", &is_inner_broadcast_value).IsOK());
    is_inner_broadcast_ = is_inner_broadcast_value != 0;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool is_inner_broadcast_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
