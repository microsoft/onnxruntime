// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class GatherGrad final : public CudaKernel {
 public:
  GatherGrad(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
