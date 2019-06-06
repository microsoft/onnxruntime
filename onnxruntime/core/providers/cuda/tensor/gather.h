// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/gather.h"

namespace onnxruntime {
namespace cuda {

class Gather final : public CudaKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : GatherBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

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
