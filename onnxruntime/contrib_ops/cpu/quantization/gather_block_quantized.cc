// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/int4.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

class GatherBlockQuantized : public OpKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("gather_axis", &gather_axis_).IsOK()) {
      gather_axis_ = 0;
    }

    if (!info.GetAttr<int64_t>("quantize_axis", &quantize_axis_).IsOK()) {
      quantize_axis_ = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 128;
    }

    ORT_ENFORCE(block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0,
                "'block_size' must be 2's power and not less than 16.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
};

Status GatherBlockQuantized::Compute(OpKernelContext* context) const {

}

ONNX_OPERATOR_KERNEL_EX(
    GatherBlockQuantized,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<UInt4x2>(), DataTypeImpl::GetTensorType<Int4x2>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    GatherBlockQuantized);

}  // namespace contrib
}  // namespace onnxruntime
