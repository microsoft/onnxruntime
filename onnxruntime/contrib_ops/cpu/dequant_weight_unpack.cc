// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdio>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"


namespace onnxruntime {
namespace contrib {

class DequantizeAndUnpackWeight final : public OpKernel {
 public:
  explicit DequantizeAndUnpackWeight(const OpKernelInfo& info) : OpKernel{info} {
    bits_ = info.GetAttrOrDefault<int64_t>("bits", 4);
    groupsize_ = info.GetAttrOrDefault<int64_t>("groupsize", 128);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8, "bits must be 4 or 8");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t bits_;
  int64_t groupsize_;
};

ONNX_OPERATOR_KERNEL_EX(
    DequantizeAndUnpackWeight,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<uint32_t, int32_t>()),
    DequantizeAndUnpackWeight);

void DequantNbitWeight(OpKernelContext* ctx, const Tensor* input_weight, Tensor* output, const Tensor* input_zeros,
                       const Tensor* input_scale, const int64_t bits_, const int64_t compress_ratio,
                       const int64_t groupsize_);

Status DequantizeAndUnpackWeight::Compute(OpKernelContext* ctx) const {
  const auto* input_weight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  // const auto* input_gidx = ctx->Input<Tensor>(5);
  const auto& qweight_shape = input_weight->Shape();
  const int64_t compress_ratio = sizeof(int32_t)*8 / bits_;
  TensorShape output_shape = qweight_shape;
  output_shape[0] = output_shape[0] * compress_ratio;
  auto* output = ctx->Output(0, output_shape);
  DequantNbitWeight(ctx, input_weight, output, input_zeros, input_scale, bits_, compress_ratio, groupsize_);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
