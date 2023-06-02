// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class QuantNbitsLinear final : public OpKernel {
 public:
  explicit QuantNbitsLinear(const OpKernelInfo& info) : OpKernel{info} {
    //ORT_ENFORCE(info.GetAttr("outfeatures", &outfeatures_).IsOK());
    //ORT_ENFORCE(info.GetAttr("infeatures", &in_features_).IsOK());
    bits_ = info.GetAttrOrDefault<int64_t>("bits", 3);
    groupsize_ = info.GetAttrOrDefault<int64_t>("groupsize", 128);
  }

  Status Compute(OpKernelContext* context) const override;

 private:

  template <typename T>
  struct ComputeImpl;

  int64_t outfeatures_;
  int64_t in_features_;
  int64_t bits_;
  int64_t groupsize_;
};

ONNX_OPERATOR_KERNEL_EX(
    QuantNbitsLinear,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()),
    QuantNbitsLinear);

int iii=0;
Status QuantNbitsLinear::Compute(OpKernelContext* ctx) const {
  const auto* input_x = ctx->Input<Tensor>(0);
  const auto* input_weight = ctx->Input<Tensor>(1);
  //const auto* input_scale = ctx->Input<Tensor>(2);
  const auto* input_zeros = ctx->Input<Tensor>(3);
  //const auto* input_bias = ctx->Input<Tensor>(4);
  //const auto* input_gidx = ctx->Input<Tensor>(5);
  const auto& input_shape = input_x->Shape();
  const auto& weight_shape = input_weight->Shape();
  TensorShapeVector output_shape = input_shape.AsShapeVector();
  output_shape[output_shape.size() - 1] = weight_shape[1];
  auto* output = ctx->Output(0, output_shape);
  auto batch = input_shape[0] * (input_shape.NumDimensions() > 2 ? input_shape[1] : 1);
  //int64_t in_features = input_shape[input_shape.NumDimensions() - 1];
  input_x->Data<MLFloat16>();
  //auto *outp=output->Data<MLFloat16>();
  //input_scale->Data<MLFloat16>();
  printf("%zu,%zu\n", batch, output->Shape()[1]);

  size_t sz = weight_shape[0] * weight_shape[1]*2;
  std::vector<int32_t> buf(sz);
  printf("%d...%d,", input_weight->Data<int32_t>()[0], input_zeros->Data<int32_t>()[0]);


  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
