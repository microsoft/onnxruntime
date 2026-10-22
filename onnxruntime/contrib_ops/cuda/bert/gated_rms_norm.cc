// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_rms_norm.h"

#include <cmath>
#include <limits>

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatedRMSNormExact, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>()),
    GatedRMSNormExact);

GatedRMSNormExact::GatedRMSNormExact(const OpKernelInfo& info) : CudaKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
  ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ >= 0, "epsilon must be finite and nonnegative");
}

Status GatedRMSNormExact::ComputeInternal(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* gate = context->Input<Tensor>(1);
  const auto* scale = context->Input<Tensor>(2);
  const auto& shape = x->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[1] == 128 &&
                        shape[0] <= std::numeric_limits<int>::max(),
                    "GatedRMSNorm X must have shape [rows,128] with rows fitting int32");
  ORT_RETURN_IF_NOT(gate->Shape() == shape && scale->Shape() == TensorShape({128}),
                    "GatedRMSNorm gate must match X and scale must have shape [128]");
  auto* output = context->Output(0, shape);
  if (shape[0] == 0) return Status::OK();
  return LaunchGatedRMSNorm(Stream(context), x->DataRaw(), gate->DataRaw(), scale->Data<float>(),
                            output->MutableDataRaw(), static_cast<int>(shape[0]), epsilon_);
}

}  // namespace onnxruntime::contrib::cuda
