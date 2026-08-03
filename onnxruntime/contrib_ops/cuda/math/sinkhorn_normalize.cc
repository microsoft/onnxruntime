// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/sinkhorn_normalize.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/sinkhorn_normalize_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SinkhornNormalize, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SinkhornNormalize);

SinkhornNormalize::SinkhornNormalize(const OpKernelInfo& info) : CudaKernel(info) {
  iterations_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("iterations", static_cast<int64_t>(1)));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
  ORT_ENFORCE(iterations_ >= 1, "iterations must be at least 1, got ", iterations_);
}

Status SinkhornNormalize::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const TensorShape& shape = input->Shape();
  const auto& dims = shape.GetDims();

  if (dims.size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input is expected to have at least 2 dimensions, got ", dims.size());
  }

  const int64_t order = dims[dims.size() - 1];
  if (dims[dims.size() - 2] != order) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The last two dimensions must be equal, got ",
                           dims[dims.size() - 2], " and ", order);
  }
  if (order < 1 || order > kSinkhornMaxOrder) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The matrix order must be in [1, ", kSinkhornMaxOrder, "], got ", order);
  }

  Tensor* output = context->Output(0, shape);
  const int64_t num_matrices = shape.Size() / (order * order);

  return LaunchSinkhornNormalize(Stream(context),
                                 input->Data<float>(),
                                 output->MutableData<float>(),
                                 static_cast<int>(num_matrices),
                                 static_cast<int>(order),
                                 iterations_,
                                 epsilon_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
