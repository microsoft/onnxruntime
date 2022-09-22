// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/optimizer/clip_grad_norm/clip_grad_norm.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace contrib {

namespace {

constexpr float Epsilon = 0.000001f;

template <typename T>
T GetL2Norm(const TensorSeq& gradients) {
  T l2_norm = 0;
  for (const auto& tensor : gradients) {
    l2_norm +=
        ReduceAggregatorSumSquare<T>(tensor.Shape().Size(), *tensor.Data<T>()).aggall(tensor.Data<T>());
  }
  return reduce_sqrt<T>(l2_norm);
}

template <typename T>
void ClipGradNorm(T total_norm, T max_norm, TensorSeq& gradients) {
  const T clip_coefficient = std::min(max_norm / (total_norm + static_cast<T>(Epsilon)), static_cast<T>(1.0f));

  for (const auto& grad : gradients) {
    auto& tensor = const_cast<Tensor&>(grad);
    MakeEigenArrayMap<T>(tensor) *= clip_coefficient;
  }
}

Status PopulateOutput(OpKernelContext* ctx, const TensorSeq* gradients, TensorSeq* clipped_gradients) {
  if (gradients == clipped_gradients) {
    return Status::OK();
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  clipped_gradients->SetType(gradients->DataType());
  clipped_gradients->Reserve(gradients->Size());
  for (const auto& grad : *gradients) {
    Tensor target_tensor(grad.DataType(), grad.Shape(), alloc);
    CopyCpuTensor(&grad, &target_tensor);
    clipped_gradients->Add(std::move(target_tensor));  // Add will check for type consistency
  }

  return Status::OK();
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    InplaceClipGradNorm,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0) /* Return updated gradients in-place */
        .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    InplaceClipGradNorm<float>);

template <typename T>
Status InplaceClipGradNorm<T>::Compute(OpKernelContext* ctx) const {
  const TensorSeq* gradients = ctx->Input<TensorSeq>(0);

  const T total_norm = GetL2Norm<T>(*gradients);

  auto grads = const_cast<TensorSeq*>(gradients);
  ClipGradNorm(total_norm, max_norm_, *grads);

  // Populate the output sequence tensors.
  TensorSeq* clipped_gradients = ctx->Output<TensorSeq>(0);
  ORT_RETURN_IF_ERROR(PopulateOutput(ctx, gradients, clipped_gradients));

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
