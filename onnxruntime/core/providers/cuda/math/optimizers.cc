// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optimizers.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

template <>
Status BinaryElementwiseInplace<ShouldBroadcastInplace>::Prepare(OpKernelContext* context, int device_id, BinaryElementwisePreparation* p) const {
  auto lhs_tensor = context->Input<Tensor>(inout_index_);
  auto rhs_tensor = context->Input<Tensor>(input_index_);
  const auto& lhs_shape = lhs_tensor->Shape();

  TensorShape output_shape = lhs_shape;
  auto output_tensor = context->MutableInput<Tensor>(inout_index_);

  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(device_id, lhs_tensor, rhs_tensor, output_tensor, p));
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    SGDOptimizer);

Status SGDOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* weights_tensor = ctx->MutableInput<Tensor>(1);
  ORT_ENFORCE(weights_tensor);
  auto& weight_dimensions = weights_tensor->Shape().GetDims();

  const Tensor* gradients_tensor = ctx->Input<Tensor>(2);
  ORT_ENFORCE(gradients_tensor);
  auto& gradient_dimensions = gradients_tensor->Shape().GetDims();

  ORT_ENFORCE(weight_dimensions == gradient_dimensions);

  const Tensor* eta_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(eta_tensor);

  BinaryElementwisePreparation prepare(this);
  SetInOutIndexBeforePrepare(2, 0);
  Prepare(ctx, ctx->GetDeviceId(), &prepare);
  ORT_RETURN_IF_ERROR(prepare.CopyToGpu());

  // gradients = gradients * eta(scalar);
  Impl_Mul<typename ToCudaType<float>::MappedType>(
      prepare.output_rank_or_simple_broadcast,
      prepare.lhs_padded_strides.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<float>::MappedType*>(prepare.lhs_tensor->template Data<float>()),
      prepare.rhs_padded_strides.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<float>::MappedType*>(prepare.rhs_tensor->template Data<float>()),
      prepare.fdm_output_strides.GpuPtr(),
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<typename ToCudaType<float>::MappedType*>(prepare.output_tensor->template MutableData<float>()),
      prepare.output_tensor->Shape().Size());

  // weights = weights - gradient - inplace update
  BinaryElementwisePreparation prepareForUpdate(this);
  SetInOutIndexBeforePrepare(1, 2);
  Prepare(ctx, ctx->GetDeviceId(), &prepareForUpdate);
  ORT_RETURN_IF_ERROR(prepare.CopyToGpu());
  Impl_Sub<typename ToCudaType<float>::MappedType>(
      prepareForUpdate.output_rank_or_simple_broadcast,
      prepareForUpdate.lhs_padded_strides.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<float>::MappedType*>(prepareForUpdate.lhs_tensor->template Data<float>()),
      prepareForUpdate.rhs_padded_strides.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<float>::MappedType*>(prepareForUpdate.rhs_tensor->template Data<float>()),
      prepareForUpdate.fdm_output_strides.GpuPtr(),
      prepareForUpdate.fdm_H,
      prepareForUpdate.fdm_C,
      reinterpret_cast<typename ToCudaType<float>::MappedType*>(prepareForUpdate.output_tensor->template MutableData<float>()),
      prepareForUpdate.output_tensor->Shape().Size());

  // TODO: Output new weights in the future

  return Status::OK();
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
ONNX_OPERATOR_KERNEL_EX(
    AdamOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    AdamOptimizer);

Status AdamOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* eta_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(eta_tensor);

  Tensor* update_count = ctx->MutableInput<Tensor>(1);
  ORT_ENFORCE(update_count);

  Tensor* weights_tensor = ctx->MutableInput<Tensor>(2);
  ORT_ENFORCE(weights_tensor);

  const Tensor* gradients_tensor = ctx->Input<Tensor>(3);
  ORT_ENFORCE(gradients_tensor);

  Tensor* moment_1_tensor = ctx->MutableInput<Tensor>(4);
  ORT_ENFORCE(moment_1_tensor);

  Tensor* moment_2_tensor = ctx->MutableInput<Tensor>(5);
  ORT_ENFORCE(moment_2_tensor);

  AdamOptimizerImpl(
      eta_tensor->template Data<float>(),
      reinterpret_cast<int64_t*>(update_count->template MutableData<int64_t>()),
      weights_tensor->template Data<float>(),
      gradients_tensor->template Data<float>(),
      moment_1_tensor->template Data<float>(),
      moment_2_tensor->template Data<float>(),
      alpha_,
      beta_,
      lambda_,
      epsilon_,
      weights_tensor->template MutableData<float>(),
      moment_1_tensor->template MutableData<float>(),
      moment_2_tensor->template MutableData<float>(),
      weights_tensor->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
