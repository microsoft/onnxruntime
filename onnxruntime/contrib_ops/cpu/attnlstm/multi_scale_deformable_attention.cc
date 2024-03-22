// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/attnlstm/multi_scale_deformable_attention.h"

#include "core/framework/op_kernel.h"

#include <memory>

namespace onnxruntime {
namespace contrib {

MultiScaleDeformableAttention::MultiScaleDeformableAttention(const OpKernelInfo& info) : OpKernel(info) {
}

Status MultiScaleDeformableAttention::Compute(_Inout_ OpKernelContext* context) const {
  const auto* value = context->Input<Tensor>(0);
  const auto* value_spatial_shapes = context->Input<Tensor>(1);
  const auto* reference_points = context->Input<Tensor>(2);
  const auto* sampling_locations = context->Input<Tensor>(3);
  const auto* attention_weights = context->Input<Tensor>(4);

  const auto& value_input_shape = value->Shape();
  const auto& value_spatial_shapes_input_shape = value_spatial_shapes->Shape();
  const auto& attention_weights_input_shape = attention_weights->Shape();

  const int64_t S = value_input_shape[1];
  const int64_t M = value_input_shape[2];
  const int64_t D = value_input_shape[3];
  const int64_t L = value_spatial_shapes_input_shape[0];
  const int64_t P = attention_weights_input_shape[4];
  const int64_t Q = attention_weights_input_shape[2];

  auto* output = context->Output(0, { 1, Q, M*D });
  float * output_ptr = output->MutableData<float>();

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  if(D == 16 && M == 8 && P == 4) {
    // TODO: check AVX512 availability
    // AVX512 implementation
    ComputeInternal(
      context,
      value->Data<float>(),
      value_spatial_shapes->Data<int64_t>(),
      reference_points->Data<float>(),
      sampling_locations->Data<float>(),
      attention_weights->Data<float>(),
      output_ptr,
      S,
      M,
      L,
      P,
      D,
      Q,
      *thread_pool,
      alloc);
  } else {
    // Generic implementation
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::NOT_IMPLEMENTED, "Not implemented!");
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    MultiScaleDeformableAttention,
    1,
    KernelDefBuilder().TypeConstraint(
        "T",
        {DataTypeImpl::GetTensorType<float>()}),
    MultiScaleDeformableAttention)

}  // namespace contrib
}  // namespace onnxruntime
