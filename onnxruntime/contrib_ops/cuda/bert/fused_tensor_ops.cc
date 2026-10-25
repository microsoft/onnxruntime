// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/fused_tensor_ops.h"

#include <algorithm>
#include <array>
#include <limits>

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    SigmoidGateMul, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 3),
    SigmoidGateMul);

SigmoidGateMul::SigmoidGateMul(const OpKernelInfo& info)
    : CudaKernel(info),
      head_dim_(narrow<int>(info.GetAttrOrDefault<int64_t>("head_dim", 0))),
      heads_(narrow<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0))),
      q_stride_(narrow<int>(info.GetAttrOrDefault<int64_t>("q_stride", 0))),
      tok_stride_(info.GetAttrOrDefault<int64_t>("tok_stride", 0)),
      q_offset_(info.GetAttrOrDefault<int64_t>("q_offset", 0)),
      broadcast_gate_(info.GetAttrOrDefault<int64_t>("broadcast_gate", 0) == 1),
      sigmoid_in_fp32_(info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1) == 1),
      silu_gate_(info.GetAttrOrDefault<int64_t>("silu_gate", 0) == 1) {
  const int64_t broadcast = info.GetAttrOrDefault<int64_t>("broadcast_gate", 0);
  ORT_ENFORCE(broadcast == 0 || broadcast == 1, "SigmoidGateMul broadcast_gate must be zero or one");
  const int64_t fp32 = info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1);
  ORT_ENFORCE(fp32 == 0 || fp32 == 1, "SigmoidGateMul sigmoid_in_fp32 must be zero or one");
  const int64_t silu = info.GetAttrOrDefault<int64_t>("silu_gate", 0);
  ORT_ENFORCE(silu == 0 || silu == 1, "SigmoidGateMul silu_gate must be zero or one");
  ORT_ENFORCE(!(broadcast_gate_ && silu_gate_), "SigmoidGateMul broadcast_gate and silu_gate are exclusive");
  ORT_ENFORCE(broadcast_gate_ || silu_gate_ || (head_dim_ > 0 && heads_ > 0 && q_stride_ >= 2LL * head_dim_ && tok_stride_ > 0 && q_offset_ >= 0),
              "SigmoidGateMul received invalid layout attributes");
  ORT_ENFORCE(!(broadcast_gate_ || silu_gate_) || (head_dim_ == 0 && heads_ == 0 && q_stride_ == 0 &&
                                                   tok_stride_ == 0 && q_offset_ == 0),
              "SigmoidGateMul broadcast_gate/silu_gate cannot be combined with packed layout attributes");
}

Status SigmoidGateMul::ComputeInternal(OpKernelContext* context) const {
  const Tensor* numerics = context->Input<Tensor>(3);
  ORT_RETURN_IF_NOT(numerics == nullptr ||
                        (numerics->IsDataType<int32_t>() && numerics->Shape().Size() == 1 &&
                         (numerics->Data<int32_t>()[0] == 0 || numerics->Data<int32_t>()[0] == 1)),
                    "prefill_numerics must be an int32 scalar containing 0 or 1");
  const bool dense_prefill = numerics != nullptr && numerics->Data<int32_t>()[0] == 1;
  const Tensor* attention = context->Input<Tensor>(0);
  const Tensor* projection = context->Input<Tensor>(1);
  const Tensor* residual = context->Input<Tensor>(2);
  const Tensor* gate_weight = context->Input<Tensor>(4);
  const int64_t count = attention->Shape().Size();
  ORT_RETURN_IF_NOT(residual == nullptr || residual->Shape() == attention->Shape(),
                    "SigmoidGateMul residual must have the same shape as attention");
  if (gate_weight != nullptr) {
    ORT_RETURN_IF_NOT(broadcast_gate_ && !silu_gate_,
                      "SigmoidGateMul gate_weight requires broadcast_gate without silu_gate");
    const auto& shape = attention->Shape();
    const auto& activation_shape = projection->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1 && shape[shape.NumDimensions() - 1] > 0 &&
                          activation_shape.NumDimensions() == shape.NumDimensions() &&
                          activation_shape[activation_shape.NumDimensions() - 1] > 0,
                      "SigmoidGateMul projected gate requires matching ranks and positive final dimensions");
    for (size_t axis = 0; axis + 1 < shape.NumDimensions(); ++axis) {
      ORT_RETURN_IF_NOT(shape[axis] == activation_shape[axis],
                        "SigmoidGateMul projected gate leading dimensions must match attention");
    }
    const int64_t input_width = activation_shape[activation_shape.NumDimensions() - 1];
    ORT_RETURN_IF_NOT(gate_weight->Shape().NumDimensions() == 2 &&
                          gate_weight->Shape()[0] == input_width && gate_weight->Shape()[1] == 1,
                      "SigmoidGateMul gate_weight must have shape [input_width, 1]");
    Tensor* output = context->Output(0, shape);
    return LaunchProjectedSigmoidGateMul(
        Stream(context), reinterpret_cast<const uint16_t*>(attention->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(projection->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(gate_weight->Data<BFloat16>()),
        residual == nullptr ? nullptr : reinterpret_cast<const uint16_t*>(residual->Data<BFloat16>()),
        reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
        count / shape[shape.NumDimensions() - 1], input_width, shape[shape.NumDimensions() - 1],
        dense_prefill ? false : sigmoid_in_fp32_);
  }
  int64_t gate_row_width = 0;
  if (silu_gate_) {
    ORT_RETURN_IF_NOT(projection->Shape() == attention->Shape(),
                      "SigmoidGateMul silu gate must have the same shape as attention");
  } else if (broadcast_gate_) {
    TensorShapeVector expected(attention->Shape().GetDims().begin(), attention->Shape().GetDims().end());
    ORT_RETURN_IF_NOT(!expected.empty() && expected.back() > 0,
                      "SigmoidGateMul broadcast_gate requires a positive final attention dimension");
    gate_row_width = expected.back();
    expected.back() = 1;
    ORT_RETURN_IF_NOT(projection->Shape() == TensorShape(expected),
                      "SigmoidGateMul broadcast gate must match attention with a singleton final dimension");
  } else {
    const int64_t row_width = static_cast<int64_t>(heads_) * head_dim_;
    ORT_RETURN_IF_NOT(count % row_width == 0,
                      "SigmoidGateMul attention size must be divisible by heads * head_dim");
    const int64_t tokens = count / row_width;
    if (tokens > 0) {
      int64_t required_extent;
      int64_t head_extent;
      ORT_RETURN_IF_NOT(SafeMultiply(tokens - 1, tok_stride_, required_extent) &&
                            SafeMultiply(static_cast<int64_t>(heads_ - 1), static_cast<int64_t>(q_stride_), head_extent) &&
                            SafeAdd(head_extent, 2LL * head_dim_, head_extent) &&
                            SafeAdd(required_extent, q_offset_, required_extent) &&
                            SafeAdd(required_extent, head_extent, required_extent),
                        "SigmoidGateMul projection extent overflows int64");
      ORT_RETURN_IF_NOT(projection->Shape().Size() >= required_extent,
                        "SigmoidGateMul projection is too small for the configured layout");
    }
  }
  Tensor* output = context->Output(0, attention->Shape());
  return LaunchSigmoidGateMul(
      Stream(context), reinterpret_cast<const uint16_t*>(attention->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(projection->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()), count, head_dim_, heads_,
      q_stride_, tok_stride_, q_offset_, gate_row_width,
      residual == nullptr ? nullptr : reinterpret_cast<const uint16_t*>(residual->Data<BFloat16>()),
      dense_prefill ? false : sigmoid_in_fp32_, silu_gate_);
}

}  // namespace onnxruntime::contrib::cuda
