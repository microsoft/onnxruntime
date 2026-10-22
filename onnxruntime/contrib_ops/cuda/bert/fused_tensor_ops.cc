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

#define REGISTER_ARGMAX_LAST_AXIS(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      ArgMaxLastAxis, kMSDomain, 1, T, kCudaExecutionProvider,          \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      ArgMaxLastAxis<T>);

REGISTER_ARGMAX_LAST_AXIS(float)
REGISTER_ARGMAX_LAST_AXIS(MLFloat16)
REGISTER_ARGMAX_LAST_AXIS(BFloat16)

#undef REGISTER_ARGMAX_LAST_AXIS

ONNX_OPERATOR_KERNEL_EX(
    EmbeddingRMSNorm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TT", BuildKernelDefConstraints<BFloat16, int8_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()),
    EmbeddingRMSNorm);

ONNX_OPERATOR_KERNEL_EX(
    RowSelect, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    RowSelect);

ONNX_OPERATOR_KERNEL_EX(
    SigmoidGateMul, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 3),
    SigmoidGateMul);

ONNX_OPERATOR_KERNEL_EX(
    UntileHyperConnectionWeights, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    UntileHyperConnectionWeights);

template <typename T>
ArgMaxLastAxis<T>::ArgMaxLastAxis(const OpKernelInfo& info)
    : CudaKernel(info), keepdims_(info.GetAttrOrDefault<int64_t>("keepdims", 0) != 0) {}

template <typename T>
Status ArgMaxLastAxis<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1 &&
                        shape[shape.NumDimensions() - 1] > 0,
                    "ArgMaxLastAxis requires a non-empty final dimension");
  const int64_t width = shape[shape.NumDimensions() - 1];
  const int64_t rows = shape.Size() / width;
  TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end() - 1);
  if (keepdims_) output_dims.push_back(1);
  Tensor* output = context->Output(0, TensorShape(output_dims));
  const size_t partial_count = SafeInt<size_t>(rows) * GetArgMaxLastAxisPartitions(width);
  auto partial_values = GetScratchBuffer<float>(partial_count, GetComputeStream(context));
  auto partial_indices = GetScratchBuffer<int64_t>(partial_count, GetComputeStream(context));
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  return LaunchArgMaxLastAxis(Stream(context), reinterpret_cast<const CudaT*>(input->Data<T>()),
                              input->GetElementType(), rows, width,
                              output->MutableData<int64_t>(), partial_values.get(), partial_indices.get());
}

EmbeddingRMSNorm::EmbeddingRMSNorm(const OpKernelInfo& info)
    : CudaKernel(info), epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)) {
  ORT_ENFORCE(epsilon_ >= 0.0f, "EmbeddingRMSNorm epsilon must be non-negative");
}

Status EmbeddingRMSNorm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* ids = context->Input<Tensor>(0);
  const Tensor* table = context->Input<Tensor>(1);
  const Tensor* weight = context->Input<Tensor>(2);
  const Tensor* committed = context->InputCount() > 3 ? context->Input<Tensor>(3) : nullptr;
  ORT_RETURN_IF_NOT(table->Shape().NumDimensions() == 2,
                    "EmbeddingRMSNorm table must have shape [vocabulary, hidden]");
  const int64_t vocab_size = table->Shape()[0];
  const int64_t hidden_size = table->Shape()[1];
  ORT_RETURN_IF_NOT(weight->Shape().NumDimensions() == 1 &&
                        weight->Shape()[0] == hidden_size,
                    "EmbeddingRMSNorm weight must match the embedding hidden size");
  const int64_t rows = ids->Shape().Size();
  TensorShapeVector output_dims(ids->Shape().GetDims().begin(), ids->Shape().GetDims().end());
  output_dims.push_back(hidden_size);
  Tensor* output = context->Output(0, TensorShape(output_dims));
  int64_t* slot_ids_ptr = nullptr;
  int32_t* seqlens_k_ptr = nullptr;
  int32_t* cu_seqlens_ptr = nullptr;
  if (context->OutputCount() > 1) {
    Tensor* slot_ids = context->Output(1, TensorShape({1, rows}));
    if (slot_ids != nullptr) slot_ids_ptr = slot_ids->MutableData<int64_t>();
  }
  if (context->OutputCount() > 2) {
    Tensor* seqlens_k = context->Output(2, TensorShape({1}));
    if (seqlens_k != nullptr) seqlens_k_ptr = seqlens_k->MutableData<int32_t>();
  }
  if (context->OutputCount() > 3) {
    Tensor* cu_seqlens = context->Output(3, TensorShape({2}));
    if (cu_seqlens != nullptr) cu_seqlens_ptr = cu_seqlens->MutableData<int32_t>();
  }
  const int32_t* committed_ptr = committed != nullptr ? committed->Data<int32_t>() : nullptr;
  if (table->IsDataType<int8_t>()) {
    const int8_t* table_i8 = table->Data<int8_t>();
    const float* row_scales = reinterpret_cast<const float*>(table_i8 + vocab_size * hidden_size);
    return LaunchEmbeddingRMSNormInt8(Stream(context), ids->Data<int64_t>(), table_i8, row_scales,
                                      weight->Data<float>(), epsilon_,
                                      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
                                      rows, vocab_size, hidden_size,
                                      committed_ptr, slot_ids_ptr, seqlens_k_ptr, cu_seqlens_ptr);
  }
  return LaunchEmbeddingRMSNorm(Stream(context), ids->Data<int64_t>(),
                                reinterpret_cast<const uint16_t*>(table->Data<BFloat16>()),
                                weight->Data<float>(), epsilon_,
                                reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
                                rows, vocab_size, hidden_size,
                                committed_ptr, slot_ids_ptr, seqlens_k_ptr, cu_seqlens_ptr);
}

Status RowSelect::ComputeInternal(OpKernelContext* context) const {
  const Tensor* selection = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(selection->Shape().Size() == 1, "RowSelect selection must have one element");
  std::array<const uint16_t*, 3> inputs{};
  std::array<uint16_t*, 3> outputs{};
  std::array<int64_t, 3> row_sizes{};
  int64_t num_rows = 2;
  for (int i = 0; i < 3; ++i) {
    const Tensor* input = context->Input<Tensor>(i + 1);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 2 && shape[0] == 1 && shape[1] >= 1,
                      "RowSelect inputs must have shape [1, R, ...]");
    num_rows = shape[1];
    row_sizes[i] = shape.Size() / shape[1];
    TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end());
    output_dims[1] = 1;
    Tensor* output = context->Output(i, TensorShape(output_dims));
    inputs[i] = reinterpret_cast<const uint16_t*>(input->Data<BFloat16>());
    outputs[i] = reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>());
  }
  return LaunchRowSelect(Stream(context), selection->Data<int64_t>(), inputs.data(),
                         outputs.data(), row_sizes.data(), num_rows);
}

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

UntileHyperConnectionWeights::UntileHyperConnectionWeights(const OpKernelInfo& info)
    : CudaKernel(info), kind_(static_cast<int>(info.GetAttrOrDefault<int64_t>("kind", -1))) {
  ORT_ENFORCE(kind_ == 0 || kind_ == 1, "UntileHyperConnectionWeights kind must be 0 or 1");
}

Status UntileHyperConnectionWeights::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input->Shape() == TensorShape({160, 20480}),
                    "UntileHyperConnectionWeights input must have shape [160, 20480]");
  const TensorShape output_shape = kind_ == 0 ? TensorShape({10240, 320}) : TensorShape({320, 10240});
  Tensor* output = context->Output(0, output_shape);
  return LaunchUntileHyperConnectionWeights(
      Stream(context), reinterpret_cast<const uint16_t*>(input->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()), kind_);
}

template class ArgMaxLastAxis<float>;
template class ArgMaxLastAxis<MLFloat16>;
template class ArgMaxLastAxis<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
