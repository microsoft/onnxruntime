// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_causal_conv_with_state.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenCausalConvWithState,                                        \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .MayInplace(4, 1)                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenCausalConvWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
VarlenCausalConvWithState<T>::VarlenCausalConvWithState(const OpKernelInfo& info) : CudaKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");

  const int64_t state_update_capacity = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity >= 0 && state_update_capacity <= kMaxStateWindow,
              "state_update_capacity must be in [0, ", kMaxStateWindow, "]");
  state_update_capacity_ = static_cast<int>(state_update_capacity);
}

template <typename T>
Status VarlenCausalConvWithState<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* weight_tensor = context->Input<Tensor>(1);
  const Tensor* cu_seqlens_tensor = context->Input<Tensor>(2);
  const Tensor* bias_tensor = context->Input<Tensor>(3);  // optional
  const Tensor* initial_state_tensor = context->Input<Tensor>(4);
  const Tensor* capture_count_tensor = context->Input<Tensor>(5);  // optional

  ORT_RETURN_IF_NOT(input_tensor != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "weight is required");
  ORT_RETURN_IF_NOT(cu_seqlens_tensor != nullptr, "cumulative_sequence_length input is required");
  ORT_RETURN_IF_NOT(initial_state_tensor != nullptr, "initial_state input is required");
  ORT_RETURN_IF_NOT((state_update_capacity_ > 0) == (capture_count_tensor != nullptr),
                    "capture_count must be present exactly when state_update_capacity is positive");

  const auto& input_shape = input_tensor->Shape();
  const auto& weight_shape = weight_tensor->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2,
                    "input must be rank 2 (total_tokens, channels), got rank ", input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must be rank 3 (channels, 1, kernel_size), got rank ", weight_shape.NumDimensions());

  const auto& cu_seqlens_shape = cu_seqlens_tensor->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1,
                    "cumulative_sequence_length must be rank 1 (batch_size + 1), got rank ",
                    cu_seqlens_shape.NumDimensions());
  ORT_RETURN_IF_NOT(cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have at least 2 elements (batch_size >= 1), got ",
                    cu_seqlens_shape[0]);
  // batch_size = cu_seqlens.Shape()[0] - 1, never derived from total_tokens: a ragged batch has
  // no fixed relationship between the two beyond total_tokens >= batch_size.
  const int64_t batch_size_64 = cu_seqlens_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max(),
                    "batch size is too large for the CUDA kernel");
  const int batch_size = static_cast<int>(batch_size_64);

  if (capture_count_tensor != nullptr) {
    const auto& capture_count_shape = capture_count_tensor->Shape();
    ORT_RETURN_IF_NOT(capture_count_shape.NumDimensions() == 1 && capture_count_shape[0] == batch_size_64,
                      "capture_count must have shape (", batch_size_64, "), got ",
                      capture_count_shape.ToString());
  }

  const int64_t total_tokens_64 = input_shape[0];
  const int64_t channels_64 = input_shape[1];
  ORT_RETURN_IF_NOT(total_tokens_64 <= std::numeric_limits<int>::max() &&
                        channels_64 <= std::numeric_limits<int>::max(),
                    "input dimensions are too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(total_tokens_64 >= batch_size_64,
                    "total_tokens must be at least batch_size because every sequence must contain a token");
  ORT_RETURN_IF_NOT(channels_64 > 0, "input channel dimension must be positive");
  const int channels = static_cast<int>(channels_64);

  ORT_RETURN_IF_NOT(weight_shape[0] == channels_64,
                    "weight[0] (", weight_shape[0], ") must match input channels (", channels_64, ")");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1,
                    "weight[1] must be 1 for depthwise convolution, got ", weight_shape[1]);
  const int64_t kernel_size_64 = weight_shape[2];
  ORT_RETURN_IF_NOT(kernel_size_64 >= 1 && kernel_size_64 <= std::numeric_limits<int>::max(),
                    "weight last dim (kernel_size) must be positive, got ", kernel_size_64);
  const int kernel_size = static_cast<int>(kernel_size_64);
  const int pad = kernel_size - 1;

  if (bias_tensor != nullptr) {
    const auto& bias_shape = bias_tensor->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 1 && bias_shape[0] == channels_64,
                      "bias must have shape (", channels_64, "), got ", bias_shape.ToString());
  }

  const TensorShape state_shape({batch_size_64, channels_64, pad});
  ORT_RETURN_IF_NOT(initial_state_tensor->Shape() == state_shape,
                    "initial_state must have shape ", state_shape.ToString(), ", got ",
                    initial_state_tensor->Shape().ToString());

  Tensor* output_tensor = context->Output(0, input_shape);
  Tensor* final_state_tensor = context->Output(1, state_shape);
  const TensorShape state_update_shape({batch_size_64, state_update_capacity_, channels_64});
  Tensor* state_update_tensor = context->Output(2, state_update_shape);
  if (state_update_capacity_ > 0 && state_update_tensor != nullptr) {
    const size_t count = SafeInt<size_t>(batch_size) * state_update_capacity_ * channels;
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        state_update_tensor->MutableDataRaw(), 0, count * sizeof(T), Stream(context)));
  }

  bool apply_silu = (activation_ == "silu" || activation_ == "swish");

  // total_tokens == batch_size selects a candidate decode path. That path still reads and
  // validates each exact [b, b + 1] offset interval before accessing any data.
  const bool all_ones = (total_tokens_64 == batch_size_64);

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchVarlenCausalConvWithStateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(weight_tensor->Data<T>()),
      bias_tensor ? reinterpret_cast<const CudaT*>(bias_tensor->Data<T>()) : nullptr,
      reinterpret_cast<const CudaT*>(initial_state_tensor->Data<T>()),
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      reinterpret_cast<CudaT*>(final_state_tensor->MutableData<T>()),
      state_update_capacity_ > 0 && state_update_tensor
          ? reinterpret_cast<CudaT*>(state_update_tensor->MutableData<T>())
          : nullptr,
      cu_seqlens_tensor->Data<int32_t>(),
      capture_count_tensor ? capture_count_tensor->Data<int32_t>() : nullptr,
      batch_size,
      static_cast<int>(total_tokens_64),
      all_ones,
      channels,
      kernel_size,
      apply_silu,
      GetDeviceProp().maxThreadsPerBlock,
      state_update_capacity_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
