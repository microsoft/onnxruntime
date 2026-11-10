// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/varlen_causal_conv_with_state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenCausalConvWithState,                                        \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                     \
          .MayInplace(4, 1)                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenCausalConvWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
VarlenCausalConvWithState<T>::VarlenCausalConvWithState(const OpKernelInfo& info) : OpKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");

  const int64_t state_update_capacity = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity >= 0 && state_update_capacity <= kMaxStateWindow,
              "state_update_capacity must be in [0, ", kMaxStateWindow, "]");
  state_update_capacity_ = static_cast<int>(state_update_capacity);

  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseDilation(info, dilation_));
}

template <typename T>
Status VarlenCausalConvWithState<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* weight_tensor = context->Input<Tensor>(1);
  const Tensor* cu_seqlens_tensor = context->Input<Tensor>(2);
  const Tensor* bias_tensor = context->Input<Tensor>(3);
  const Tensor* initial_state_tensor = context->Input<Tensor>(4);
  const Tensor* capture_count_tensor = context->Input<Tensor>(5);

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
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1 && cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must be rank 1 with at least 2 elements");
  const int64_t batch_size = cu_seqlens_shape[0] - 1;
  const int64_t total_tokens = input_shape[0];
  const int64_t channels = input_shape[1];
  ORT_RETURN_IF_NOT(total_tokens >= batch_size && channels > 0,
                    "total_tokens must be at least batch_size and channels must be positive");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1 && weight_shape[2] >= 1,
                    "weight must have shape (channels, 1, kernel_size)");

  const int64_t kernel_size = weight_shape[2];
  const int64_t pad = (kernel_size - 1) * dilation_;

  if (bias_tensor != nullptr) {
    ORT_RETURN_IF_NOT(bias_tensor->Shape().NumDimensions() == 1 && bias_tensor->Shape()[0] == channels,
                      "bias must have shape (channels)");
  }
  if (capture_count_tensor != nullptr) {
    ORT_RETURN_IF_NOT(capture_count_tensor->Shape().NumDimensions() == 1 &&
                          capture_count_tensor->Shape()[0] == batch_size,
                      "capture_count must have shape (batch_size)");
  }

  const TensorShape state_shape({batch_size, channels, pad});
  ORT_RETURN_IF_NOT(initial_state_tensor->Shape() == state_shape,
                    "initial_state must have shape ", state_shape.ToString());

  Tensor* output_tensor = context->Output(0, input_shape);
  Tensor* final_state_tensor = context->Output(1, state_shape);
  Tensor* state_update_tensor =
      context->Output(2, TensorShape({batch_size, state_update_capacity_, channels}));
  if (state_update_tensor != nullptr && state_update_tensor->SizeInBytes() > 0) {
    std::memset(state_update_tensor->MutableDataRaw(), 0, state_update_tensor->SizeInBytes());
  }

  const T* input_data = input_tensor->Data<T>();
  const T* weight_data = weight_tensor->Data<T>();
  const T* bias_data = bias_tensor != nullptr ? bias_tensor->Data<T>() : nullptr;
  const T* init_state_data = initial_state_tensor->Data<T>();
  const int32_t* cu_seqlens = cu_seqlens_tensor->Data<int32_t>();
  ORT_RETURN_IF_NOT(cu_seqlens[0] == 0 && cu_seqlens[batch_size] == total_tokens,
                    "cumulative_sequence_length must start at 0 and end at total_tokens");
  for (int64_t b = 0; b < batch_size; ++b) {
    ORT_RETURN_IF_NOT(cu_seqlens[b] <= cu_seqlens[b + 1],
                      "cumulative_sequence_length must be non-decreasing");
  }
  const int32_t* capture_counts =
      capture_count_tensor != nullptr ? capture_count_tensor->Data<int32_t>() : nullptr;
  T* output_data = output_tensor->MutableData<T>();
  T* final_state_data = final_state_tensor->MutableData<T>();
  T* state_update_data =
      state_update_tensor != nullptr ? state_update_tensor->MutableData<T>() : nullptr;

  const bool apply_silu = (activation_ == "silu" || activation_ == "swish");
  const int64_t total_tasks = batch_size * channels;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(total_tasks),
      [&](ptrdiff_t task_idx) {
        const int64_t b = task_idx / channels;
        const int64_t c = task_idx % channels;

        int64_t start = std::clamp<int64_t>(cu_seqlens[b], 0, total_tokens);
        int64_t end = std::clamp<int64_t>(cu_seqlens[b + 1], start, total_tokens);
        const int64_t seq_len = end - start;

        const int64_t state_base = (b * channels + c) * pad;
        std::vector<float> window(static_cast<size_t>(pad + seq_len));
        std::vector<T> window_raw(static_cast<size_t>(pad + seq_len));
        for (int64_t i = 0; i < pad; ++i) {
          const T v = init_state_data[state_base + i];
          window_raw[static_cast<size_t>(i)] = v;
          window[static_cast<size_t>(i)] = static_cast<float>(v);
        }
        for (int64_t t = 0; t < seq_len; ++t) {
          const T v = input_data[(start + t) * channels + c];
          window_raw[static_cast<size_t>(pad + t)] = v;
          window[static_cast<size_t>(pad + t)] = static_cast<float>(v);
        }

        const T* w = weight_data + c * kernel_size;
        const float bias_val = bias_data != nullptr ? static_cast<float>(bias_data[c]) : 0.0f;

        for (int64_t t = 0; t < seq_len; ++t) {
          float sum = bias_val;
          for (int64_t k = 0; k < kernel_size; ++k) {
            sum += static_cast<float>(w[k]) * window[static_cast<size_t>(t + k * dilation_)];
          }
          if (apply_silu) {
            sum = sum / (1.0f + std::exp(-sum));
          }
          output_data[(start + t) * channels + c] = static_cast<T>(sum);
        }

        if (capture_counts != nullptr && state_update_data != nullptr && state_update_capacity_ > 0) {
          const int64_t captured = std::max<int64_t>(
              0, std::min<int64_t>({capture_counts[b], state_update_capacity_, seq_len}));
          for (int64_t t = 0; t < captured; ++t) {
            state_update_data[(b * state_update_capacity_ + t) * channels + c] =
                window_raw[static_cast<size_t>(pad + t)];
          }
        }

        for (int64_t i = 0; i < pad; ++i) {
          final_state_data[state_base + i] = window_raw[static_cast<size_t>(seq_len + i)];
        }
      },
      0);

  return Status::OK();
}

template class VarlenCausalConvWithState<float>;
template class VarlenCausalConvWithState<MLFloat16>;
template class VarlenCausalConvWithState<BFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
