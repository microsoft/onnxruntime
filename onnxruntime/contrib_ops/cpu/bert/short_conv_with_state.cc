// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/short_conv_with_state.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "contrib_ops/cpu/bert/kernel_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_SHORT_CONV_WITH_STATE_TYPED(T)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShortConvWithState,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShortConvWithState<T>);

REGISTER_SHORT_CONV_WITH_STATE_TYPED(float)
REGISTER_SHORT_CONV_WITH_STATE_TYPED(MLFloat16)

#undef REGISTER_SHORT_CONV_WITH_STATE_TYPED

template <typename T>
ShortConvWithState<T>::ShortConvWithState(const OpKernelInfo& info) : OpKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConvWithState<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* past_state = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* weight = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);

  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  const TensorShape& scale_shape = norm_scale->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape.NumDimensions() == 2,
                    "norm_scale must have shape (hc_mult, hidden_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  const int64_t kernel_size = weight_shape[2];
  const int64_t state_len = dilation_ * (kernel_size - 1);

  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1 && weight_shape[2] == kernel_size,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");

  if (past_state != nullptr) {
    const TensorShape& past_shape = past_state->Shape();
    ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3 &&
                          past_shape[0] == batch_size &&
                          past_shape[1] == channels &&
                          past_shape[2] == state_len,
                      "past_state must have shape (batch_size, channels, dilation*(kernel_size-1))");
  }
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  TensorShape present_shape({batch_size, channels, state_len});
  Tensor* present_state = context->Output(1, present_shape);

  if (input_shape.Size() == 0) {
    if (state_len > 0 && past_state != nullptr) {
      std::memcpy(present_state->MutableData<T>(), past_state->Data<T>(),
                  narrow<size_t>(present_shape.Size()) * sizeof(T));
    } else if (state_len > 0) {
      std::memset(present_state->MutableData<T>(), 0, narrow<size_t>(present_shape.Size()) * sizeof(T));
    }
    return Status::OK();
  }

  const T* input_data = input->Data<T>();
  const T* scale_data = norm_scale->Data<T>();
  const T* weight_data = weight->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();
  const T* past_data = past_state == nullptr ? nullptr : past_state->Data<T>();
  T* output_data = output->MutableData<T>();
  T* present_data = present_state->MutableData<T>();
  const bool apply_silu = activation_ == "silu" || activation_ == "swish";

  auto* tp = context->GetOperatorThreadPool();

  if (sequence_length == 1) {
    // ==== Decode fast path (L==1): no scratch allocation ====
    // For each (batch, hc_mult) pair: normalize the single token, then for each channel
    // within that group compute the dilated convolution directly from past_state + normed value,
    // shift state left by 1 position, and append the new normed value.
    const int64_t total_tasks = batch_size * channels;
    ThreadPool::TryParallelFor(
        tp, narrow<ptrdiff_t>(total_tasks), static_cast<double>(kernel_size),
        [&](ptrdiff_t begin, ptrdiff_t end) {
          for (int64_t task = begin; task < end; ++task) {
            const int64_t b = task / channels;
            const int64_t flat_c = task % channels;
            const int64_t g = flat_c / hidden_size;
            const int64_t c = flat_c % hidden_size;

            // Normalize the single input token for this (b, g) row.
            const T* input_row = input_data + (b * hc_mult + g) * hidden_size;
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden_size; ++i) {
              const float v = static_cast<float>(input_row[i]);
              sum_sq += v * v;
            }
            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + epsilon_);
            const float scale_val = static_cast<float>(scale_data[g * hidden_size + c]);
            const T normed_val = static_cast<T>(static_cast<float>(input_row[c]) * inv_rms * scale_val);

            // Dilated convolution: dot product over kernel_size taps.
            // State layout: [B, C, state_len] where state_len = dilation * (kernel_size - 1).
            // The "timeline" is conceptually [past_state..., normed_val].
            // For tap k (0..K-1), the source position in this timeline is:
            //   state_len - (kernel_size - 1 - k) * dilation
            // If k == kernel_size - 1, source = state_len (the new normed_val).
            const T* past_row = past_data != nullptr
                                    ? past_data + (b * channels + flat_c) * state_len
                                    : nullptr;
            float conv_sum = bias_data != nullptr ? static_cast<float>(bias_data[flat_c]) : 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
              const int64_t src = state_len - (kernel_size - 1 - k) * dilation_;
              float src_val;
              if (src == state_len) {
                src_val = static_cast<float>(normed_val);
              } else if (src >= 0 && src < state_len && past_row != nullptr) {
                src_val = static_cast<float>(past_row[src]);
              } else {
                src_val = 0.0f;
              }
              conv_sum += src_val * static_cast<float>(weight_data[flat_c * kernel_size + k]);
            }
            if (apply_silu) {
              conv_sum = kernel_helper::SiluFloat(conv_sum);
            }
            output_data[(b * hc_mult + g) * hidden_size + c] = static_cast<T>(conv_sum);

            // Update present_state: shift past left by 1, append normed_val at end.
            T* present_row = present_data + (b * channels + flat_c) * state_len;
            if (state_len > 0) {
              if (past_row != nullptr && state_len > 1) {
                std::memcpy(present_row, past_row + 1, static_cast<size_t>(state_len - 1) * sizeof(T));
              } else if (state_len > 1) {
                std::memset(present_row, 0, static_cast<size_t>(state_len - 1) * sizeof(T));
              }
              present_row[state_len - 1] = normed_val;
            }
          }
        });
    return Status::OK();
  }

  // ==== Prefill / multi-token path ====
  // Step 1: Compute branchwise RMSNorm of input and store into a temporary buffer.
  // Layout: normed[b][t][g][c] where (b,t,g) is the row and c is within hidden_size.
  // Store normed values in type T so that precision is consistent across chunk boundaries
  // (past_state carries T-rounded values).
  const int64_t rows = batch_size * sequence_length * hc_mult;
  const int64_t normed_count = batch_size * sequence_length * channels;

  // Use ORT temp allocator to avoid per-call heap allocation.
  auto alloc = context->GetOperatorThreadPool() != nullptr
                   ? context->GetAllocator(0, OrtMemTypeDefault)
                   : context->GetAllocator(0, OrtMemTypeDefault);
  auto normed_buffer = IAllocator::MakeUniquePtr<T>(alloc, static_cast<size_t>(normed_count));
  T* normed = normed_buffer.get();

  ThreadPool::TryParallelFor(
      tp, narrow<ptrdiff_t>(rows), static_cast<double>(hidden_size),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const T* input_row = input_data + row * hidden_size;
          float sum_sq = 0.0f;
          for (int64_t i = 0; i < hidden_size; ++i) {
            const float value = static_cast<float>(input_row[i]);
            sum_sq += value * value;
          }
          const float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + epsilon_);

          const int64_t g = row % hc_mult;
          const int64_t base = (row / hc_mult) * channels + g * hidden_size;
          T* normed_row = normed + static_cast<size_t>(base);
          for (int64_t i = 0; i < hidden_size; ++i) {
            const float scale = static_cast<float>(scale_data[g * hidden_size + i]);
            normed_row[i] = static_cast<T>(static_cast<float>(input_row[i]) * rms * scale);
          }
        }
      });

  // Step 2: For each (batch, channel) pair, perform the dilated causal convolution
  // using past_state + normed current timesteps.
  const int64_t total_tasks = batch_size * channels;
  const double cost_per_task = static_cast<double>(sequence_length * kernel_size);

  ThreadPool::TryParallelFor(
      tp, narrow<ptrdiff_t>(total_tasks), cost_per_task,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        const int64_t timeline_len = state_len + sequence_length;
        std::vector<float> timeline(static_cast<size_t>(timeline_len));
        std::vector<float> w_buf;
        if constexpr (!std::is_same_v<T, float>) {
          w_buf.resize(static_cast<size_t>(kernel_size));
        }

        for (int64_t task = begin; task < end; ++task) {
          const int64_t b = task / channels;
          const int64_t flat_c = task % channels;

          if (past_data != nullptr) {
            const T* past_row = past_data + (b * channels + flat_c) * state_len;
            for (int64_t i = 0; i < state_len; ++i) {
              timeline[static_cast<size_t>(i)] = static_cast<float>(past_row[i]);
            }
          } else {
            std::memset(timeline.data(), 0, static_cast<size_t>(state_len) * sizeof(float));
          }

          for (int64_t t = 0; t < sequence_length; ++t) {
            timeline[static_cast<size_t>(state_len + t)] =
                static_cast<float>(normed[static_cast<size_t>((b * sequence_length + t) * channels + flat_c)]);
          }

          const float* w_ptr;
          if constexpr (std::is_same_v<T, float>) {
            w_ptr = reinterpret_cast<const float*>(weight_data) + flat_c * kernel_size;
          } else {
            for (int64_t k = 0; k < kernel_size; ++k) {
              w_buf[static_cast<size_t>(k)] = static_cast<float>(weight_data[flat_c * kernel_size + k]);
            }
            w_ptr = w_buf.data();
          }
          const float bias_val = bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[flat_c]);

          for (int64_t t = 0; t < sequence_length; ++t) {
            float sum = bias_val;
            for (int64_t k = 0; k < kernel_size; ++k) {
              const int64_t src = (state_len + t) - (kernel_size - 1 - k) * dilation_;
              if (src >= 0 && src < timeline_len) {
                sum += timeline[static_cast<size_t>(src)] * w_ptr[k];
              }
            }
            if (apply_silu) {
              sum = kernel_helper::SiluFloat(sum);
            }
            const int64_t g = flat_c / hidden_size;
            const int64_t c = flat_c % hidden_size;
            output_data[((b * sequence_length + t) * hc_mult + g) * hidden_size + c] = static_cast<T>(sum);
          }

          T* present_row = present_data + (b * channels + flat_c) * state_len;
          for (int64_t i = 0; i < state_len; ++i) {
            present_row[i] = static_cast<T>(timeline[static_cast<size_t>(timeline_len - state_len + i)]);
          }
        }
      });

  return Status::OK();
}

template class ShortConvWithState<float>;
template class ShortConvWithState<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
