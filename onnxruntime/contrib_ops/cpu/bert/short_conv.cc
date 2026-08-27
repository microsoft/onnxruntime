// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/short_conv.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "contrib_ops/cpu/bert/kernel_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_SHORT_CONV_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShortConv,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShortConv<T>);

REGISTER_SHORT_CONV_TYPED(float)
REGISTER_SHORT_CONV_TYPED(MLFloat16)

#undef REGISTER_SHORT_CONV_TYPED

template <typename T>
ShortConv<T>::ShortConv(const OpKernelInfo& info) : OpKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConv<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* past_state = context->Input<Tensor>(4);

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

  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  // The convolution receptive field reaches this many positions before the current token, so this is
  // exactly the amount of normed history that has to be carried between invocations.
  const int64_t state_length = (kernel_size - 1) * dilation_;
  const TensorShape state_shape({batch_size, state_length, hc_mult, hidden_size});
  if (past_state != nullptr) {
    ORT_RETURN_IF_NOT(past_state->Shape() == state_shape,
                      "past_state must have shape (batch_size, (kernel_size - 1) * dilation, hc_mult, hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  Tensor* present_state = context->Output(1, state_shape);
  if (input_shape.Size() == 0) {
    // No new positions, so present_state is past_state unchanged (zeros when there is no history).
    // It still has to be written: output buffers are not zero-initialized.
    if (present_state != nullptr && state_shape.Size() > 0) {
      T* present_data = present_state->MutableData<T>();
      const size_t count = static_cast<size_t>(state_shape.Size());
      if (past_state != nullptr) {
        const T* past_begin = past_state->Data<T>();
        std::copy(past_begin, past_begin + count, present_data);
      } else {
        std::fill(present_data, present_data + count, T{});
      }
    }
    return Status::OK();
  }

  const T* input_data = input->Data<T>();
  const T* weight_data = weight->Data<T>();
  const T* scale_data = norm_scale->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();
  const T* past_data = past_state == nullptr ? nullptr : past_state->Data<T>();
  T* output_data = output->MutableData<T>();
  const bool apply_silu = activation_ == "silu" || activation_ == "swish";
  const int64_t rows = batch_size * sequence_length * hc_mult;
  const int64_t total = rows * hidden_size;

  // The RMS reduction only depends on the (batch, sequence, hc_mult) row, so compute it once per row
  // instead of repeating it for every output channel and convolution tap.
  std::vector<float> inv_rms(static_cast<size_t>(rows));
  ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(rows), static_cast<double>(hidden_size),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const T* input_row = input_data + row * hidden_size;
          float sum_sq = 0.0f;
          for (int64_t i = 0; i < hidden_size; ++i) {
            const float value = static_cast<float>(input_row[i]);
            sum_sq += value * value;
          }
          inv_rms[static_cast<size_t>(row)] = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + epsilon_);
        }
      });

  ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(total), static_cast<double>(kernel_size),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
          const int64_t c = linear % hidden_size;
          const int64_t g = (linear / hidden_size) % hc_mult;
          const int64_t t = (linear / channels) % sequence_length;
          const int64_t b = linear / (sequence_length * channels);
          const int64_t flat_channel = g * hidden_size + c;
          const float scale = static_cast<float>(scale_data[flat_channel]);

          float sum = bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[flat_channel]);
          for (int64_t k = 0; k < kernel_size; ++k) {
            const int64_t source_t = t - (kernel_size - 1 - k) * dilation_;
            float normed;
            if (source_t >= 0) {
              const int64_t source_row = (b * sequence_length + source_t) * hc_mult + g;
              normed = static_cast<float>(input_data[source_row * hidden_size + c]) *
                       inv_rms[static_cast<size_t>(source_row)] * scale;
            } else if (past_data != nullptr) {
              // past_state is right-aligned, so position -1 is its last slot.
              const int64_t slot = state_length + source_t;
              if (slot < 0) {
                continue;
              }
              normed = static_cast<float>(past_data[((b * state_length + slot) * hc_mult + g) * hidden_size + c]);
            } else {
              continue;
            }
            sum += normed * static_cast<float>(weight_data[flat_channel * kernel_size + k]);
          }
          output_data[linear] = static_cast<T>(apply_silu ? kernel_helper::SiluFloat(sum) : sum);
        }
      });

  if (present_state != nullptr && state_length > 0) {
    T* present_data = present_state->MutableData<T>();
    ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), narrow<ptrdiff_t>(batch_size * state_length * hc_mult),
        static_cast<double>(hidden_size),
        [&](ptrdiff_t begin, ptrdiff_t end) {
          for (int64_t slot_row = begin; slot_row < end; ++slot_row) {
            const int64_t g = slot_row % hc_mult;
            const int64_t slot = (slot_row / hc_mult) % state_length;
            const int64_t b = slot_row / (state_length * hc_mult);
            // Virtual position of this slot relative to the end of the current chunk.
            const int64_t source_t = sequence_length - state_length + slot;
            T* present_row = present_data + slot_row * hidden_size;

            if (source_t >= 0) {
              const int64_t source_row = (b * sequence_length + source_t) * hc_mult + g;
              const T* input_row = input_data + source_row * hidden_size;
              const float row_inv_rms = inv_rms[static_cast<size_t>(source_row)];
              for (int64_t i = 0; i < hidden_size; ++i) {
                present_row[i] = static_cast<T>(static_cast<float>(input_row[i]) * row_inv_rms *
                                                static_cast<float>(scale_data[g * hidden_size + i]));
              }
            } else if (past_data != nullptr) {
              const int64_t past_slot = state_length + source_t;
              const T* past_row = past_data + ((b * state_length + past_slot) * hc_mult + g) * hidden_size;
              std::copy(past_row, past_row + hidden_size, present_row);
            } else {
              std::fill(present_row, present_row + hidden_size, T{});
            }
          }
        });
  }

  return Status::OK();
}

template class ShortConv<float>;
template class ShortConv<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
