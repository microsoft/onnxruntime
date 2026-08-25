// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/short_conv.h"

#include <cmath>

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

#undef REGISTER_SHORT_CONV_TYPED

namespace {

inline float SigmoidFloat(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

inline float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

}  // namespace

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

  Tensor* output = context->Output(0, input_shape);
  if (input_shape.Size() == 0) {
    return Status::OK();
  }

  const T* input_data = input->Data<T>();
  const T* weight_data = weight->Data<T>();
  const T* scale_data = norm_scale->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();
  T* output_data = output->MutableData<T>();
  const bool apply_silu = activation_ == "silu" || activation_ == "swish";
  const int64_t total = batch_size * sequence_length * channels;

  ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(total), static_cast<double>(kernel_size * hidden_size),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
          const int64_t c = linear % hidden_size;
          const int64_t g = (linear / hidden_size) % hc_mult;
          const int64_t t = (linear / channels) % sequence_length;
          const int64_t b = linear / (sequence_length * channels);
          const int64_t flat_channel = g * hidden_size + c;

          float sum = bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[flat_channel]);
          for (int64_t k = 0; k < kernel_size; ++k) {
            const int64_t source_t = t - (kernel_size - 1 - k) * dilation_;
            if (source_t < 0) {
              continue;
            }

            const int64_t row_base = ((b * sequence_length + source_t) * hc_mult + g) * hidden_size;
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden_size; ++i) {
              const float value = static_cast<float>(input_data[row_base + i]);
              sum_sq += value * value;
            }
            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + epsilon_);
            const float normed = static_cast<float>(input_data[row_base + c]) * inv_rms *
                                 static_cast<float>(scale_data[g * hidden_size + c]);
            sum += normed * static_cast<float>(weight_data[flat_channel * kernel_size + k]);
          }
          output_data[linear] = static_cast<T>(apply_silu ? SiluFloat(sum) : sum);
        }
      });

  return Status::OK();
}

template class ShortConv<float>;

}  // namespace contrib
}  // namespace onnxruntime
