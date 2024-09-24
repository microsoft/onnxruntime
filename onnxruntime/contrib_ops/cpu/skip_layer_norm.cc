// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/force_inline.h"
#include "skip_layer_norm.h"
#include "skip_layer_norm_helper.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipLayerNormalization,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, false>);                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipSimplifiedLayerNormalization,                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

// Utility to convert from MLFloat16 to float only when the input type is MLFloat16.
template <typename T, typename Ret>
ORT_FORCEINLINE Ret ConvertMLFloat16ToDoubleOrFloatIfNeeded(T val);

template <>
ORT_FORCEINLINE float ConvertMLFloat16ToDoubleOrFloatIfNeeded<MLFloat16, float>(MLFloat16 val) {
  return val.ToFloat();
}

template <>
ORT_FORCEINLINE double ConvertMLFloat16ToDoubleOrFloatIfNeeded<MLFloat16, double>(MLFloat16 val) {
  return static_cast<double>(ConvertMLFloat16ToDoubleOrFloatIfNeeded<MLFloat16, float>(val));
}

template <>
ORT_FORCEINLINE constexpr float ConvertMLFloat16ToDoubleOrFloatIfNeeded<float, float>(float val) {
  return val;
}

template <>
ORT_FORCEINLINE constexpr double ConvertMLFloat16ToDoubleOrFloatIfNeeded<double, double>(double val) {
  return val;
}

// Function template that only converts the input value to MLFloat16 if T is MLFloat16.
template <typename T>
ORT_FORCEINLINE constexpr typename std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>, T>
ConvertDoubleOrFloatToMLFloat16IfNeeded(T val) {
  return val;
}

template <typename T>
ORT_FORCEINLINE constexpr typename std::enable_if_t<std::is_same_v<T, MLFloat16>, T>
ConvertDoubleOrFloatToMLFloat16IfNeeded(float val) {
  return MLFloat16(val);
}

template <typename T>
ORT_FORCEINLINE constexpr typename std::enable_if_t<std::is_same_v<T, MLFloat16>, T>
ConvertDoubleOrFloatToMLFloat16IfNeeded(double val) {
  return MLFloat16(static_cast<float>(val));
}

template <typename T, bool simplified>
SkipLayerNorm<T, simplified>::SkipLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

template <typename T, bool simplified>
Status SkipLayerNorm<T, simplified>::Compute(OpKernelContext* p_ctx) const {
  const Tensor* input = p_ctx->Input<Tensor>(0);
  const Tensor* skip = p_ctx->Input<Tensor>(1);
  const Tensor* gamma = p_ctx->Input<Tensor>(2);
  const Tensor* beta = p_ctx->Input<Tensor>(3);
  const Tensor* bias = p_ctx->Input<Tensor>(4);
  Tensor* output = p_ctx->Output(0, input->Shape());
  // For inferencing, we support one more optional output which is the sum
  // of the input and skip tensors
  Tensor* skip_input_bias_add_output = p_ctx->Output(3, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();
  int hidden_size = static_cast<int>(input_dims[input_dims_size - 1]);

  ORT_RETURN_IF_ERROR(onnxruntime::contrib::skip_layer_norm_helper::CheckInputs<Tensor>(input,
                                                                                        skip,
                                                                                        gamma,
                                                                                        beta,
                                                                                        bias,
                                                                                        hidden_size,
                                                                                        input_dims_size));

  int64_t task_count = input->Shape().SizeToDimension(input_dims_size - 1);

  const T* input_data = input->Data<T>();
  const T* skip_data = skip->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* output_data = output->MutableData<T>();

  // For inferencing, we support one more optional output which is the sum
  // of the input and skip tensors
  T* skip_input_bias_add_output_data = skip_input_bias_add_output != nullptr ? skip_input_bias_add_output->MutableData<T>() : nullptr;

  const auto& skip_size = skip->Shape().Size();

  concurrency::ThreadPool::TryBatchParallelFor(
      p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        auto offset = task_idx * hidden_size;

        const T* p_input = input_data + offset;
        const T* p_skip = skip_data + (offset % skip_size);
        T* p_output = output_data + offset;
        T* p_skip_input_bias_add_output_data = skip_input_bias_add_output_data != nullptr ? skip_input_bias_add_output_data + offset : nullptr;

        using DoubleOrFloat = typename std::conditional<
            std::is_same<T, double>::value,  // If T is double
            double,                          // Use double
            float                            // Otherwise, use float (covers float and MLFloat16)
            >::type;

        DoubleOrFloat mean(0.0f);
        DoubleOrFloat mean_square(0.0f);

        std::unique_ptr<DoubleOrFloat[]> output_buffer = std::make_unique<DoubleOrFloat[]>(hidden_size);
        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          DoubleOrFloat input_value = ConvertMLFloat16ToDoubleOrFloatIfNeeded<T, DoubleOrFloat>(p_input[h]);
          DoubleOrFloat skip_value = ConvertMLFloat16ToDoubleOrFloatIfNeeded<T, DoubleOrFloat>(p_skip[h]);

          DoubleOrFloat value = input_value + skip_value;

          if (nullptr != bias_data) {
            value += ConvertMLFloat16ToDoubleOrFloatIfNeeded<T, DoubleOrFloat>(bias_data[h]);
          }

          output_buffer[h] = value;
          T converted_value = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>(value);
          if (nullptr != p_skip_input_bias_add_output_data) {
            p_skip_input_bias_add_output_data[h] = converted_value;
          }

          mean += value;
          mean_square += value * value;
        }

        mean = mean / hidden_size;
        if (simplified) {
          mean_square = sqrt(mean_square / hidden_size + epsilon_);
        } else {
          mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon_);
        }

        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          DoubleOrFloat gamma_value = ConvertMLFloat16ToDoubleOrFloatIfNeeded<T, DoubleOrFloat>(gamma_data[h]);
          if (simplified) {
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>(output_buffer[h] / mean_square * gamma_value);
          } else if (nullptr == beta_data) {
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>((output_buffer[h] - mean) / mean_square * gamma_value);
          } else {
            DoubleOrFloat beta_value = ConvertMLFloat16ToDoubleOrFloatIfNeeded<T, DoubleOrFloat>(beta_data[h]);
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>((output_buffer[h] - mean) / mean_square * gamma_value + beta_value);
          }
        }
      },
      0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
