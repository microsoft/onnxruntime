// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
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


template <typename T>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatIfNeeded(const T* p_input, int num_elems);

template <typename T>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatIfNeeded(
  const std::enable_if_t<std::is_same_v<T,float> || std::is_same_v<T, double>, T>* p_input, int num_elems)
{
  return nullptr;
}

template<>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatIfNeeded<MLFloat16>(const MLFloat16* p_input, int num_elems)
{
  if (!p_input) {
    return nullptr;
  }

  // Efficiently convert all the MLFloat16 values to floats.
  std::shared_ptr<std::vector<float>> vec = std::make_shared<std::vector<float>>(num_elems);
  MlasConvertHalfToFloatBuffer(p_input, &(*vec)[0], num_elems);

  return vec;
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

        std::shared_ptr<std::vector<float>> float_input = ConvertHalfToFloatIfNeeded<T>(p_input, hidden_size);
        const DoubleOrFloat* converted_input =
          float_input == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(p_input)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_input)[0]);
        std::shared_ptr<std::vector<float>> float_skip = ConvertHalfToFloatIfNeeded<T>(p_skip, hidden_size);
        const DoubleOrFloat* converted_skip =
          float_skip == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(p_skip)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_skip)[0]);
        std::shared_ptr<std::vector<float>> float_bias = ConvertHalfToFloatIfNeeded<T>(bias_data, hidden_size);
        const DoubleOrFloat* converted_bias =
          float_bias == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(bias_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_bias)[0]);

        std::unique_ptr<DoubleOrFloat[]> output_buffer = std::make_unique<DoubleOrFloat[]>(hidden_size);
        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          DoubleOrFloat value = converted_input[h] + converted_skip[h];

          if (nullptr != bias_data) {
            value += converted_bias[h];
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

        std::shared_ptr<std::vector<float>> float_gamma = ConvertHalfToFloatIfNeeded<T>(gamma_data, hidden_size);
        const DoubleOrFloat* converted_gamma =
          float_gamma == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(gamma_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_gamma)[0]);
        std::shared_ptr<std::vector<float>> float_beta = ConvertHalfToFloatIfNeeded<T>(beta_data, hidden_size);
        const DoubleOrFloat* converted_beta =
          float_beta == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(beta_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_beta)[0]);
        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          if (simplified) {
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>(
              output_buffer[h] / mean_square * converted_gamma[h]);
          } else if (nullptr == beta_data) {
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>(
              (output_buffer[h] - mean) / mean_square * converted_gamma[h]);
          } else {
            p_output[h] = ConvertDoubleOrFloatToMLFloat16IfNeeded<T>(
              (output_buffer[h] - mean) / mean_square * converted_gamma[h] + converted_beta[h]);
          }
        }
      },
      0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
