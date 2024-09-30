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


namespace {

double* CreateBufferIfMLFloat16(double* p_output, int num_elems)
{
  return p_output;
}

float* CreateBufferIfMLFloat16(float* p_output, int num_elems)
{
  return p_output;
}

float* CreateBufferIfMLFloat16(MLFloat16* p_output, int num_elems)
{
  if (!p_output) {
    return nullptr;
  }

  return new float[num_elems];
}


template <typename T>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatBufferIfNeeded(const T* p_input, int num_elems);

template <typename T>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatBufferIfNeeded(
  const std::enable_if_t<std::is_same_v<T,float> || std::is_same_v<T, double>, T>* p_input, int num_elems)
{
  return nullptr;
}

template<>
std::shared_ptr<std::vector<float>> ConvertHalfToFloatBufferIfNeeded<MLFloat16>(const MLFloat16* p_input, int num_elems)
{
  if (!p_input) {
    return nullptr;
  }

  // Efficiently convert all the MLFloat16 values to floats.
  std::shared_ptr<std::vector<float>> vec = std::make_shared<std::vector<float>>(num_elems);
  MlasConvertHalfToFloatBuffer(p_input, &(*vec)[0], num_elems);

  return vec;
}


void ConvertFloatBufferToMLFloat16(const float* output_buffer, MLFloat16* p_output, int num_elems)
{
  if (!output_buffer || !p_output) {
    return;
  }

  MlasConvertFloatToHalfBuffer(output_buffer, p_output, num_elems);
}

} // namespace


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

        std::shared_ptr<std::vector<float>> float_input = ConvertHalfToFloatBufferIfNeeded<T>(p_input, hidden_size);
        const DoubleOrFloat* converted_input =
          float_input == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(p_input)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_input)[0]);
        std::shared_ptr<std::vector<float>> float_skip = ConvertHalfToFloatBufferIfNeeded<T>(p_skip, hidden_size);
        const DoubleOrFloat* converted_skip =
          float_skip == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(p_skip)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_skip)[0]);
        std::shared_ptr<std::vector<float>> float_bias = ConvertHalfToFloatBufferIfNeeded<T>(bias_data, hidden_size);
        const DoubleOrFloat* converted_bias =
          float_bias == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(bias_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_bias)[0]);

        // If T is float or double, then output_buffer will be the same as p_output, so we don't allocate new memory.
        // If T is MLFloat16, then we allocate hidden_size floats in output_buffer.
        DoubleOrFloat* output_buffer = static_cast<DoubleOrFloat*>(CreateBufferIfMLFloat16(p_output, hidden_size));

        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          DoubleOrFloat val = converted_input[h] + converted_skip[h];

          if (nullptr != bias_data) {
            val += converted_bias[h];
          }

          output_buffer[h] = val;
          mean += val;
          mean_square += val * val;

          if (nullptr != p_skip_input_bias_add_output_data && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
            p_skip_input_bias_add_output_data[h] = *(reinterpret_cast<T*>(&val));
          }
        }

        if (nullptr != p_skip_input_bias_add_output_data && std::is_same_v<T, MLFloat16>) {
          ConvertFloatBufferToMLFloat16(reinterpret_cast<float*>(output_buffer),
                                        reinterpret_cast<MLFloat16*>(p_skip_input_bias_add_output_data),
                                        hidden_size);
        }

        mean = mean / hidden_size;
        if (simplified) {
          mean_square = sqrt(mean_square / hidden_size + epsilon_);
        } else {
          mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon_);
        }

        std::shared_ptr<std::vector<float>> float_gamma = ConvertHalfToFloatBufferIfNeeded<T>(gamma_data, hidden_size);
        const DoubleOrFloat* converted_gamma =
          float_gamma == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(gamma_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_gamma)[0]);
        std::shared_ptr<std::vector<float>> float_beta = ConvertHalfToFloatBufferIfNeeded<T>(beta_data, hidden_size);
        const DoubleOrFloat* converted_beta =
          float_beta == nullptr
          ? reinterpret_cast<const DoubleOrFloat*>(beta_data)
          : reinterpret_cast<const DoubleOrFloat*>(&(*float_beta)[0]);
        for (size_t h = 0; h < static_cast<size_t>(hidden_size); h++) {
          if (simplified) {
            output_buffer[h] = output_buffer[h] / mean_square * converted_gamma[h];
          } else if (nullptr == beta_data) {
            output_buffer[h] = (output_buffer[h] - mean) / mean_square * converted_gamma[h];
          } else {
            output_buffer[h] = (output_buffer[h] - mean) / mean_square * converted_gamma[h] + converted_beta[h];
          }
        }

        if (std::is_same_v<decltype(p_output), MLFloat16>) {
          ConvertFloatBufferToMLFloat16(
            reinterpret_cast<float*>(output_buffer), reinterpret_cast<MLFloat16*>(p_output), hidden_size);
          delete[] output_buffer;
        }
      },
      0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
