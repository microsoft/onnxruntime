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

template <typename T, typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>, void>>
void ComputeJob(
    const T* input_data,
    const T* skip_data,
    const T* gamma_data,
    const T* beta_data,
    const T* bias_data,
    ptrdiff_t task_idx,
    int hidden_size,
    int64_t skip_size,
    float epsilon,
    bool simplified,
    T* output_data,
    T* skip_input_bias_add_output_data) {
  auto offset = task_idx * hidden_size;
  const T* p_input = input_data + offset;
  const T* p_skip = skip_data + (offset % skip_size);
  T* p_output = output_data + offset;
  T* p_skip_input_bias_add_output = skip_input_bias_add_output_data == nullptr ? nullptr : skip_input_bias_add_output_data + offset;

  T mean(0.0f);
  T mean_square(0.0f);

  for (decltype(hidden_size) h = 0; h < hidden_size; h++) {
    T val = p_input[h] + p_skip[h];

    if (nullptr != bias_data) {
      val += bias_data[h];
    }

    if (nullptr != p_skip_input_bias_add_output) {
      p_skip_input_bias_add_output[h] = val;
    }

    p_output[h] = val;
    mean += val;
    mean_square += val * val;
  }

  mean = mean / hidden_size;
  if (simplified) {
    mean_square = sqrt(mean_square / hidden_size + epsilon);
  } else {
    mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon);
  }

  for (decltype(hidden_size) h = 0; h < hidden_size; h++) {
    if (simplified) {
      p_output[h] = p_output[h] / mean_square * gamma_data[h];
    } else if (nullptr == beta_data) {
      p_output[h] = (p_output[h] - mean) / mean_square * gamma_data[h];
    } else {
      p_output[h] = (p_output[h] - mean) / mean_square * gamma_data[h] + beta_data[h];
    }
  }
}

void ConvertMLFloat16ToFloatIfNeeded(const Tensor& tensor, AllocatorPtr alloc, IAllocatorUniquePtr<float>& dest, bool& is_packed) {
  if (tensor.GetElementType() == utils::ToTensorProtoElementType<MLFloat16>()) {
    auto tensor_data_ptr = tensor.Data<MLFloat16>();
    auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
    auto float_ptr = IAllocator::MakeUniquePtr<float>(alloc, tensor_size, true);

    MlasConvertHalfToFloatBuffer(tensor_data_ptr, float_ptr.get(), tensor_size);
    dest = std::move(float_ptr);
    is_packed = true;
  }
}

}  // namespace

template <typename T, bool simplified>
SkipLayerNorm<T, simplified>::SkipLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      prepacked_skip_fp32_size_(0),
      prepacked_skip_fp32_data_(nullptr),
      prepacked_gamma_fp32_data_(nullptr),
      prepacked_beta_fp32_data_(nullptr),
      prepacked_bias_fp32_data_(nullptr) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

template <typename T, bool simplified>
Status SkipLayerNorm<T, simplified>::Compute(OpKernelContext* p_ctx) const {
  const Tensor* input = p_ctx->Input<Tensor>(0);
  const Tensor* skip = prepacked_skip_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(1);
  const Tensor* gamma = prepacked_gamma_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(2);
  const Tensor* beta = simplified ? nullptr : (prepacked_beta_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(3));
  const Tensor* bias = prepacked_bias_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(simplified ? 3 : 4);
  Tensor* output = p_ctx->Output(0, input->Shape());
  // For inferencing, we support one more optional output which is the sum of the input and skip tensors
  Tensor* skip_input_bias_add_output = p_ctx->Output(3, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();
  int hidden_size = static_cast<int>(input_dims[input_dims_size - 1]);

  ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckPotentiallyPrepackedInputs<Tensor>(input,
                                                                                      skip,
                                                                                      gamma,
                                                                                      beta,
                                                                                      bias,
                                                                                      hidden_size,
                                                                                      input_dims_size,
                                                                                      prepacked_skip_fp32_data_ != nullptr,
                                                                                      prepacked_gamma_fp32_data_ != nullptr));

  int64_t task_count = input->Shape().SizeToDimension(input_dims_size - 1);

  const T* input_data = input->Data<T>();
  const T* skip_data = skip == nullptr ? nullptr : skip->Data<T>();
  const T* gamma_data = gamma == nullptr ? nullptr : gamma->Data<T>();
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* output_data = output->MutableData<T>();

  // For inferencing, we support one more optional output which is the sum of the input and skip tensors
  T* skip_input_bias_add_output_data = skip_input_bias_add_output == nullptr ? nullptr : skip_input_bias_add_output->MutableData<T>();
  const int64_t skip_size = skip ? skip->Shape().Size() : prepacked_skip_fp32_size_;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    const size_t total_data_size = static_cast<size_t>(input->Shape().Size());

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(p_ctx->GetTempSpaceAllocator(&alloc));

    IAllocatorUniquePtr<float> input_fp32;
    IAllocatorUniquePtr<float> output_fp32;
    IAllocatorUniquePtr<float> skip_input_bias_add_output_fp32;
    IAllocatorUniquePtr<float> skip_fp32;
    IAllocatorUniquePtr<float> gamma_fp32;
    IAllocatorUniquePtr<float> beta_fp32;
    IAllocatorUniquePtr<float> bias_fp32;

    const float* input_data_f = nullptr;
    const float* skip_data_f = nullptr;
    const float* gamma_data_f = nullptr;
    const float* beta_data_f = nullptr;
    const float* bias_data_f = nullptr;
    float* output_data_f = nullptr;
    float* skip_input_bias_add_output_data_f = nullptr;

    const size_t num_elems = static_cast<size_t>(hidden_size);

    input_fp32 = IAllocator::MakeUniquePtr<float>(alloc, total_data_size);
    MlasConvertHalfToFloatBuffer(input_data, input_fp32.get(), total_data_size);
    input_data_f = input_fp32.get();

    output_fp32 = IAllocator::MakeUniquePtr<float>(alloc, total_data_size);
    output_data_f = output_fp32.get();

    skip_input_bias_add_output_fp32 = IAllocator::MakeUniquePtr<float>(alloc, total_data_size);
    skip_input_bias_add_output_data_f = skip_input_bias_add_output_fp32.get();

    if (skip_data) {
      skip_fp32 = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(skip_size));
      MlasConvertHalfToFloatBuffer(skip_data, skip_fp32.get(), static_cast<size_t>(skip_size));
      skip_data_f = skip_fp32.get();
    } else if (prepacked_skip_fp32_data_) {
      skip_data_f = prepacked_skip_fp32_data_.get();
    }

    if (gamma_data) {
      gamma_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(gamma_data, gamma_fp32.get(), num_elems);
      gamma_data_f = gamma_fp32.get();
    } else if (prepacked_gamma_fp32_data_) {
      gamma_data_f = prepacked_gamma_fp32_data_.get();
    }

    if (beta_data) {
      beta_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(beta_data, beta_fp32.get(), num_elems);
      beta_data_f = beta_fp32.get();
    } else if (prepacked_beta_fp32_data_) {
      beta_data_f = prepacked_beta_fp32_data_.get();
    }

    if (bias_data) {
      bias_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(bias_data, bias_fp32.get(), num_elems);
      bias_data_f = bias_fp32.get();
    } else if (prepacked_bias_fp32_data_) {
      bias_data_f = prepacked_bias_fp32_data_.get();
    }

    concurrency::ThreadPool::TryBatchParallelFor(
        p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          ComputeJob(input_data_f, skip_data_f, gamma_data_f, beta_data_f, bias_data_f, task_idx, hidden_size, skip_size,
                     epsilon_, simplified, output_data_f, skip_input_bias_add_output_data_f);
        },
        0);
    MlasConvertFloatToHalfBuffer(output_data_f, output_data, total_data_size);
    if (skip_input_bias_add_output_data != nullptr)
      MlasConvertFloatToHalfBuffer(skip_input_bias_add_output_data_f, skip_input_bias_add_output_data, total_data_size);
  } else {
    concurrency::ThreadPool::TryBatchParallelFor(
        p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          ComputeJob(input_data, skip_data, gamma_data, beta_data, bias_data, task_idx, hidden_size, skip_size,
                     epsilon_, simplified, output_data, skip_input_bias_add_output_data);
        },
        0);
  }

  return Status::OK();
}

template <typename T, bool simplified>
Status SkipLayerNorm<T, simplified>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                             bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;
  if (input_idx == 1) {  // skip
    prepacked_skip_fp32_size_ = tensor.Shape().Size();
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_skip_fp32_data_, is_packed);
  } else if (input_idx == 2) {  // gamma
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_gamma_fp32_data_, is_packed);
  } else if (input_idx == 3) {
    if constexpr (simplified) {
      // bias
      ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
    } else {
      // beta
      ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_beta_fp32_data_, is_packed);
    }
  } else if (input_idx == 4) {  // bias
    ORT_ENFORCE(!simplified, "SkipSimplifiedLayerNormalization should only has 4 inputs (input, skip, gamma, and beta). Got 5.");
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
