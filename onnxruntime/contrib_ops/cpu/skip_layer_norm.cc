// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/common/float16.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/force_inline.h"
#include "core/util/narrow_float_utils.h"
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
REGISTER_KERNEL_TYPED(BFloat16)

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
    T* skip_input_bias_add_output_data,
    float* mean_data,
    float* inv_std_var_data) {
  auto offset = task_idx * hidden_size;
  const T* p_input = input_data + offset;
  const T* p_skip = skip_data + (offset % skip_size);
  T* p_output = output_data + offset;
  T* p_skip_input_bias_add_output = skip_input_bias_add_output_data == nullptr ? nullptr : skip_input_bias_add_output_data + offset;

  T mean(0.0f);
  T M2(0.0f);
  T sum_sq(0.0f);

  for (decltype(hidden_size) h = 0; h < hidden_size; h++) {
    T val = p_input[h] + p_skip[h];

    if (nullptr != bias_data) {
      val += bias_data[h];
    }

    if (nullptr != p_skip_input_bias_add_output) {
      p_skip_input_bias_add_output[h] = val;
    }

    p_output[h] = val;
    if (simplified) {
      sum_sq += val * val;
    } else {
      T delta = val - mean;
      mean += delta / static_cast<T>(h + 1);
      T delta2 = val - mean;
      M2 += delta * delta2;
    }
  }

  const T std_dev = simplified
                        ? sqrt(sum_sq / hidden_size + epsilon)
                        : sqrt(M2 / hidden_size + epsilon);

  if (mean_data != nullptr) {
    // Simplified normalization has no centering term.
    mean_data[task_idx] = simplified ? 0.0f : static_cast<float>(mean);
  }
  if (inv_std_var_data != nullptr) {
    inv_std_var_data[task_idx] = static_cast<float>(1 / std_dev);
  }

  for (decltype(hidden_size) h = 0; h < hidden_size; h++) {
    if (simplified) {
      p_output[h] = p_output[h] / std_dev * gamma_data[h];
    } else if (nullptr == beta_data) {
      p_output[h] = (p_output[h] - mean) / std_dev * gamma_data[h];
    } else {
      p_output[h] = (p_output[h] - mean) / std_dev * gamma_data[h] + beta_data[h];
    }
  }
}

}  // namespace

template <typename T, bool simplified>
SkipLayerNorm<T, simplified>::SkipLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      has_prepacked_skip_(false),
      has_prepacked_gamma_(false),
      has_prepacked_beta_(false),
      has_prepacked_bias_(false),
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
  const Tensor* skip = has_prepacked_skip_ ? nullptr : p_ctx->Input<Tensor>(1);
  const Tensor* gamma = has_prepacked_gamma_ ? nullptr : p_ctx->Input<Tensor>(2);
  const Tensor* beta = simplified ? nullptr : (has_prepacked_beta_ ? nullptr : p_ctx->Input<Tensor>(3));
  const Tensor* bias = has_prepacked_bias_ ? nullptr : p_ctx->Input<Tensor>(simplified ? 3 : 4);

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();
  if (input_dims_size != 3 && input_dims_size != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 or 2 dimensions, got ", input_dims_size);
  }

  const int64_t hidden_size_i64 = input_dims[input_dims_size - 1];
  if (hidden_size_i64 <= 0 || hidden_size_i64 > std::numeric_limits<int>::max()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "hidden_size must be positive and no greater than ", std::numeric_limits<int>::max(),
                           ". Got ", hidden_size_i64, ".");
  }
  const int hidden_size = static_cast<int>(hidden_size_i64);

  if (has_prepacked_skip_) {
    ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckSkipShape(input->Shape(), prepacked_skip_shape_));
  }

  if (has_prepacked_gamma_) {
    ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckGammaShape(prepacked_gamma_shape_, hidden_size));
  }

  if (has_prepacked_beta_) {
    ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckBetaShape(prepacked_beta_shape_, hidden_size));
  }

  if (has_prepacked_bias_) {
    ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckBiasShape(prepacked_bias_shape_, hidden_size));
  }

  ORT_RETURN_IF_ERROR(skip_layer_norm_helper::CheckPotentiallyPrepackedInputs<Tensor>(input,
                                                                                      skip,
                                                                                      gamma,
                                                                                      beta,
                                                                                      bias,
                                                                                      hidden_size,
                                                                                      input_dims_size,
                                                                                      has_prepacked_skip_,
                                                                                      has_prepacked_gamma_));

  Tensor* output = p_ctx->Output(0, input->Shape());
  const TensorShape stat_shape([&input_dims]() {
    TensorShapeVector dims(input_dims.begin(), input_dims.end());
    dims.back() = 1;
    return dims;
  }());
  Tensor* mean = p_ctx->Output(1, stat_shape);
  Tensor* inv_std_var = p_ctx->Output(2, stat_shape);
  Tensor* skip_input_bias_add_output = p_ctx->Output(3, input->Shape());

  int64_t task_count = input->Shape().SizeToDimension(input_dims_size - 1);

  const T* input_data = input->Data<T>();
  const T* skip_data = skip == nullptr ? nullptr : skip->Data<T>();
  const T* gamma_data = gamma == nullptr ? nullptr : gamma->Data<T>();
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* output_data = output->MutableData<T>();
  T* skip_input_bias_add_output_data = skip_input_bias_add_output == nullptr ? nullptr : skip_input_bias_add_output->MutableData<T>();
  float* mean_data = mean == nullptr ? nullptr : mean->MutableData<float>();
  float* inv_std_var_data = inv_std_var == nullptr ? nullptr : inv_std_var->MutableData<float>();
  const int64_t skip_size = skip ? skip->Shape().Size() : prepacked_skip_shape_.Size();

  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
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
    NarrowToFloat<T>(input_data, input_fp32.get(), total_data_size);
    input_data_f = input_fp32.get();

    output_fp32 = IAllocator::MakeUniquePtr<float>(alloc, total_data_size);
    output_data_f = output_fp32.get();

    if (skip_input_bias_add_output_data != nullptr) {
      skip_input_bias_add_output_fp32 = IAllocator::MakeUniquePtr<float>(alloc, total_data_size);
      skip_input_bias_add_output_data_f = skip_input_bias_add_output_fp32.get();
    }

    if (skip_data) {
      skip_fp32 = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(skip_size));
      NarrowToFloat<T>(skip_data, skip_fp32.get(), static_cast<size_t>(skip_size));
      skip_data_f = skip_fp32.get();
    } else if (has_prepacked_skip_) {
      skip_data_f = prepacked_skip_fp32_data_.get();
    }

    if (gamma_data) {
      gamma_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      NarrowToFloat<T>(gamma_data, gamma_fp32.get(), num_elems);
      gamma_data_f = gamma_fp32.get();
    } else if (has_prepacked_gamma_) {
      gamma_data_f = prepacked_gamma_fp32_data_.get();
    }

    if (beta_data) {
      beta_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      NarrowToFloat<T>(beta_data, beta_fp32.get(), num_elems);
      beta_data_f = beta_fp32.get();
    } else if (has_prepacked_beta_) {
      beta_data_f = prepacked_beta_fp32_data_.get();
    }

    if (bias_data) {
      bias_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      NarrowToFloat<T>(bias_data, bias_fp32.get(), num_elems);
      bias_data_f = bias_fp32.get();
    } else if (has_prepacked_bias_) {
      bias_data_f = prepacked_bias_fp32_data_.get();
    }

    concurrency::ThreadPool::TryBatchParallelFor(
        p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          ComputeJob(input_data_f, skip_data_f, gamma_data_f, beta_data_f, bias_data_f, task_idx, hidden_size, skip_size,
                     epsilon_, simplified, output_data_f, skip_input_bias_add_output_data_f, mean_data, inv_std_var_data);
        },
        0);
    FloatToNarrow<T>(output_data_f, output_data, total_data_size);
    if (skip_input_bias_add_output_data != nullptr)
      FloatToNarrow<T>(skip_input_bias_add_output_data_f, skip_input_bias_add_output_data, total_data_size);
  } else {
    concurrency::ThreadPool::TryBatchParallelFor(
        p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          ComputeJob(input_data, skip_data, gamma_data, beta_data, bias_data, task_idx, hidden_size, skip_size,
                     epsilon_, simplified, output_data, skip_input_bias_add_output_data, mean_data, inv_std_var_data);
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
    ConvertNarrowFloatToFloatIfNeeded(tensor, alloc, prepacked_skip_fp32_data_, is_packed);
    if (is_packed) {
      prepacked_skip_shape_ = tensor.Shape();
      has_prepacked_skip_ = true;
    }
  } else if (input_idx == 2) {  // gamma
    ConvertNarrowFloatToFloatIfNeeded(tensor, alloc, prepacked_gamma_fp32_data_, is_packed);
    if (is_packed) {
      prepacked_gamma_shape_ = tensor.Shape();
      has_prepacked_gamma_ = true;
    }
  } else if (input_idx == 3) {
    if constexpr (simplified) {
      // bias
      ConvertNarrowFloatToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
      if (is_packed) {
        prepacked_bias_shape_ = tensor.Shape();
        has_prepacked_bias_ = true;
      }
    } else {
      // beta
      ConvertNarrowFloatToFloatIfNeeded(tensor, alloc, prepacked_beta_fp32_data_, is_packed);
      if (is_packed) {
        prepacked_beta_shape_ = tensor.Shape();
        has_prepacked_beta_ = true;
      }
    }
  } else if (input_idx == 4) {  // bias
    ORT_ENFORCE(!simplified, "SkipSimplifiedLayerNormalization should only has 4 inputs (input, skip, gamma, and beta). Got 5.");
    ConvertNarrowFloatToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
    if (is_packed) {
      prepacked_bias_shape_ = tensor.Shape();
      has_prepacked_bias_ = true;
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
