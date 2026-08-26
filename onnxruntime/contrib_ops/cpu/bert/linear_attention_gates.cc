// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/linear_attention_gates.h"

#include <cmath>

#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(Op, T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      Op,                                                              \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kCpuExecutionProvider,                                           \
      KernelDefBuilder()                                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("TF", DataTypeImpl::GetTensorType<float>()), \
      Op<T>);

REGISTER_KERNEL_TYPED(LinearAttentionGate, float)
REGISTER_KERNEL_TYPED(LinearAttentionGate, MLFloat16)

#undef REGISTER_KERNEL_TYPED

#define REGISTER_KERNEL_TYPED(Op, T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Op,                                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Op<T>);

REGISTER_KERNEL_TYPED(GatedRMSNorm, float)
REGISTER_KERNEL_TYPED(GatedRMSNorm, MLFloat16)

#undef REGISTER_KERNEL_TYPED

namespace {

inline float SigmoidFloat(float value) {
  float output;
  MlasComputeLogistic(&value, &output, 1);
  return output;
}

inline float SoftplusFloat(float value) {
  return value > 0.0f ? value + std::log(std::exp(-value) + 1.0f) : std::log(std::exp(value) + 1.0f);
}

}  // namespace

template <typename T>
Status LinearAttentionGate<T>::Compute(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* dt_bias = context->Input<Tensor>(1);
  const Tensor* decay_scale = context->Input<Tensor>(2);
  const Tensor* b = context->Input<Tensor>(3);  // optional

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "a must have rank >= 1");
  const int64_t num_heads = a_shape[a_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_heads > 0, "a last dimension must be positive");

  ORT_RETURN_IF_NOT(dt_bias->Shape().Size() == num_heads,
                    "dt_bias must have ", num_heads, " elements, got ", dt_bias->Shape().Size());
  ORT_RETURN_IF_NOT(decay_scale->Shape().Size() == num_heads,
                    "decay_scale must have ", num_heads, " elements, got ", decay_scale->Shape().Size());

  Tensor* decay = context->Output(0, a_shape);
  Tensor* beta = context->Output(1, a_shape);

  if (beta != nullptr) {
    ORT_RETURN_IF_NOT(b != nullptr, "The b input is required when the beta output is requested");
    ORT_RETURN_IF_NOT(b->Shape() == a_shape, "b must have the same shape as a");
  }

  const int64_t count = a_shape.Size();
  if (count == 0) {
    return Status::OK();
  }

  const T* a_data = a->Data<T>();
  const T* b_data = b == nullptr ? nullptr : b->Data<T>();
  const float* dt_bias_data = dt_bias->Data<float>();
  const float* decay_scale_data = decay_scale->Data<float>();
  T* decay_data = decay->MutableData<T>();
  T* beta_data = beta == nullptr ? nullptr : beta->MutableData<T>();

  const int64_t num_tokens = count / num_heads;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(num_tokens),
      [&](ptrdiff_t token) {
        const int64_t offset = token * num_heads;
        for (int64_t h = 0; h < num_heads; ++h) {
          const int64_t idx = offset + h;
          const float biased = static_cast<float>(a_data[idx]) + dt_bias_data[h];
          decay_data[idx] = static_cast<T>(decay_scale_data[h] * SoftplusFloat(biased));
          if (beta_data != nullptr) {
            beta_data[idx] = static_cast<T>(SigmoidFloat(static_cast<float>(b_data[idx])));
          }
        }
      },
      0);

  return Status::OK();
}

template <typename T>
GatedRMSNorm<T>::GatedRMSNorm(const OpKernelInfo& info) : OpKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
  const std::string activation = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation == "silu" || activation == "sigmoid",
              "GatedRMSNorm: activation must be 'silu' or 'sigmoid', got '", activation, "'");
  use_sigmoid_activation_ = activation == "sigmoid";
}

template <typename T>
Status GatedRMSNorm<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  const Tensor* gate = context->Input<Tensor>(2);

  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(gate->Shape() == shape, "gate must have the same shape as X");

  const int64_t norm_size = scale->Shape().Size();
  ORT_RETURN_IF_NOT(norm_size > 0, "scale must not be empty");
  const int64_t last_dim = shape[shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(last_dim % norm_size == 0,
                    "X last dimension (", last_dim, ") must be a multiple of the scale length (",
                    norm_size, ")");

  Tensor* output = context->Output(0, shape);
  const int64_t count = shape.Size();
  if (count == 0) {
    return Status::OK();
  }
  const int64_t num_rows = count / norm_size;

  const T* input_data = input->Data<T>();
  const T* scale_data = scale->Data<T>();
  const T* gate_data = gate->Data<T>();
  T* output_data = output->MutableData<T>();

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(num_rows),
      [&](ptrdiff_t row) {
        const int64_t offset = row * norm_size;
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
          const float v = static_cast<float>(input_data[offset + i]);
          sum_sq += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon_);
        for (int64_t i = 0; i < norm_size; ++i) {
          const float z = static_cast<float>(gate_data[offset + i]);
          const float normalized = static_cast<float>(input_data[offset + i]) * inv_rms *
                                   static_cast<float>(scale_data[i]);
          const float activated = use_sigmoid_activation_ ? SigmoidFloat(z) : (z * SigmoidFloat(z));
          output_data[offset + i] = static_cast<T>(normalized * activated);
        }
      },
      0);

  return Status::OK();
}

template class LinearAttentionGate<float>;
template class LinearAttentionGate<MLFloat16>;
template class GatedRMSNorm<float>;
template class GatedRMSNorm<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
