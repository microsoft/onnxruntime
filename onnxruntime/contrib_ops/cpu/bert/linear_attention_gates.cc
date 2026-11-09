// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/linear_attention_gates.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

using namespace onnxruntime::contrib::linear_attention_gates_helper;

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
REGISTER_KERNEL_TYPED(LinearAttentionGate, BFloat16)
REGISTER_KERNEL_TYPED(MoEGroupRouter, float)
REGISTER_KERNEL_TYPED(MoEGroupRouter, MLFloat16)
REGISTER_KERNEL_TYPED(MoEGroupRouter, BFloat16)

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
REGISTER_KERNEL_TYPED(GatedRMSNorm, BFloat16)

#undef REGISTER_KERNEL_TYPED

inline float SigmoidFloat(float value) {
  float output;
  MlasComputeLogistic(&value, &output, 1);
  return output;
}

inline float SoftplusFloat(float value) {
  return value > 0.0f ? value + std::log(std::exp(-value) + 1.0f) : std::log(std::exp(value) + 1.0f);
}

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
  activation_ = ParseGatedRMSNormActivationOrThrow(info.GetAttrOrDefault<std::string>("activation", "silu"));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
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
        if (activation_ == GatedRMSNormActivation::kSilu) {
          for (int64_t i = 0; i < norm_size; ++i) {
            const float z = static_cast<float>(gate_data[offset + i]);
            const float normalized = static_cast<float>(input_data[offset + i]) * inv_rms *
                                     static_cast<float>(scale_data[i]);
            output_data[offset + i] = static_cast<T>(normalized * (z * SigmoidFloat(z)));
          }
        } else {
          for (int64_t i = 0; i < norm_size; ++i) {
            const float z = static_cast<float>(gate_data[offset + i]);
            const float normalized = static_cast<float>(input_data[offset + i]) * inv_rms *
                                     static_cast<float>(scale_data[i]);
            output_data[offset + i] = static_cast<T>(normalized * SigmoidFloat(z));
          }
        }
      },
      0);

  return Status::OK();
}

template class LinearAttentionGate<float>;
template class LinearAttentionGate<MLFloat16>;
template class LinearAttentionGate<BFloat16>;
template class GatedRMSNorm<float>;
template class GatedRMSNorm<MLFloat16>;
template class GatedRMSNorm<BFloat16>;

template <typename T>
MoEGroupRouter<T>::MoEGroupRouter(const OpKernelInfo& info) : OpKernel(info) {
  ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("top_k", &top_k_));
  n_group_ = info.GetAttrOrDefault<int64_t>("n_group", 1);
  topk_group_ = info.GetAttrOrDefault<int64_t>("topk_group", 1);
  group_topk_ = info.GetAttrOrDefault<int64_t>("group_topk", 2);
  router_sentinel_ = info.GetAttrOrDefault<float>("router_sentinel", -10000.0f);
  ORT_THROW_IF_ERROR(ValidateMoEGroupRouterAttributes(top_k_, n_group_, topk_group_, group_topk_));
}

template <typename T>
Status MoEGroupRouter<T>::Compute(OpKernelContext* context) const {
  const Tensor* logits = context->Input<Tensor>(0);
  const Tensor* expert_bias = context->Input<Tensor>(1);
  const auto& in_shape = logits->Shape();
  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "logits must have rank >= 2");
  const int64_t num_experts = in_shape[in_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_experts > 0 && expert_bias->Shape().Size() == num_experts,
                    "expert_bias length must match num_experts");
  ORT_RETURN_IF_NOT(n_group_ > 0 && num_experts % n_group_ == 0 && topk_group_ > 0 &&
                        topk_group_ <= n_group_ && top_k_ <= num_experts,
                    "Invalid MoEGroupRouter group/top_k parameters");

  const int64_t num_rows = in_shape.Size() / num_experts;
  TensorShape out_shape({num_rows, num_experts});
  Tensor* output = context->Output(0, out_shape);
  if (in_shape.Size() == 0) {
    return Status::OK();
  }

  const T* logits_data = logits->Data<T>();
  const float* bias_data = expert_bias->Data<float>();
  T* out_data = output->MutableData<T>();
  const int64_t experts_per_group = num_experts / n_group_;
  const int64_t group_k = std::min(group_topk_, experts_per_group);

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(num_rows),
      [&](ptrdiff_t row) {
        const T* row_in = logits_data + row * num_experts;
        T* row_out = out_data + row * num_experts;
        std::vector<float> scores(static_cast<size_t>(num_experts));
        std::vector<float> biased(static_cast<size_t>(num_experts));
        for (int64_t e = 0; e < num_experts; ++e) {
          scores[static_cast<size_t>(e)] = SigmoidFloat(static_cast<float>(row_in[e]));
          biased[static_cast<size_t>(e)] = scores[static_cast<size_t>(e)] + bias_data[e];
          row_out[e] = static_cast<T>(router_sentinel_);
        }
        std::vector<std::pair<float, int64_t>> grp_scores(static_cast<size_t>(n_group_));
        std::vector<float> grp_buf(static_cast<size_t>(experts_per_group));
        for (int64_t g = 0; g < n_group_; ++g) {
          for (int64_t i = 0; i < experts_per_group; ++i) {
            grp_buf[static_cast<size_t>(i)] = biased[static_cast<size_t>(g * experts_per_group + i)];
          }
          std::partial_sort(grp_buf.begin(), grp_buf.begin() + onnxruntime::narrow<ptrdiff_t>(group_k), grp_buf.end(),
                            std::greater<float>());
          float sum = 0.0f;
          for (int64_t i = 0; i < group_k; ++i) sum += grp_buf[static_cast<size_t>(i)];
          grp_scores[static_cast<size_t>(g)] = {sum, g};
        }
        std::partial_sort(grp_scores.begin(), grp_scores.begin() + onnxruntime::narrow<ptrdiff_t>(topk_group_), grp_scores.end(),
                          [](const auto& a, const auto& b) {
                            return a.first != b.first ? a.first > b.first : a.second < b.second;
                          });
        std::vector<std::pair<float, int64_t>> candidates;
        candidates.reserve(static_cast<size_t>(topk_group_ * experts_per_group));
        for (int64_t gi = 0; gi < topk_group_; ++gi) {
          const int64_t g = grp_scores[static_cast<size_t>(gi)].second;
          for (int64_t i = 0; i < experts_per_group; ++i) {
            const int64_t e = g * experts_per_group + i;
            candidates.emplace_back(biased[static_cast<size_t>(e)], e);
          }
        }
        const int64_t eff_k = std::min<int64_t>(top_k_, static_cast<int64_t>(candidates.size()));
        std::partial_sort(candidates.begin(), candidates.begin() + onnxruntime::narrow<ptrdiff_t>(eff_k), candidates.end(),
                          [](const auto& a, const auto& b) {
                            return a.first != b.first ? a.first > b.first : a.second < b.second;
                          });
        for (int64_t k = 0; k < eff_k; ++k) {
          const int64_t e = candidates[static_cast<size_t>(k)].second;
          const float s = std::max(scores[static_cast<size_t>(e)], 1e-20f);
          row_out[e] = static_cast<T>(std::log(s));
        }
      },
      0);

  return Status::OK();
}

template class MoEGroupRouter<float>;
template class MoEGroupRouter<MLFloat16>;
template class MoEGroupRouter<BFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
