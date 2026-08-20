// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_linear_attention.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T, G_TYPES)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      VarlenLinearAttention,                                                   \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("G", G_TYPES)                                        \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())           \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())         \
          .MayInplace(4, 1),                                                   \
      VarlenLinearAttention<T>);

REGISTER_KERNEL_TYPED(float, DataTypeImpl::GetTensorType<float>())
REGISTER_KERNEL_TYPED(MLFloat16, (BuildKernelDefConstraints<MLFloat16, float>()))
REGISTER_KERNEL_TYPED(BFloat16, (BuildKernelDefConstraints<BFloat16, float>()))

#undef REGISTER_KERNEL_TYPED

template <typename T>
VarlenLinearAttention<T>::VarlenLinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");

  decay_activation_ = info.GetAttrOrDefault<std::string>("decay_activation", "none");
  ORT_ENFORCE(decay_activation_ == "none" || decay_activation_ == "softplus_decay",
              "decay_activation must be one of: none, softplus_decay");

  beta_activation_ = info.GetAttrOrDefault<std::string>("beta_activation", "none");
  ORT_ENFORCE(beta_activation_ == "none" || beta_activation_ == "sigmoid" ||
                  beta_activation_ == "twice_sigmoid",
              "beta_activation must be one of: none, sigmoid, twice_sigmoid");

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  const int64_t max_checkpoints = info.GetAttrOrDefault<int64_t>("max_checkpoints", 0);
  ORT_ENFORCE(max_checkpoints >= 0 && max_checkpoints <= 8,
              "max_checkpoints must be in [0, 8]");
  max_checkpoints_ = static_cast<int>(max_checkpoints);
}

namespace {

Status CheckTokenGateShape(const Tensor& tensor, int64_t total_tokens, int v_num_heads,
                           int d_k, bool allow_key_dim, bool& per_key_dim,
                           const char* name) {
  const auto& shape = tensor.Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 || (allow_key_dim && shape.NumDimensions() == 3),
                    name, " must have shape [N,Hv]",
                    allow_key_dim ? " or [N,Hv,K]" : "");
  ORT_RETURN_IF_NOT(shape[0] == total_tokens, name, " token dimension must equal N");
  ORT_RETURN_IF_NOT(shape[1] == v_num_heads, name, " head dimension must equal Hv");
  per_key_dim = shape.NumDimensions() == 3;
  if (per_key_dim) {
    ORT_RETURN_IF_NOT(shape[2] == d_k, name, " key dimension must equal K");
  }
  return Status::OK();
}

Status CheckDecayParamShape(const Tensor& tensor, int v_num_heads, int d_k,
                            bool& per_key_dim, const char* name) {
  const auto& shape = tensor.Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 1 || shape.NumDimensions() == 2,
                    name, " must have shape [Hv] or [Hv,K]");
  ORT_RETURN_IF_NOT(shape[0] == v_num_heads, name, " head dimension must equal Hv");
  per_key_dim = shape.NumDimensions() == 2;
  if (per_key_dim) {
    ORT_RETURN_IF_NOT(shape[1] == d_k, name, " key dimension must equal K");
  }
  return Status::OK();
}

}  // namespace

template <typename T>
Status VarlenLinearAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* initial_state = context->Input<Tensor>(4);
  const Tensor* decay = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* a_log = context->Input<Tensor>(7);
  const Tensor* dt_bias = context->Input<Tensor>(8);

  ORT_RETURN_IF_NOT(query && key && value && cu_seqlens && initial_state,
                    "query, key, value, cumulative_sequence_length, and initial_state are required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();
  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 3, "query must have rank 3 [N,Hq,K]");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 3, "key must have rank 3 [N,Hk,K]");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 3, "value must have rank 3 [N,Hv,V]");

  const int64_t total_tokens_64 = q_shape[0];
  ORT_RETURN_IF_NOT(total_tokens_64 > 0 && total_tokens_64 <= std::numeric_limits<int>::max(),
                    "N must be in [1, INT_MAX]");
  ORT_RETURN_IF_NOT(k_shape[0] == total_tokens_64 && v_shape[0] == total_tokens_64,
                    "query, key, and value must have the same N");
  ORT_RETURN_IF_NOT(q_shape[2] > 0 && q_shape[2] <= std::numeric_limits<int>::max(),
                    "K must be in [1, INT_MAX]");
  ORT_RETURN_IF_NOT(k_shape[2] == q_shape[2], "query and key must have the same K");
  ORT_RETURN_IF_NOT(v_shape[2] > 0 && v_shape[2] <= std::numeric_limits<int>::max(),
                    "V must be in [1, INT_MAX]");
  ORT_RETURN_IF_NOT(q_shape[1] > 0 && q_shape[1] <= std::numeric_limits<int>::max() &&
                        k_shape[1] > 0 && k_shape[1] <= std::numeric_limits<int>::max() &&
                        v_shape[1] > 0 && v_shape[1] <= std::numeric_limits<int>::max(),
                    "head counts must be in [1, INT_MAX]");

  const int q_num_heads = static_cast<int>(q_shape[1]);
  const int k_num_heads = static_cast<int>(k_shape[1]);
  const int v_num_heads = static_cast<int>(v_shape[1]);
  const int d_k = static_cast<int>(q_shape[2]);
  const int d_v = static_cast<int>(v_shape[2]);
  ORT_RETURN_IF_NOT(v_num_heads % k_num_heads == 0,
                    "Hv must be divisible by Hk");
  if (q_num_heads >= v_num_heads) {
    ORT_RETURN_IF_NOT(q_num_heads % v_num_heads == 0,
                      "standard mapping requires Hq divisible by Hv");
  } else {
    ORT_RETURN_IF_NOT(v_num_heads % q_num_heads == 0,
                      "inverse mapping requires Hv divisible by Hq");
  }

  const auto& offsets_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(offsets_shape.NumDimensions() == 1 && offsets_shape[0] >= 2,
                    "cumulative_sequence_length must have shape [B+1] with B >= 1");
  const int64_t batch_size_64 = offsets_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max(),
                    "B is too large for the CUDA kernel");
  const int batch_size = static_cast<int>(batch_size_64);

  const TensorShape expected_state_shape({batch_size_64, v_shape[1], v_shape[2], q_shape[2]});
  ORT_RETURN_IF_NOT(initial_state->Shape() == expected_state_shape,
                    "initial_state must have shape [B,Hv,V,K] = ", expected_state_shape);

  const bool needs_decay = update_rule_ == "gated" || update_rule_ == "gated_delta";
  const bool needs_beta = update_rule_ == "delta" || update_rule_ == "gated_delta";
  const bool needs_retrieval = needs_beta;
  ORT_RETURN_IF_NOT(needs_decay == (decay != nullptr),
                    needs_decay ? "decay is required for the selected update_rule"
                                : "decay must be omitted for the selected update_rule");
  ORT_RETURN_IF_NOT(needs_beta == (beta != nullptr),
                    needs_beta ? "beta is required for the selected update_rule"
                               : "beta must be omitted for the selected update_rule");

  bool decay_per_key_dim = false;
  bool decay_params_per_key_dim = false;
  if (decay) {
    ORT_RETURN_IF_ERROR(CheckTokenGateShape(*decay, total_tokens_64, v_num_heads, d_k,
                                            true, decay_per_key_dim, "decay"));
  }

  const bool softplus_decay = decay_activation_ == "softplus_decay";
  ORT_RETURN_IF_NOT(!softplus_decay || needs_decay,
                    "softplus_decay requires a gated update_rule");
  ORT_RETURN_IF_NOT(softplus_decay == (a_log != nullptr) && softplus_decay == (dt_bias != nullptr),
                    softplus_decay ? "A_log and dt_bias are required for softplus_decay"
                                   : "A_log and dt_bias must be omitted when decay_activation=none");
  if (softplus_decay) {
    bool a_log_per_key_dim = false;
    bool dt_bias_per_key_dim = false;
    ORT_RETURN_IF_ERROR(CheckDecayParamShape(*a_log, v_num_heads, d_k, a_log_per_key_dim, "A_log"));
    ORT_RETURN_IF_ERROR(CheckDecayParamShape(*dt_bias, v_num_heads, d_k, dt_bias_per_key_dim, "dt_bias"));
    ORT_RETURN_IF_NOT(a_log_per_key_dim == dt_bias_per_key_dim,
                      "A_log and dt_bias must use the same [Hv] or [Hv,K] shape");
    decay_params_per_key_dim = a_log_per_key_dim;
  }

  bool beta_per_head = false;
  if (beta) {
    const auto& beta_shape = beta->Shape();
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 2 && beta_shape[0] == total_tokens_64,
                      "beta must have shape [N,Hv] or [N,1]");
    ORT_RETURN_IF_NOT(beta_shape[1] == v_num_heads || beta_shape[1] == 1,
                      "beta must have shape [N,Hv] or [N,1]");
    beta_per_head = beta_shape[1] == v_num_heads;
  }

  const Tensor* gate_type_source = decay ? decay : beta;
  if (decay && beta) {
    ORT_RETURN_IF_NOT(decay->DataType() == beta->DataType(),
                      "decay and beta must have the same G type");
  }
  if constexpr (std::is_same_v<T, float>) {
    ORT_RETURN_IF_NOT(!gate_type_source || gate_type_source->IsDataType<float>(),
                      "T=float requires G=float");
  } else {
    ORT_RETURN_IF_NOT(!gate_type_source || gate_type_source->IsDataType<T>() ||
                          gate_type_source->IsDataType<float>(),
                      "T=float16/bfloat16 requires G=T or G=float");
  }

  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  const int out_heads = std::max(q_num_heads, v_num_heads);
  Tensor* output = context->Output(0, TensorShape({total_tokens_64, out_heads, v_shape[2]}));
  Tensor* final_state = context->Output(1, expected_state_shape);
  Tensor* checkpoints = context->Output(
      2, TensorShape({max_checkpoints_, batch_size_64, v_shape[1], v_shape[2], q_shape[2]}));

  const auto decay_activation = softplus_decay ? VarlenDecayActivation::kSoftplusDecay
                                                : VarlenDecayActivation::kNone;
  VarlenBetaActivation beta_activation = VarlenBetaActivation::kNone;
  if (beta_activation_ == "sigmoid") {
    beta_activation = VarlenBetaActivation::kSigmoid;
  } else if (beta_activation_ == "twice_sigmoid") {
    beta_activation = VarlenBetaActivation::kTwiceSigmoid;
  }

  using CudaT = typename OrtToCudaType<T>::type;
  const CudaT* q_data = reinterpret_cast<const CudaT*>(query->Data<T>());
  const CudaT* k_data = reinterpret_cast<const CudaT*>(key->Data<T>());
  const CudaT* v_data = reinterpret_cast<const CudaT*>(value->Data<T>());
  CudaT* out_data = reinterpret_cast<CudaT*>(output->MutableData<T>());
  const int total_tokens = static_cast<int>(total_tokens_64);

#define LAUNCH_WITH_GATE_TYPE(G_HOST, G_CUDA)                                                        \
  return LaunchVarlenLinearAttentionKernel<CudaT, G_CUDA>(                                          \
      Stream(context), q_data, k_data, v_data,                                                       \
      decay ? reinterpret_cast<const G_CUDA*>(decay->Data<G_HOST>()) : nullptr,                      \
      beta ? reinterpret_cast<const G_CUDA*>(beta->Data<G_HOST>()) : nullptr,                        \
      a_log ? a_log->Data<float>() : nullptr, dt_bias ? dt_bias->Data<float>() : nullptr,             \
      out_data, initial_state->Data<float>(), final_state->MutableData<float>(),                      \
      checkpoints ? checkpoints->MutableData<float>() : nullptr, cu_seqlens->Data<int32_t>(),        \
      total_tokens, batch_size, q_num_heads, k_num_heads, v_num_heads, d_k, d_v, scale,              \
      needs_decay, decay_per_key_dim, decay_activation, decay_params_per_key_dim, needs_beta,         \
      beta_per_head, beta_activation, needs_retrieval, max_checkpoints_,                              \
      GetDeviceProp().maxThreadsPerBlock, GetDeviceProp().sharedMemPerBlockOptin)

  if (!gate_type_source || gate_type_source->IsDataType<T>()) {
    LAUNCH_WITH_GATE_TYPE(T, CudaT);
  }
  LAUNCH_WITH_GATE_TYPE(float, float);

#undef LAUNCH_WITH_GATE_TYPE
}

template class VarlenLinearAttention<float>;
template class VarlenLinearAttention<MLFloat16>;
template class VarlenLinearAttention<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
