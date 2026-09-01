// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "contrib_ops/cuda/bert/gated_delta_net_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;
using onnxruntime::cuda::OrtToCudaType;
namespace gdn = gated_delta_net;

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      GatedDeltaNet,                                                    \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 10)                      \
          .MayInplace(6, 1),                                            \
      GatedDeltaNet<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GatedDeltaNet<T>::GatedDeltaNet(const OpKernelInfo& info) : CudaKernel(info) {
  const std::string rule = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  if (rule == "linear") {
    update_rule_ = gdn::UpdateRule::kLinear;
  } else if (rule == "gated") {
    update_rule_ = gdn::UpdateRule::kGated;
  } else if (rule == "delta") {
    update_rule_ = gdn::UpdateRule::kDelta;
  } else if (rule == "gated_delta") {
    update_rule_ = gdn::UpdateRule::kGatedDelta;
  } else {
    ORT_THROW("update_rule must be one of: linear, gated, delta, gated_delta");
  }

  const std::string gate = info.GetAttrOrDefault<std::string>("gate_activation", "none");
  if (gate == "none") {
    gate_activation_ = gdn::GateActivation::kNone;
  } else if (gate == "qwen") {
    gate_activation_ = gdn::GateActivation::kQwen;
  } else {
    ORT_THROW("gate_activation must be one of: none, qwen");
  }

  const std::string beta = info.GetAttrOrDefault<std::string>("beta_activation", "none");
  if (beta == "none") {
    beta_activation_ = gdn::BetaActivation::kNone;
  } else if (beta == "sigmoid") {
    beta_activation_ = gdn::BetaActivation::kSigmoid;
  } else {
    ORT_THROW("beta_activation must be one of: none, sigmoid");
  }

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  const int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  ORT_ENFORCE(chunk_size > 0 && chunk_size <= std::numeric_limits<int>::max(),
              "chunk_size must be positive and fit in int32, got ", chunk_size);
  chunk_size_ = static_cast<int>(chunk_size);
  const int64_t state_update_capacity =
      info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity >= 0 && state_update_capacity <= 8,
              "state_update_capacity must be in [0, 8], got ", state_update_capacity);
  state_update_capacity_ = static_cast<int>(state_update_capacity);
  qk_l2_norm_ = info.GetAttrOrDefault<int64_t>("qk_l2_norm", 0) != 0;

  forced_engine_ = gdn::EngineFromName(
      ParseEnvironmentVariableWithDefault<std::string>("ORT_GDN_PLAN", ""));
}

template <typename T>
Status GatedDeltaNet<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename OrtToCudaType<T>::type CudaT;

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* decay = context->Input<Tensor>(4);
  const Tensor* beta = context->Input<Tensor>(5);
  const Tensor* initial_state = context->Input<Tensor>(6);
  const Tensor* a_log = context->Input<Tensor>(7);
  const Tensor* dt_bias = context->Input<Tensor>(8);
  const Tensor* capture_count = context->Input<Tensor>(9);
  const Tensor* state_update_active = context->Input<Tensor>(10);

  ORT_RETURN_IF_NOT((state_update_capacity_ > 0) == (capture_count != nullptr),
                    "capture_count must be present exactly when state_update_capacity is positive");

  const bool needs_decay =
      update_rule_ == gdn::UpdateRule::kGated ||
      update_rule_ == gdn::UpdateRule::kGatedDelta;
  const bool needs_beta =
      update_rule_ == gdn::UpdateRule::kDelta ||
      update_rule_ == gdn::UpdateRule::kGatedDelta;
  ORT_RETURN_IF_NOT(needs_decay == (decay != nullptr),
                    "decay input presence must match update_rule");
  ORT_RETURN_IF_NOT(needs_beta == (beta != nullptr),
                    "beta input presence must match update_rule");

  ORT_RETURN_IF_NOT(query != nullptr && key != nullptr && value != nullptr,
                    "query, key and value are required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();
  const size_t qkv_rank = q_shape.NumDimensions();
  ORT_RETURN_IF_NOT(qkv_rank == 3 || qkv_rank == 4,
                    "query, key and value must be rank 3 [total_tokens, num_heads, head_size] or "
                    "rank 4 [batch, sequence, num_heads, head_size]");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == qkv_rank && v_shape.NumDimensions() == qkv_rank,
                    "query, key and value must have the same rank");

  // Leading token axes: one when packed, two when the batch and sequence axes are explicit.
  // Both spellings have the identical token-major memory layout, so only the shapes differ.
  const size_t token_dims = qkv_rank - 2;
  const int64_t total_tokens = q_shape.SizeToDimension(token_dims);
  ORT_RETURN_IF_NOT(k_shape.SizeToDimension(token_dims) == total_tokens &&
                        v_shape.SizeToDimension(token_dims) == total_tokens,
                    "query, key and value must agree on total_tokens");
  const int64_t max_int = std::numeric_limits<int>::max();
  ORT_RETURN_IF_NOT(total_tokens > 0 && total_tokens <= max_int,
                    "total_tokens must be positive and fit in int32");

  // Head counts are derived from the shapes; there are no head-count attributes.
  ORT_RETURN_IF_NOT(q_shape[token_dims] <= max_int && k_shape[token_dims] <= max_int &&
                        v_shape[token_dims] <= max_int && q_shape[token_dims + 1] <= max_int &&
                        v_shape[token_dims + 1] <= max_int,
                    "head counts and head sizes must fit in int32");
  const int num_heads_q = static_cast<int>(q_shape[token_dims]);
  const int num_heads_k = static_cast<int>(k_shape[token_dims]);
  const int num_heads_v = static_cast<int>(v_shape[token_dims]);
  const int head_size_qk = static_cast<int>(q_shape[token_dims + 1]);
  const int head_size_v = static_cast<int>(v_shape[token_dims + 1]);

  ORT_RETURN_IF_NOT(k_shape[token_dims + 1] == head_size_qk,
                    "key head_size must equal query head_size");
  ORT_RETURN_IF_NOT(num_heads_q == num_heads_k,
                    "num_heads_q (", num_heads_q, ") must equal num_heads_k (", num_heads_k, ")");
  ORT_RETURN_IF_NOT(num_heads_v > 0 && num_heads_q > 0 && num_heads_v % num_heads_q == 0,
                    "num_heads_v (", num_heads_v, ") must be a positive multiple of num_heads_q (",
                    num_heads_q, ")");

  if (initial_state != nullptr) {
    const size_t r = initial_state->Shape().NumDimensions();
    ORT_RETURN_IF_NOT(r == 4,
                      "initial_state must be rank 4 [batch, num_heads_v, head_size_v, "
                      "head_size_qk]");
  }

  int64_t batch_dim = 1;
  if (cu_seqlens != nullptr) {
    ORT_RETURN_IF_NOT(qkv_rank == 3,
                      "cu_seqlens describes ragged packing, so query/key/value must be rank 3");
    ORT_RETURN_IF_NOT(cu_seqlens->Shape().NumDimensions() == 1 && cu_seqlens->Shape()[0] >= 2,
                      "cu_seqlens must be rank 1 with at least 2 elements");
    batch_dim = cu_seqlens->Shape()[0] - 1;
  } else if (qkv_rank == 4) {
    batch_dim = q_shape[0];
  } else {
    // Uniform packing. The batch size comes from the state, which GenAI always binds.
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "cu_seqlens is absent, so initial_state is required to determine batch size");
    batch_dim = initial_state->Shape()[initial_state->Shape().NumDimensions() - 4];
  }
  ORT_RETURN_IF_NOT(batch_dim > 0 && batch_dim <= max_int,
                    "batch size must be positive and fit in int32");
  const int batch = static_cast<int>(batch_dim);
  if (cu_seqlens == nullptr && qkv_rank == 3) {
    ORT_RETURN_IF_NOT(batch > 0 && total_tokens % batch == 0,
                      "total_tokens (", total_tokens, ") must be divisible by batch (", batch, ")");
  }

  if (initial_state != nullptr) {
    const auto& s = initial_state->Shape();
    const size_t o = s.NumDimensions() - 4;
    ORT_RETURN_IF_NOT(s[o] == batch && s[o + 1] == num_heads_v && s[o + 2] == head_size_v &&
                          s[o + 3] == head_size_qk,
                      "initial_state must be [batch, num_heads_v, head_size_v, head_size_qk] "
                      "(V-major)");
  }

  // decay and beta carry the same leading token axes as query/key/value; a trailing
  // head_size_qk axis makes the decay per key dimension.
  const bool decay_per_key_dim =
      decay != nullptr && decay->Shape().NumDimensions() == token_dims + 2;
  ORT_RETURN_IF(state_update_capacity_ > 0 && decay_per_key_dim,
                "transition capture does not support per-key-dimension decay");
  if (decay != nullptr) {
    const auto& d = decay->Shape();
    ORT_RETURN_IF_NOT(d.NumDimensions() == token_dims + 1 || decay_per_key_dim,
                      "decay must be [...tokens, num_heads_v] or "
                      "[...tokens, num_heads_v, head_size_qk]");
    ORT_RETURN_IF_NOT(d.SizeToDimension(token_dims) == total_tokens && d[token_dims] == num_heads_v,
                      "decay must be [...tokens, num_heads_v] or "
                      "[...tokens, num_heads_v, head_size_qk]");
    if (decay_per_key_dim) {
      ORT_RETURN_IF_NOT(d[token_dims + 1] == head_size_qk,
                        "per-key-dim decay last axis must be head_size_qk");
    }
  }
  if (beta != nullptr) {
    const auto& b = beta->Shape();
    ORT_RETURN_IF_NOT(b.NumDimensions() == token_dims + 1 &&
                          b.SizeToDimension(token_dims) == total_tokens &&
                          b[token_dims] == num_heads_v,
                      "beta must be [...tokens, num_heads_v]");
  }
  if (gate_activation_ == gdn::GateActivation::kQwen) {
    ORT_RETURN_IF_NOT(a_log != nullptr && dt_bias != nullptr,
                      "gate_activation=qwen requires a_log and dt_bias");
    ORT_RETURN_IF_NOT(a_log->Shape().NumDimensions() == 1 && a_log->Shape()[0] == num_heads_v,
                      "a_log must be [num_heads_v]");
    ORT_RETURN_IF_NOT(dt_bias->Shape().NumDimensions() == 1 &&
                          dt_bias->Shape()[0] == num_heads_v,
                      "dt_bias must be [num_heads_v]");
  } else {
    ORT_RETURN_IF_NOT(a_log == nullptr && dt_bias == nullptr,
                      "a_log and dt_bias require gate_activation=qwen");
  }

  if (capture_count != nullptr) {
    ORT_RETURN_IF_NOT(capture_count->Shape().NumDimensions() == 1 &&
                          capture_count->Shape()[0] == batch,
                      "capture_count must be [batch]");
  }
  if (state_update_active != nullptr) {
    ORT_RETURN_IF_NOT(state_update_active->Shape().NumDimensions() == 1 &&
                          state_update_active->Shape()[0] == 1,
                      "state_update_active must be [1]");
  }

  const int out_heads = std::max(num_heads_q, num_heads_v);
  TensorShapeVector out_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  out_dims.push_back(out_heads);
  out_dims.push_back(head_size_v);
  Tensor* output = context->Output(0, TensorShape(out_dims));

  TensorShape state_shape{batch, num_heads_v, head_size_v, head_size_qk};
  Tensor* final_state = context->Output(1, state_shape);

  const int64_t state_update_width = static_cast<int64_t>(state_update_capacity_) *
                                     (num_heads_v + num_heads_k * head_size_qk +
                                      num_heads_v * head_size_v);
  Tensor* state_update = context->Output(2, TensorShape{batch, state_update_width});

  gdn::Descriptor desc{};
  desc.total_tokens = total_tokens;
  desc.batch = batch;
  desc.num_heads_q = num_heads_q;
  desc.num_heads_k = num_heads_k;
  desc.num_heads_v = num_heads_v;
  desc.head_size_qk = head_size_qk;
  desc.head_size_v = head_size_v;
  desc.chunk_size = chunk_size_;
  desc.state_update_capacity = state_update_capacity_;
  desc.state_update_active =
      state_update_capacity_ > 0 &&
      (state_update_active == nullptr || state_update_active->Data<int32_t>()[0] != 0);
  desc.update_rule = update_rule_;
  desc.gate_activation = gate_activation_;
  desc.beta_activation = beta_activation_;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    desc.io_type = gdn::IoType::kFloat16;
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    desc.io_type = gdn::IoType::kBFloat16;
  } else {
    desc.io_type = gdn::IoType::kFloat;
  }
  desc.qk_l2_norm = qk_l2_norm_;
  desc.decay_per_key_dim = decay_per_key_dim;
  desc.ragged = cu_seqlens != nullptr;
  desc.preferred_engine = forced_engine_;

  const cudaDeviceProp& prop = GetDeviceProp();
  desc.sm_major = prop.major;

  gdn::Plan plan = gdn::SelectPlan(desc, prop.sharedMemPerBlockOptin);
  if (forced_engine_ == gdn::Engine::kChunked ||
      forced_engine_ == gdn::Engine::kChunkedSplit) {
    ORT_RETURN_IF_NOT(
        plan.engine == forced_engine_,
        "GatedDeltaNet: the ", gdn::EngineName(forced_engine_),
        " engine cannot serve this descriptor (",
        plan.reject_reason ? plan.reject_reason : "no reason recorded", ")");
  } else if (forced_engine_ == gdn::Engine::kRecurrent) {
    plan.engine = gdn::Engine::kRecurrent;
    // Match the fallback path's kernel choice: the override must measure what the
    // heuristic would have run, not the generic kernel.
    plan.warp_specialized = gdn::RecurrentIsWarpSpecialized(head_size_qk);
    plan.state_update_tail_pass = false;
    plan.short_row_tail_pass = false;
  } else if (forced_engine_ == gdn::Engine::kCudnn) {
    plan.engine = gdn::Engine::kCudnn;
  }
  ORT_RETURN_IF_NOT(plan.supported, "GatedDeltaNet: no supported plan (",
                    plan.reject_reason ? plan.reject_reason : "unknown", ")");
  ORT_RETURN_IF_NOT(plan.engine != gdn::Engine::kCudnn,
                    "GatedDeltaNet: the cuDNN engine is reserved and not implemented");

  if (plan.engine == gdn::Engine::kRecurrent && !plan.warp_specialized) {
    const size_t smem = gdn::RecurrentSmemBytes(head_size_qk, head_size_v);
    ORT_RETURN_IF_NOT(smem <= prop.sharedMemPerBlockOptin,
                      "GatedDeltaNet: the recurrent engine needs ", smem,
                      " bytes of shared memory but the device allows ",
                      prop.sharedMemPerBlockOptin);
  }

  gdn::VariantPack<CudaT> pack{};
  pack.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  pack.key = reinterpret_cast<const CudaT*>(key->Data<T>());
  pack.value = reinterpret_cast<const CudaT*>(value->Data<T>());
  pack.cu_seqlens = cu_seqlens != nullptr ? cu_seqlens->Data<int32_t>() : nullptr;
  pack.capture_count = desc.state_update_active ? capture_count->Data<int32_t>() : nullptr;
  pack.decay = decay != nullptr ? decay->Data<float>() : nullptr;
  pack.beta = beta != nullptr ? beta->Data<float>() : nullptr;
  pack.initial_state = initial_state != nullptr ? initial_state->Data<float>() : nullptr;
  pack.a_log = a_log != nullptr ? a_log->Data<float>() : nullptr;
  pack.dt_bias = dt_bias != nullptr ? dt_bias->Data<float>() : nullptr;
  pack.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  pack.final_state = final_state != nullptr ? final_state->MutableData<float>() : nullptr;
  pack.state_update = state_update != nullptr ? state_update->MutableData<float>() : nullptr;

  const cudaStream_t stream = Stream(context);
  // Only the inactive case is cleared. While capture is active the kernels write just the
  // captured prefix, and the schema declares the remaining entries unspecified.
  if (state_update != nullptr && state_update_capacity_ > 0 && !desc.state_update_active) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(pack.state_update, 0, state_update->SizeInBytes(), stream));
  }
  const float scale = scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size_qk));

  IAllocatorUniquePtr<uint8_t> workspace;
  if (plan.workspace_bytes > 0) {
    workspace = GetScratchBuffer<uint8_t>(plan.workspace_bytes, GetComputeStream(context));
    pack.workspace = workspace.get();
  }

  return gdn::LaunchGatedDeltaNet<CudaT>(
      desc, plan, pack, scale, prop.maxThreadsPerBlock,
      static_cast<size_t>(prop.sharedMemPerBlockOptin), stream);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
