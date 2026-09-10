// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/sparse_attention_indexer.h"

#include <cmath>
#include <string>

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;
namespace sai = onnxruntime::contrib::sparse_attention_indexer;

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      SparseAttentionIndexer,                                           \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("TB", DataTypeImpl::GetTensorType<bool>())    \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())  \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      SparseAttentionIndexer<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

namespace {

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "SparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "SparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

}  // namespace

template <typename T>
SparseAttentionIndexer<T>::SparseAttentionIndexer(const OpKernelInfo& info) : CudaKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "SparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(sai::TryParsePolicy(policy_mode, policy_), "SparseAttentionIndexer: policy_mode must be '",
              sai::kPolicyModeQsa, "' or '", sai::kPolicyModeCsa, "', got '", policy_mode, "'");

  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "SparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0, "SparseAttentionIndexer: compress_ratio must be > 0, got ", compress_ratio_);

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  float head_weight_scale = 0.0f;
  const bool has_head_weight_scale = info.GetAttr<float>("head_weight_scale", &head_weight_scale).IsOK();

  if (policy_ == sai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget, "SparseAttentionIndexer: token_budget is required when policy_mode is 'qsa'");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale,
                "SparseAttentionIndexer: index_topk and head_weight_scale must not be set when policy_mode is 'qsa'");
    ORT_ENFORCE(token_budget_ > 0 && token_budget_ % compress_ratio_ == 0,
                "SparseAttentionIndexer: token_budget must be > 0 and divisible by compress_ratio, got token_budget=",
                token_budget_, " compress_ratio=", compress_ratio_);
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(has_index_topk, "SparseAttentionIndexer: index_topk is required when policy_mode is 'csa'");
    ORT_ENFORCE(!has_token_budget, "SparseAttentionIndexer: token_budget must not be set when policy_mode is 'csa'");
    ORT_ENFORCE(index_topk_ > 0, "SparseAttentionIndexer: index_topk must be > 0, got ", index_topk_);
    token_budget_ = 0;
  }

  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "SparseAttentionIndexer: epsilon must be >= 0, got ", epsilon_);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  head_weight_scale_ = has_head_weight_scale ? head_weight_scale : 0.0f;
}

template <typename T>
Status SparseAttentionIndexer<T>::ComputeInternal(OpKernelContext* context) const {
  const bool is_qsa = policy_ == sai::Policy::kQsa;
  for (int index = sai::kMask; index < sai::kInputCount; ++index) {
    const bool policy_owns_slot = is_qsa ? (index <= sai::kPastKey) : (index >= sai::kGate);
    const bool provided = index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
    ORT_RETURN_IF(provided != policy_owns_slot, "SparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode '" : " is required for policy_mode '",
                  is_qsa ? sai::kPolicyModeQsa : sai::kPolicyModeCsa, "'");
  }

  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

template <typename T>
Status SparseAttentionIndexer<T>::ComputeQsa(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* query = context->Input<Tensor>(sai::kQuery);
  const Tensor* key = context->Input<Tensor>(sai::kKey);
  const Tensor* key_norm_weight = context->Input<Tensor>(sai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(sai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(sai::kSinCache);
  const Tensor* mask = context->Input<Tensor>(sai::kMask);
  const Tensor* past_key = context->Input<Tensor>(sai::kPastKey);

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 4,
                    "SparseAttentionIndexer: query must have shape (batch_size, sequence_length, num_heads, head_size)"
                    ", got ",
                    query_shape.ToString());
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const int64_t num_heads = query_shape[2];
  const int64_t head_size = query_shape[3];

  const auto& cos_shape = cos_cache->Shape();
  ORT_RETURN_IF_NOT(cos_shape.NumDimensions() == 3 && cos_shape[0] == batch_size && cos_shape[1] > 0,
                    "SparseAttentionIndexer: cos_cache must have shape "
                    "(batch_size, max_rotary_sequence_length, rotary_width), got ",
                    cos_shape.ToString());
  const int64_t max_rotary_length = cos_shape[1];
  const int64_t rotary_width = cos_shape[2];
  ORT_RETURN_IF_NOT(sin_cache->Shape() == cos_shape,
                    "SparseAttentionIndexer: sin_cache must have the same shape as "
                    "cos_cache");
  ORT_RETURN_IF_NOT(rotary_width > 0 && rotary_width % 2 == 0 && rotary_width <= head_size,
                    "SparseAttentionIndexer: policy_mode 'qsa' requires an even rotary_width in (0, head_size], got ",
                    rotary_width);

  const auto& past_shape = past_key->Shape();
  ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3 && past_shape[0] == batch_size && past_shape[2] == head_size,
                    "SparseAttentionIndexer: past_key must have shape (batch_size, past_sequence_length, head_size)"
                    ", got ",
                    past_shape.ToString());
  const int64_t past_sequence_length = past_shape[1];
  const int64_t total_sequence_length = past_sequence_length + sequence_length;

  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));

  const auto& mask_shape = mask->Shape();
  const bool mask_is_4d = mask_shape.NumDimensions() == 4;
  ORT_RETURN_IF_NOT(
      (mask_is_4d && mask_shape[0] == batch_size && mask_shape[1] == 1 && mask_shape[2] == sequence_length &&
       mask_shape[3] == total_sequence_length) ||
          (mask_shape.NumDimensions() == 3 && mask_shape[0] == batch_size && mask_shape[1] == sequence_length &&
           mask_shape[2] == total_sequence_length),
      "SparseAttentionIndexer: mask must have shape (batch_size, 1, sequence_length, total_sequence_length) or "
      "(batch_size, sequence_length, total_sequence_length) with total_sequence_length=",
      total_sequence_length, ", got ", mask_shape.ToString());

  SparseAttentionIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.sequence_length = static_cast<int>(sequence_length);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.rotary_width = static_cast<int>(rotary_width);
  params.max_rotary_length = static_cast<int>(max_rotary_length);
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.capacity = static_cast<int>(
      sai::SelectedCapacity(sai::Policy::kQsa, token_budget_, index_topk_, compress_ratio_));
  params.epsilon = epsilon_;
  params.scale = scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.past_sequence_length = static_cast<int>(past_sequence_length);
  params.total_sequence_length = static_cast<int>(total_sequence_length);
  params.max_block_count = static_cast<int>(total_sequence_length / compress_ratio_);
  params.block_topk = static_cast<int>(token_budget_ / compress_ratio_);

  Tensor* selected_indices = context->Output(sai::kSelectedIndices,
                                             TensorShape({batch_size, sequence_length, params.capacity}));
  Tensor* present_key = context->Output(sai::kPresentKey, TensorShape({batch_size, total_sequence_length, head_size}));
  ORT_RETURN_IF(selected_indices == nullptr || present_key == nullptr,
                "SparseAttentionIndexer: policy_mode 'qsa' requires both selected_indices and present_key outputs");

  auto float_workspace = GetScratchBuffer<float>(GetQsaWorkspaceFloatCount(params), context->GetComputeStream());
  auto int_workspace = GetScratchBuffer<int32_t>(GetQsaWorkspaceIntCount(params), context->GetComputeStream());

  return LaunchQsaSparseAttentionIndexer<CudaT>(
      Stream(context), params,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      mask->Data<bool>(),
      reinterpret_cast<const CudaT*>(past_key->Data<T>()),
      selected_indices->MutableData<int32_t>(),
      reinterpret_cast<CudaT*>(present_key->MutableData<T>()),
      float_workspace.get(),
      int_workspace.get());
}

template <typename T>
Status SparseAttentionIndexer<T>::ComputeCsa(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* query = context->Input<Tensor>(sai::kQuery);
  const Tensor* key = context->Input<Tensor>(sai::kKey);
  const Tensor* key_norm_weight = context->Input<Tensor>(sai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(sai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(sai::kSinCache);
  const Tensor* gate = context->Input<Tensor>(sai::kGate);
  const Tensor* position_bias = context->Input<Tensor>(sai::kPositionBias);
  const Tensor* head_weights = context->Input<Tensor>(sai::kHeadWeights);
  const Tensor* position_ids = context->Input<Tensor>(sai::kPositionIds);
  const Tensor* past_compressed_key = context->Input<Tensor>(sai::kPastCompressedKey);
  const Tensor* past_kv_buffer = context->Input<Tensor>(sai::kPastKvBuffer);
  const Tensor* past_gate_buffer = context->Input<Tensor>(sai::kPastGateBuffer);

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 4,
                    "SparseAttentionIndexer: query must have shape (batch_size, sequence_length, num_heads, head_size)"
                    ", got ",
                    query_shape.ToString());
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const int64_t num_heads = query_shape[2];
  const int64_t head_size = query_shape[3];
  const int64_t width = 2 * head_size;

  const auto& cos_shape = cos_cache->Shape();
  ORT_RETURN_IF_NOT(cos_shape.NumDimensions() == 3 && cos_shape[0] == batch_size && cos_shape[1] > 0,
                    "SparseAttentionIndexer: cos_cache must have shape "
                    "(batch_size, max_rotary_sequence_length, rotary_width), got ",
                    cos_shape.ToString());
  const int64_t max_rotary_length = cos_shape[1];
  const int64_t rotary_width = cos_shape[2];
  ORT_RETURN_IF_NOT(sin_cache->Shape() == cos_shape,
                    "SparseAttentionIndexer: sin_cache must have the same shape as "
                    "cos_cache");
  ORT_RETURN_IF_NOT(rotary_width > 0 && 2 * rotary_width <= head_size,
                    "SparseAttentionIndexer: policy_mode 'csa' requires 0 < 2 * rotary_width <= head_size, got "
                    "rotary_width=",
                    rotary_width, " head_size=", head_size);

  const auto& past_compressed_shape = past_compressed_key->Shape();
  ORT_RETURN_IF_NOT(past_compressed_shape.NumDimensions() == 3 && past_compressed_shape[0] == batch_size &&
                        past_compressed_shape[2] == head_size,
                    "SparseAttentionIndexer: past_compressed_key must have shape "
                    "(batch_size, past_compressed_length, head_size), got ",
                    past_compressed_shape.ToString());
  const int64_t past_compressed_length = past_compressed_shape[1];

  const auto& past_buffer_shape = past_kv_buffer->Shape();
  ORT_RETURN_IF_NOT(past_buffer_shape.NumDimensions() == 3 && past_buffer_shape[0] == batch_size &&
                        past_buffer_shape[2] == width,
                    "SparseAttentionIndexer: past_kv_buffer must have shape "
                    "(batch_size, buffer_length, 2 * head_size), got ",
                    past_buffer_shape.ToString());
  const int64_t past_buffer_length = past_buffer_shape[1];

  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(position_bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {batch_size, sequence_length, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {batch_size, sequence_length}));
  ORT_RETURN_IF_ERROR(CheckShape(past_gate_buffer, "past_gate_buffer",
                                 {batch_size, past_buffer_length, width}));

  sai::CsaWindowPlan plan;
  ORT_RETURN_IF_NOT(sai::TryComputeCsaWindowPlan(past_buffer_length, sequence_length, compress_ratio_, plan),
                    "SparseAttentionIndexer: past_kv_buffer sequence length must be in [0, 2 * compress_ratio), got ",
                    past_buffer_length);
  const int64_t present_compressed_length = past_compressed_length + plan.new_window_count;

  SparseAttentionIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.sequence_length = static_cast<int>(sequence_length);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.rotary_width = static_cast<int>(rotary_width);
  params.max_rotary_length = static_cast<int>(max_rotary_length);
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.capacity = static_cast<int>(
      sai::SelectedCapacity(sai::Policy::kCsa, token_budget_, index_topk_, compress_ratio_));
  params.epsilon = epsilon_;
  params.scale = scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.past_compressed_length = static_cast<int>(past_compressed_length);
  params.present_compressed_length = static_cast<int>(present_compressed_length);
  params.past_buffer_length = static_cast<int>(past_buffer_length);
  params.overlap_length = static_cast<int>(plan.overlap_length);
  params.new_window_count = static_cast<int>(plan.new_window_count);
  params.present_buffer_length = static_cast<int>(plan.present_buffer_length);
  params.present_buffer_start = static_cast<int>(plan.present_buffer_start);
  params.index_topk = static_cast<int>(index_topk_);
  params.head_weight_scale =
      head_weight_scale_ != 0.0f ? head_weight_scale_ : 1.0f / std::sqrt(static_cast<float>(num_heads));

  Tensor* selected_indices = context->Output(sai::kSelectedIndices,
                                             TensorShape({batch_size, sequence_length, params.capacity}));
  Tensor* present_compressed_key = context->Output(
      sai::kPresentCompressedKey, TensorShape({batch_size, present_compressed_length, head_size}));
  Tensor* present_kv_buffer =
      context->Output(sai::kPresentKvBuffer, TensorShape({batch_size, plan.present_buffer_length, width}));
  Tensor* present_gate_buffer =
      context->Output(sai::kPresentGateBuffer, TensorShape({batch_size, plan.present_buffer_length, width}));
  ORT_RETURN_IF(selected_indices == nullptr || present_compressed_key == nullptr || present_kv_buffer == nullptr ||
                    present_gate_buffer == nullptr,
                "SparseAttentionIndexer: policy_mode 'csa' requires selected_indices, present_compressed_key, "
                "present_kv_buffer and present_gate_buffer outputs");

  auto float_workspace = GetScratchBuffer<float>(GetCsaWorkspaceFloatCount(params), context->GetComputeStream());

  const CudaT* empty = nullptr;
  return LaunchCsaSparseAttentionIndexer<CudaT>(
      Stream(context), params,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(gate->Data<T>()),
      reinterpret_cast<const CudaT*>(position_bias->Data<T>()),
      reinterpret_cast<const CudaT*>(head_weights->Data<T>()),
      position_ids->Data<int64_t>(),
      past_compressed_length > 0 ? reinterpret_cast<const CudaT*>(past_compressed_key->Data<T>()) : empty,
      past_buffer_length > 0 ? reinterpret_cast<const CudaT*>(past_kv_buffer->Data<T>()) : empty,
      past_buffer_length > 0 ? reinterpret_cast<const CudaT*>(past_gate_buffer->Data<T>()) : empty,
      selected_indices->MutableData<int32_t>(),
      present_compressed_length > 0 ? reinterpret_cast<CudaT*>(present_compressed_key->MutableData<T>()) : nullptr,
      plan.present_buffer_length > 0 ? reinterpret_cast<CudaT*>(present_kv_buffer->MutableData<T>()) : nullptr,
      plan.present_buffer_length > 0 ? reinterpret_cast<CudaT*>(present_gate_buffer->MutableData<T>()) : nullptr,
      float_workspace.get());
}

template class SparseAttentionIndexer<float>;
template class SparseAttentionIndexer<MLFloat16>;
template class SparseAttentionIndexer<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
