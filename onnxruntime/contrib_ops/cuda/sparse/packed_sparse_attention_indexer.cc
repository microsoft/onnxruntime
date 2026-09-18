// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;
namespace psai = onnxruntime::contrib::packed_sparse_attention_indexer;

#define REGISTER_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      PackedSparseAttentionIndexer,                                    \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kCudaExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()) \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()) \
          .MayInplace(11, 2)                                           \
          .MayInplace(14, 5),                                          \
      PackedSparseAttentionIndexer<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

namespace {

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "PackedSparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "PackedSparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

Status CheckIntDimension(const char* name, int64_t value, bool allow_zero = true) {
  ORT_RETURN_IF(value < (allow_zero ? 0 : 1) || value > std::numeric_limits<int>::max(),
                "PackedSparseAttentionIndexer: ", name, " must be in ", allow_zero ? "[0, INT_MAX]" : "(0, INT_MAX]",
                ", got ", value);
  return Status::OK();
}

// cos_cache / sin_cache may be shared across the batch ([max_position, rotary_width]) or
// request-specific ([batch_size, max_position, rotary_width]).
struct RotaryCacheShape {
  bool batched;
  int64_t max_rotary_length;
  int64_t rotary_width;
};

Status CheckRotaryCache(const Tensor* cos_cache, const Tensor* sin_cache, int64_t batch_size,
                        RotaryCacheShape& out) {
  ORT_RETURN_IF(cos_cache == nullptr, "PackedSparseAttentionIndexer: cos_cache is required");
  const auto& cos_shape = cos_cache->Shape();
  out.batched = cos_shape.NumDimensions() == 3;
  ORT_RETURN_IF_NOT(
      (out.batched && cos_shape[0] == batch_size && cos_shape[1] > 0) ||
          (cos_shape.NumDimensions() == 2 && cos_shape[0] > 0),
      "PackedSparseAttentionIndexer: cos_cache must have shape (max_position, rotary_width) or "
      "(batch_size, max_position, rotary_width), got ",
      cos_shape.ToString());
  out.max_rotary_length = out.batched ? cos_shape[1] : cos_shape[0];
  out.rotary_width = out.batched ? cos_shape[2] : cos_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("max_rotary_sequence_length", out.max_rotary_length, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("rotary_width", out.rotary_width, false));
  ORT_RETURN_IF_NOT(sin_cache != nullptr && sin_cache->Shape() == cos_shape,
                    "PackedSparseAttentionIndexer: sin_cache must have the same shape as cos_cache");
  return Status::OK();
}

}  // namespace

template <typename T>
PackedSparseAttentionIndexer<T>::PackedSparseAttentionIndexer(const OpKernelInfo& info) : CudaKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "PackedSparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(psai::TryParsePolicy(policy_mode, policy_), "PackedSparseAttentionIndexer: policy_mode must be '",
              psai::kPolicyModeQsa, "' or '", psai::kPolicyModeCsa, "', got '", policy_mode, "'");

  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "PackedSparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0 &&
                  compress_ratio_ <= (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2,
              "PackedSparseAttentionIndexer: compress_ratio must be positive and produce a generic buffer capacity "
              "no greater than INT_MAX, got ",
              compress_ratio_);

  ORT_ENFORCE(info.GetAttr<int64_t>("state_capacity", &state_capacity_).IsOK(),
              "PackedSparseAttentionIndexer: state_capacity is required");
  ORT_ENFORCE(state_capacity_ > 0 && state_capacity_ <= std::numeric_limits<int>::max(),
              "PackedSparseAttentionIndexer: state_capacity must be in (0, INT_MAX], got ", state_capacity_);

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  float head_weight_scale = 0.0f;
  has_head_weight_scale_ = info.GetAttr<float>("head_weight_scale", &head_weight_scale).IsOK();

  if (policy_ == psai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget,
                "PackedSparseAttentionIndexer: token_budget is required when policy_mode is 'qsa'");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale_,
                "PackedSparseAttentionIndexer: index_topk and head_weight_scale must not be set when policy_mode "
                "is 'qsa'");
    ORT_ENFORCE(token_budget_ > 0 && token_budget_ % compress_ratio_ == 0 &&
                    token_budget_ <= std::numeric_limits<int>::max() - compress_ratio_ + 1,
                "PackedSparseAttentionIndexer: token_budget must be > 0, divisible by compress_ratio, and produce "
                "a selected capacity no greater than INT_MAX, got token_budget=",
                token_budget_, " compress_ratio=", compress_ratio_);
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(has_index_topk, "PackedSparseAttentionIndexer: index_topk is required when policy_mode is 'csa'");
    ORT_ENFORCE(!has_token_budget,
                "PackedSparseAttentionIndexer: token_budget must not be set when policy_mode is 'csa'");
    ORT_ENFORCE(index_topk_ > 0 && index_topk_ <= std::numeric_limits<int>::max(),
                "PackedSparseAttentionIndexer: index_topk must be in (0, INT_MAX], got ", index_topk_);
    token_budget_ = 0;
  }

  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "PackedSparseAttentionIndexer: epsilon must be >= 0, got ", epsilon_);
  has_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  head_weight_scale_ = head_weight_scale;
}

template <typename T>
Status PackedSparseAttentionIndexer<T>::ComputeInternal(OpKernelContext* context) const {
  const bool is_qsa = policy_ == psai::Policy::kQsa;
  constexpr int kCsaOnlyInputs[] = {psai::kGate, psai::kPositionBias, psai::kHeadWeights};
  for (int index : kCsaOnlyInputs) {
    const bool provided = index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
    ORT_RETURN_IF(provided != !is_qsa, "PackedSparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode 'qsa'" : " is required for policy_mode 'csa'");
  }
  const bool position_ids_provided =
      psai::kPositionIds < context->InputCount() && context->Input<Tensor>(psai::kPositionIds) != nullptr;
  ORT_RETURN_IF(!is_qsa && !position_ids_provided,
                "PackedSparseAttentionIndexer: input ", psai::kPositionIds,
                " (position_ids) is required for policy_mode 'csa'");
  const bool gate_buffer_provided =
      psai::kPastGateBuffer < context->InputCount() && context->Input<Tensor>(psai::kPastGateBuffer) != nullptr;
  ORT_RETURN_IF(gate_buffer_provided != !is_qsa, "PackedSparseAttentionIndexer: input ", psai::kPastGateBuffer,
                gate_buffer_provided ? " must be omitted for policy_mode 'qsa'"
                                     : " is required for policy_mode 'csa'");

  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

template <typename T>
Status PackedSparseAttentionIndexer<T>::ComputeQsa(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* query = context->Input<Tensor>(psai::kQuery);
  const Tensor* key = context->Input<Tensor>(psai::kKey);
  const Tensor* query_norm_weight = context->Input<Tensor>(psai::kQueryNormWeight);
  const Tensor* key_norm_weight = context->Input<Tensor>(psai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(psai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(psai::kSinCache);
  const Tensor* cumulative_sequence_lengths = context->Input<Tensor>(psai::kCumulativeSequenceLengths);
  const Tensor* past_sequence_lengths = context->Input<Tensor>(psai::kPastSequenceLengths);
  const Tensor* position_ids = context->Input<Tensor>(psai::kPositionIds);
  const Tensor* past_key_state = context->Input<Tensor>(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context->Input<Tensor>(psai::kPastKvBuffer);
  const Tensor* past_state_lengths = context->Input<Tensor>(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2,
                    "PackedSparseAttentionIndexer: query must have shape (total_tokens, num_heads * head_size), "
                    "got ",
                    query_shape.ToString());
  const int64_t total_tokens = query_shape[0];
  ORT_RETURN_IF(query_norm_weight == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm_weight->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1,
                    "PackedSparseAttentionIndexer: query_norm_weight must have shape (head_size), got ",
                    query_norm_shape.ToString());
  const int64_t head_size = query_norm_shape[0];
  ORT_RETURN_IF(head_size <= 0 || query_shape[1] % head_size != 0,
                "PackedSparseAttentionIndexer: query width must be divisible by head_size");
  const int64_t num_heads = query_shape[1] / head_size;
  ORT_RETURN_IF_ERROR(CheckIntDimension("total_tokens", total_tokens));
  ORT_RETURN_IF_ERROR(CheckIntDimension("num_heads", num_heads, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("head_size", head_size, false));

  ORT_RETURN_IF(cumulative_sequence_lengths == nullptr,
                "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cumulative_sequence_lengths->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: cumulative_sequence_lengths must have shape (batch_size + 1), "
                    "got ",
                    cu_shape.ToString());
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF_ERROR(CheckIntDimension("batch_size", batch_size));
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_sequence_lengths, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(query_norm_weight, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));
  if (position_ids != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));
  }

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(
      rotary.rotary_width > 0 && rotary.rotary_width % 2 == 0 && rotary.rotary_width <= head_size,
      "PackedSparseAttentionIndexer: policy_mode 'qsa' requires an even rotary_width in (0, head_size], got ",
      rotary.rotary_width);

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: past_key_state must have shape "
                    "(batch_size, state_capacity, head_size), got ",
                    key_state_shape.ToString());
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("state_capacity", state_capacity, false));
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kQsa, token_budget_, index_topk_, compress_ratio_);

  Tensor* present_gate_buffer = context->Output(psai::kPresentGateBuffer, TensorShape({0}));
  ORT_RETURN_IF(present_gate_buffer != nullptr,
                "PackedSparseAttentionIndexer: output ", psai::kPresentGateBuffer,
                " must be omitted for policy_mode 'qsa'");
  Tensor* selected_indices = context->Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context->Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context->Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context->Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, head_size}));
  Tensor* present_state_lengths =
      context->Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  ORT_RETURN_IF(selected_indices == nullptr || selected_counts == nullptr || present_key_state == nullptr ||
                    present_kv_buffer == nullptr || present_state_lengths == nullptr,
                "PackedSparseAttentionIndexer: policy_mode 'qsa' requires selected_indices, selected_counts, "
                "present_key_state, present_kv_buffer and present_state_lengths outputs");

  PackedSparseAttentionIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.total_tokens = static_cast<int>(total_tokens);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.rotary_width = static_cast<int>(rotary.rotary_width);
  params.max_rotary_length = static_cast<int>(rotary.max_rotary_length);
  params.cos_cache_batched = rotary.batched;
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.state_capacity = static_cast<int>(state_capacity);
  params.buffer_capacity = static_cast<int>(buffer_capacity);
  params.capacity = static_cast<int>(capacity);
  params.has_position_ids = position_ids != nullptr;
  params.epsilon = epsilon_;
  params.scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.block_topk = static_cast<int>(token_budget_ / compress_ratio_);

  auto float_workspace = GetScratchBuffer<float>(GetQsaPackedWorkspaceFloatCount(params), GetComputeStream(context));
  auto overflow_flags =
      GetScratchBuffer<int32_t>(static_cast<size_t>(std::max<int64_t>(batch_size, 1)), GetComputeStream(context));

  return LaunchQsaPackedSparseAttentionIndexer<CudaT>(
      Stream(context), params,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      cumulative_sequence_lengths->Data<int32_t>(),
      past_sequence_lengths->Data<int32_t>(),
      position_ids != nullptr ? position_ids->Data<int64_t>() : nullptr,
      reinterpret_cast<const CudaT*>(past_key_state->Data<T>()),
      reinterpret_cast<const CudaT*>(past_kv_buffer->Data<T>()),
      past_state_lengths->Data<int32_t>(),
      selected_indices->MutableData<int32_t>(),
      selected_counts->MutableData<int32_t>(),
      reinterpret_cast<CudaT*>(present_key_state->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_kv_buffer->MutableData<T>()),
      present_state_lengths->MutableData<int32_t>(),
      float_workspace.get(),
      overflow_flags.get());
}

template <typename T>
Status PackedSparseAttentionIndexer<T>::ComputeCsa(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* query = context->Input<Tensor>(psai::kQuery);
  const Tensor* key = context->Input<Tensor>(psai::kKey);
  const Tensor* query_norm_weight = context->Input<Tensor>(psai::kQueryNormWeight);
  const Tensor* key_norm_weight = context->Input<Tensor>(psai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(psai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(psai::kSinCache);
  const Tensor* cumulative_sequence_lengths = context->Input<Tensor>(psai::kCumulativeSequenceLengths);
  const Tensor* past_sequence_lengths = context->Input<Tensor>(psai::kPastSequenceLengths);
  const Tensor* gate = context->Input<Tensor>(psai::kGate);
  const Tensor* position_bias = context->Input<Tensor>(psai::kPositionBias);
  const Tensor* head_weights = context->Input<Tensor>(psai::kHeadWeights);
  const Tensor* position_ids = context->Input<Tensor>(psai::kPositionIds);
  const Tensor* past_key_state = context->Input<Tensor>(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context->Input<Tensor>(psai::kPastKvBuffer);
  const Tensor* past_gate_buffer = context->Input<Tensor>(psai::kPastGateBuffer);
  const Tensor* past_state_lengths = context->Input<Tensor>(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2,
                    "PackedSparseAttentionIndexer: query must have shape (total_tokens, num_heads * head_size), "
                    "got ",
                    query_shape.ToString());
  const int64_t total_tokens = query_shape[0];
  ORT_RETURN_IF(query_norm_weight == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm_weight->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1,
                    "PackedSparseAttentionIndexer: query_norm_weight must have shape (head_size), got ",
                    query_norm_shape.ToString());
  const int64_t head_size = query_norm_shape[0];
  ORT_RETURN_IF(head_size <= 0 || query_shape[1] % head_size != 0,
                "PackedSparseAttentionIndexer: query width must be divisible by head_size");
  const int64_t num_heads = query_shape[1] / head_size;
  ORT_RETURN_IF_ERROR(CheckIntDimension("total_tokens", total_tokens));
  ORT_RETURN_IF_ERROR(CheckIntDimension("num_heads", num_heads, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("head_size", head_size, false));
  ORT_RETURN_IF(head_size > std::numeric_limits<int>::max() / 2,
                "PackedSparseAttentionIndexer: 2 * head_size must be no greater than INT_MAX");
  const int64_t width = 2 * head_size;

  ORT_RETURN_IF(cumulative_sequence_lengths == nullptr,
                "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cumulative_sequence_lengths->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: cumulative_sequence_lengths must have shape (batch_size + 1), "
                    "got ",
                    cu_shape.ToString());
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF_ERROR(CheckIntDimension("batch_size", batch_size));
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_sequence_lengths, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(query_norm_weight, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(position_bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {total_tokens, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && 2 * rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: policy_mode 'csa' requires 0 < 2 * rotary_width <= head_size, "
                    "got rotary_width=",
                    rotary.rotary_width, " head_size=", head_size);

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: past_key_state must have shape "
                    "(batch_size, state_capacity, head_size), got ",
                    key_state_shape.ToString());
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("state_capacity", state_capacity, false));
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_gate_buffer, "past_gate_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kCsa, token_budget_, index_topk_, compress_ratio_);

  Tensor* selected_indices = context->Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context->Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context->Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context->Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_gate_buffer =
      context->Output(psai::kPresentGateBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_state_lengths =
      context->Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  ORT_RETURN_IF(selected_indices == nullptr || selected_counts == nullptr || present_key_state == nullptr ||
                    present_kv_buffer == nullptr || present_gate_buffer == nullptr ||
                    present_state_lengths == nullptr,
                "PackedSparseAttentionIndexer: policy_mode 'csa' requires selected_indices, selected_counts, "
                "present_key_state, present_kv_buffer, present_gate_buffer and present_state_lengths outputs");

  PackedSparseAttentionIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.total_tokens = static_cast<int>(total_tokens);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.rotary_width = static_cast<int>(rotary.rotary_width);
  params.max_rotary_length = static_cast<int>(rotary.max_rotary_length);
  params.cos_cache_batched = rotary.batched;
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.state_capacity = static_cast<int>(state_capacity);
  params.buffer_capacity = static_cast<int>(buffer_capacity);
  params.capacity = static_cast<int>(capacity);
  params.has_position_ids = true;
  params.epsilon = epsilon_;
  params.scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.index_topk = static_cast<int>(index_topk_);
  params.head_weight_scale =
      has_head_weight_scale_ ? head_weight_scale_ : 1.0f / std::sqrt(static_cast<float>(num_heads));

  auto float_workspace = GetScratchBuffer<float>(GetCsaPackedWorkspaceFloatCount(params), GetComputeStream(context));
  auto overflow_flags =
      GetScratchBuffer<int32_t>(static_cast<size_t>(std::max<int64_t>(batch_size, 1)), GetComputeStream(context));

  return LaunchCsaPackedSparseAttentionIndexer<CudaT>(
      Stream(context), params,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(gate->Data<T>()),
      reinterpret_cast<const CudaT*>(position_bias->Data<T>()),
      reinterpret_cast<const CudaT*>(head_weights->Data<T>()),
      cumulative_sequence_lengths->Data<int32_t>(),
      past_sequence_lengths->Data<int32_t>(),
      position_ids->Data<int64_t>(),
      reinterpret_cast<const CudaT*>(past_key_state->Data<T>()),
      reinterpret_cast<const CudaT*>(past_kv_buffer->Data<T>()),
      reinterpret_cast<const CudaT*>(past_gate_buffer->Data<T>()),
      past_state_lengths->Data<int32_t>(),
      selected_indices->MutableData<int32_t>(),
      selected_counts->MutableData<int32_t>(),
      reinterpret_cast<CudaT*>(present_key_state->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_kv_buffer->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_gate_buffer->MutableData<T>()),
      present_state_lengths->MutableData<int32_t>(),
      float_workspace.get(),
      overflow_flags.get());
}

template class PackedSparseAttentionIndexer<float>;
template class PackedSparseAttentionIndexer<MLFloat16>;
template class PackedSparseAttentionIndexer<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
