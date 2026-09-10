// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/sparse_paged_attention.h"

#include <cmath>
#include <type_traits>

#include "contrib_ops/cpu/bert/paged_attention_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_SPARSE_KERNEL_TYPED(T, TCACHE)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      SparsePagedAttention, kMSDomain, 1, T##_##TCACHE,                       \
      kCudaExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<TCACHE>())   \
          .TypeConstraint("T_AUX", DataTypeImpl::GetTensorType<T>())          \
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>())        \
          .InputMemoryType(OrtMemTypeCPUInput, 21),                           \
      SparsePagedAttention<T, TCACHE>);

REGISTER_SPARSE_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_SPARSE_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_SPARSE_KERNEL_TYPED(MLFloat16, int8_t)
REGISTER_SPARSE_KERNEL_TYPED(BFloat16, int8_t)

#undef REGISTER_SPARSE_KERNEL_TYPED

template <typename TCACHE>
constexpr KVCacheDataType SparseCacheStorageDataType() {
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    return KVCacheDataType::INT8;
  } else if constexpr (std::is_same<TCACHE, BFloat16>::value) {
    return KVCacheDataType::BFLOAT16;
  } else {
    return KVCacheDataType::FLOAT16;
  }
}

template <typename T, typename TCACHE>
SparsePagedAttention<T, TCACHE>::SparsePagedAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
              kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  local_window_size_ =
      static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  is_causal_ = info.GetAttrOrDefault<int64_t>("is_causal", 1) == 1;
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ =
      info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  rotary_offset_ =
      static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_offset", 0));
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  qk_norm_epsilon_ =
      info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  ORT_ENFORCE(std::isfinite(qk_norm_epsilon_) && qk_norm_epsilon_ > 0.0f);
  k_quant_type_ = StringToKVQuantizationType(
      info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
  v_quant_type_ = StringToKVQuantizationType(
      info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));

  const std::string attention_mode =
      info.GetAttrOrDefault<std::string>("attention_mode", "selected_only");
  ORT_ENFORCE(attention_mode == "selected_only" ||
                  attention_mode == "local_plus_selected",
              "'attention_mode' must be 'selected_only' or "
              "'local_plus_selected'.");
  attention_mode_ = attention_mode == "selected_only"
                        ? SparseAttentionMode::kSelectedOnly
                        : SparseAttentionMode::kLocalPlusSelected;

  const std::string selected_kv_source =
      info.GetAttrOrDefault<std::string>("selected_kv_source", "main");
  ORT_ENFORCE(selected_kv_source == "main" ||
                  selected_kv_source == "auxiliary",
              "'selected_kv_source' must be 'main' or 'auxiliary'.");
  selected_kv_source_ = selected_kv_source == "main"
                            ? SelectedKvSource::kMain
                            : SelectedKvSource::kAuxiliary;
  ORT_ENFORCE(
      info.GetAttrOrDefault<std::string>("auxiliary_cache_layout", "contiguous") ==
          "contiguous",
      "Only auxiliary_cache_layout='contiguous' is supported.");
  auxiliary_kv_shared_ =
      info.GetAttrOrDefault<int64_t>("auxiliary_kv_shared", 0) == 1;
}

template <typename T, typename TCACHE>
Status SparsePagedAttention<T, TCACHE>::ComputeInternal(
    OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_cache = context->Input<Tensor>(3);
  const Tensor* value_cache = context->Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context->Input<Tensor>(5);
  const Tensor* past_seqlens = context->Input<Tensor>(6);
  const Tensor* block_table = context->Input<Tensor>(7);
  const Tensor* slot_mapping = context->Input<Tensor>(8);
  const Tensor* selected_indices = context->Input<Tensor>(9);
  const Tensor* selected_counts = context->Input<Tensor>(10);
  const Tensor* auxiliary_key = context->Input<Tensor>(11);
  const Tensor* auxiliary_value = context->Input<Tensor>(12);
  const Tensor* auxiliary_lengths = context->Input<Tensor>(13);
  const Tensor* cos_cache = context->Input<Tensor>(14);
  const Tensor* sin_cache = context->Input<Tensor>(15);
  const Tensor* head_sink = context->Input<Tensor>(16);
  const Tensor* q_norm_weight = context->Input<Tensor>(17);
  const Tensor* k_norm_weight = context->Input<Tensor>(18);
  const Tensor* k_scale = context->Input<Tensor>(19);
  const Tensor* v_scale = context->Input<Tensor>(20);
  const Tensor* attention_metadata = context->Input<Tensor>(21);

  auto& device_prop = GetDeviceProp();
  PagedAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(paged_attention_helper::CheckInputs(
      query, key, value, key_cache, value_cache, cumulative_seqlens_q,
      past_seqlens, block_table, cos_cache, sin_cache, slot_mapping, head_sink,
      q_norm_weight, k_norm_weight, k_scale, v_scale,
      /*attention_metadata*/ static_cast<const Tensor*>(nullptr), &parameters,
      num_heads_, kv_num_heads_, scale_, softcap_, qk_norm_epsilon_,
      k_quant_type_, v_quant_type_, KVCacheDataType::DEFAULT,
      KVCacheDataType::DEFAULT, SparseCacheStorageDataType<TCACHE>(),
      /*is_latent_kv*/ false, /*v_head_size*/ 0, rotary_offset_,
      scale_ != 0.0f, device_prop.maxThreadsPerBlock));
  parameters.local_window_size = local_window_size_;
  parameters.is_causal = is_causal_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache are required when do_rotary=1.");
  }

  const auto& selected_dims = selected_indices->Shape().GetDims();
  if (selected_dims.size() != 2 ||
      selected_dims[0] != parameters.token_count) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "selected_indices must have shape (token_count, max_selected_entries).");
  }
  const int max_selected_entries = static_cast<int>(selected_dims[1]);
  const auto& count_dims = selected_counts->Shape().GetDims();
  if (count_dims.size() != 1 ||
      count_dims[0] != parameters.token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "selected_counts must have shape (token_count).");
  }

  int auxiliary_capacity = 0;
  int auxiliary_num_heads = 0;
  if (selected_kv_source_ == SelectedKvSource::kAuxiliary) {
    if (auxiliary_key == nullptr || auxiliary_lengths == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_key and auxiliary_lengths are required when "
          "selected_kv_source='auxiliary'.");
    }
    if (!auxiliary_kv_shared_ && auxiliary_value == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_value is required when auxiliary_kv_shared=0.");
    }
    if (auxiliary_kv_shared_ && auxiliary_value != nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_value must be absent when auxiliary_kv_shared=1.");
    }
    const auto& aux_dims = auxiliary_key->Shape().GetDims();
    if (aux_dims.size() != 4 || aux_dims[0] != parameters.batch_size ||
        (aux_dims[2] != 1 && aux_dims[2] != kv_num_heads_) ||
        aux_dims[3] != parameters.head_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_key must have shape (batch_size, capacity, 1 or "
          "kv_num_heads, head_size).");
    }
    auxiliary_capacity = static_cast<int>(aux_dims[1]);
    auxiliary_num_heads = static_cast<int>(aux_dims[2]);
    if (auxiliary_value != nullptr &&
        auxiliary_value->Shape() != auxiliary_key->Shape()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_value must have the same shape as auxiliary_key.");
    }
    const auto& aux_length_dims = auxiliary_lengths->Shape().GetDims();
    if (aux_length_dims.size() != 1 ||
        aux_length_dims[0] != parameters.batch_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "auxiliary_lengths must have shape (batch_size).");
    }
  } else if (auxiliary_key != nullptr || auxiliary_value != nullptr ||
             auxiliary_lengths != nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Auxiliary inputs must be absent when selected_kv_source='main'.");
  }

  if (attention_metadata != nullptr) {
    // This direct-read functional kernel does not allocate attention-length-dependent
    // temporary storage. Keep the complete bounds contract for future staged kernels.
    const auto& dims = attention_metadata->Shape().GetDims();
    if (dims.size() != 1 || dims[0] != 5) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "attention_metadata must have shape (5): [max_query_len_bound, "
          "max_local_main_len_bound, max_selected_entries_bound, "
          "max_auxiliary_len_bound, max_combined_attention_len_bound].");
    }
    const int* metadata = attention_metadata->Data<int32_t>();
    for (int i = 0; i < 5; ++i) {
      if (metadata[i] < 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "attention_metadata entries must be non-negative.");
      }
    }
  }

  TensorShapeVector output_shape{
      parameters.token_count, parameters.hidden_size};
  Tensor* output = context->Output(0, output_shape);
  Tensor* key_cache_out = context->Output(1, key_cache->Shape());
  Tensor* value_cache_out = context->Output(2, value_cache->Shape());
  if ((key_cache_out != nullptr &&
       key_cache_out->MutableData<TCACHE>() != key_cache->Data<TCACHE>()) ||
      (value_cache_out != nullptr &&
       value_cache_out->MutableData<TCACHE>() != value_cache->Data<TCACHE>())) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "key/value cache inputs and outputs must alias the same buffers.");
  }
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  size_t workspace_bytes = 0;
  if (do_rotary_ || parameters.use_qk_norm) {
    workspace_bytes = sizeof(T) * static_cast<size_t>(parameters.token_count) *
                      (parameters.hidden_size + parameters.kv_hidden_size);
  } else if (parameters.is_packed_qkv) {
    workspace_bytes = sizeof(T) * static_cast<size_t>(parameters.token_count) *
                      parameters.hidden_size;
  }
  auto workspace =
      GetScratchBuffer<void>(workspace_bytes, GetComputeStream(context));

  using CudaT = typename ToCudaType<T>::MappedType;
  using CudaTCache = typename ToCudaType<TCACHE>::MappedType;
  PagedAttentionData<CudaT, CudaTCache> data;
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key =
      key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr
                   ? nullptr
                   : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.key_cache = reinterpret_cast<CudaTCache*>(
      const_cast<TCACHE*>(key_cache->Data<TCACHE>()));
  data.value_cache = reinterpret_cast<CudaTCache*>(
      const_cast<TCACHE*>(value_cache->Data<TCACHE>()));
  data.k_scale = k_scale == nullptr ? nullptr : k_scale->Data<float>();
  data.v_scale = v_scale == nullptr ? nullptr : v_scale->Data<float>();
  data.cumulative_seqlens_q = cumulative_seqlens_q->Data<int32_t>();
  data.past_seqlens = past_seqlens->Data<int32_t>();
  data.block_table = block_table->Data<int32_t>();
  data.slot_mapping =
      slot_mapping == nullptr ? nullptr : slot_mapping->Data<int32_t>();
  data.cos_cache = cos_cache == nullptr
                       ? nullptr
                       : reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
  data.sin_cache = sin_cache == nullptr
                       ? nullptr
                       : reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  data.head_sink = head_sink == nullptr
                       ? nullptr
                       : reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  data.q_norm_weight =
      q_norm_weight == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
  data.k_norm_weight =
      k_norm_weight == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
  data.workspace_buffer = reinterpret_cast<CudaT*>(workspace.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  auto ort_stream = GetOrtStream(context);
  return SparseQkvToContext<CudaT, CudaTCache>(
      device_prop, ort_stream.get(), parameters, data,
      selected_indices->Data<int32_t>(), selected_counts->Data<int32_t>(),
      max_selected_entries,
      auxiliary_key == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(auxiliary_key->Data<T>()),
      auxiliary_value == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(auxiliary_value->Data<T>()),
      auxiliary_lengths == nullptr ? nullptr
                                   : auxiliary_lengths->Data<int32_t>(),
      auxiliary_capacity, auxiliary_num_heads, attention_mode_,
      selected_kv_source_, auxiliary_kv_shared_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
