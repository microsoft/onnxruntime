// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "contrib_ops/cuda/bert/deepseek_v4_attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

enum class DeepSeekV4AttentionMode {
  kSliding,
  kCsa,
  kHca,
};

constexpr int kFirstHcaInputIndex = 17;
constexpr int kLastHcaInputIndex = 23;
constexpr int kFirstCsaInputIndex = 24;
constexpr int kLastCsaInputIndex = 43;

inline bool HasInput(const OpKernelContext* context, int index) {
  return index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
}

inline bool HasAnyInputInRange(const OpKernelContext* context, int begin, int end) {
  for (int i = begin; i <= end; ++i) {
    if (HasInput(context, i)) {
      return true;
    }
  }
  return false;
}

DeepSeekV4AttentionMode ParseAttentionMode(const std::string& mode) {
  if (mode == "sliding") {
    return DeepSeekV4AttentionMode::kSliding;
  }
  if (mode == "csa") {
    return DeepSeekV4AttentionMode::kCsa;
  }
  if (mode == "hca") {
    return DeepSeekV4AttentionMode::kHca;
  }
  ORT_THROW("DeepSeekV4Attention: unsupported attention_mode='", mode, "'.");
}

template <typename T>
class DeepSeekV4Attention final : public CudaKernel {
 public:
  explicit DeepSeekV4Attention(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads_).IsOK() && num_heads_ > 0,
                "DeepSeekV4Attention: num_heads must be > 0.");

    ORT_ENFORCE(info.GetAttr("head_size", &head_size_).IsOK() && head_size_ > 0,
                "DeepSeekV4Attention: head_size must be > 0.");

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads == 1,
                "DeepSeekV4Attention: kv_num_heads must be 1.");

    ORT_ENFORCE(info.GetAttr("q_lora_rank", &q_lora_rank_).IsOK() && q_lora_rank_ > 0,
                "DeepSeekV4Attention: q_lora_rank must be > 0.");

    ORT_ENFORCE(info.GetAttr("o_groups", &o_groups_).IsOK() && o_groups_ > 0,
                "DeepSeekV4Attention: o_groups must be > 0.");

    ORT_ENFORCE(info.GetAttr("o_lora_rank", &o_lora_rank_).IsOK() && o_lora_rank_ > 0,
                "DeepSeekV4Attention: o_lora_rank must be > 0.");

    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ <= head_size_,
                "DeepSeekV4Attention: rotary_dim must be in (0, head_size].");

    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1,
                "DeepSeekV4Attention: rotary_interleaved must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_trailing", 0) == 1,
                "DeepSeekV4Attention: rotary_trailing must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("do_output_derotate", 0) == 1,
                "DeepSeekV4Attention: do_output_derotate must be 1.");

    ORT_ENFORCE(info.GetAttr("local_window_size", &local_window_size_).IsOK() && local_window_size_ > 0,
                "DeepSeekV4Attention: local_window_size must be > 0.");

    ORT_ENFORCE(info.GetAttr("rms_norm_epsilon", &rms_norm_epsilon_).IsOK() && rms_norm_epsilon_ > 0.0f,
                "DeepSeekV4Attention: rms_norm_epsilon must be > 0.");

    scale_ = info.GetAttrOrDefault<float>("scale", 1.0f / std::sqrt(static_cast<float>(head_size_)));

    const std::string attention_mode = info.GetAttrOrDefault<std::string>("attention_mode", "sliding");
    attention_mode_ = ParseAttentionMode(attention_mode);

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa ||
        attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      float compress_rate = 0.0f;
      ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate).IsOK() && compress_rate > 0.0f,
                  "DeepSeekV4Attention: compress_rate must be provided and > 0 for csa/hca mode.");
      compress_rate_ = static_cast<int64_t>(compress_rate);
      ORT_ENFORCE(static_cast<float>(compress_rate_) == compress_rate,
                  "DeepSeekV4Attention: compress_rate must be an integer value.");
    }

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      ORT_ENFORCE(info.GetAttr("index_topk", &index_topk_).IsOK() && index_topk_ > 0,
                  "DeepSeekV4Attention: index_topk must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_num_heads", &index_num_heads_).IsOK() && index_num_heads_ > 0,
                  "DeepSeekV4Attention: index_num_heads must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_head_dim", &index_head_dim_).IsOK() && index_head_dim_ > 0,
                  "DeepSeekV4Attention: index_head_dim must be provided and > 0 for csa mode.");
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    // ------------------------------------------------------------------
    // 1. Validate mode vs inputs.
    // ------------------------------------------------------------------
    const bool has_hca_inputs = HasAnyInputInRange(ctx, kFirstHcaInputIndex, kLastHcaInputIndex);
    const bool has_csa_inputs = HasAnyInputInRange(ctx, kFirstCsaInputIndex, kLastCsaInputIndex);

    if (attention_mode_ == DeepSeekV4AttentionMode::kSliding) {
      ORT_RETURN_IF(has_hca_inputs || has_csa_inputs,
                    "DeepSeekV4Attention(sliding): compressor inputs are not allowed.");
    } else if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF(!has_hca_inputs, "DeepSeekV4Attention(hca): HCA inputs are required.");
      ORT_RETURN_IF(has_csa_inputs, "DeepSeekV4Attention(hca): CSA inputs are not allowed.");
    } else {
      ORT_RETURN_IF(!has_csa_inputs, "DeepSeekV4Attention(csa): CSA inputs are required.");
      ORT_RETURN_IF(has_hca_inputs, "DeepSeekV4Attention(csa): HCA inputs are not allowed.");
    }

    // ------------------------------------------------------------------
    // 2. Gather required inputs.
    // ------------------------------------------------------------------
    const Tensor* hidden_states    = ctx->Input<Tensor>(0);
    const Tensor* position_ids     = ctx->Input<Tensor>(1);
    const Tensor* attention_bias   = ctx->Input<Tensor>(2);  // optional
    const Tensor* past_key         = ctx->Input<Tensor>(3);
    const Tensor* past_value       = ctx->Input<Tensor>(4);
    const Tensor* seqlens_k        = ctx->Input<Tensor>(5);
    // input 6  total_sequence_length — shape-checked but not used in compute
    const Tensor* cos_cache        = ctx->Input<Tensor>(7);
    const Tensor* sin_cache        = ctx->Input<Tensor>(8);
    const Tensor* q_a_weight       = ctx->Input<Tensor>(9);
    const Tensor* q_a_norm_weight  = ctx->Input<Tensor>(10);
    const Tensor* q_b_weight       = ctx->Input<Tensor>(11);
    const Tensor* kv_weight        = ctx->Input<Tensor>(12);
    const Tensor* kv_norm_weight   = ctx->Input<Tensor>(13);
    const Tensor* o_a_weight       = ctx->Input<Tensor>(14);
    const Tensor* o_b_weight       = ctx->Input<Tensor>(15);
    const Tensor* head_sink        = ctx->Input<Tensor>(16);
    const Tensor* hca_kv_weight = ctx->Input<Tensor>(17);
    const Tensor* hca_gate_weight = ctx->Input<Tensor>(18);
    const Tensor* hca_position_bias = ctx->Input<Tensor>(19);
    const Tensor* hca_kv_norm_weight = ctx->Input<Tensor>(20);
    const Tensor* past_hca_pending_kv = ctx->Input<Tensor>(21);
    const Tensor* past_hca_pending_gate = ctx->Input<Tensor>(22);
    const Tensor* past_hca_entries = ctx->Input<Tensor>(23);
    const Tensor* csa_kv_weight = ctx->Input<Tensor>(24);
    const Tensor* csa_gate_weight = ctx->Input<Tensor>(25);
    const Tensor* csa_position_bias = ctx->Input<Tensor>(26);
    const Tensor* csa_kv_norm_weight = ctx->Input<Tensor>(27);
    const Tensor* index_kv_weight = ctx->Input<Tensor>(28);
    const Tensor* index_gate_weight = ctx->Input<Tensor>(29);
    const Tensor* index_position_bias = ctx->Input<Tensor>(30);
    const Tensor* index_kv_norm_weight = ctx->Input<Tensor>(31);
    const Tensor* index_q_b_weight = ctx->Input<Tensor>(32);
    const Tensor* index_weights_proj_weight = ctx->Input<Tensor>(33);
    const Tensor* past_csa_pending_kv = ctx->Input<Tensor>(34);
    const Tensor* past_csa_pending_gate = ctx->Input<Tensor>(35);
    const Tensor* past_csa_entries = ctx->Input<Tensor>(36);
    const Tensor* past_csa_overlap_kv = ctx->Input<Tensor>(37);
    const Tensor* past_csa_overlap_gate = ctx->Input<Tensor>(38);
    const Tensor* past_index_pending_kv = ctx->Input<Tensor>(39);
    const Tensor* past_index_pending_gate = ctx->Input<Tensor>(40);
    const Tensor* past_index_entries = ctx->Input<Tensor>(41);
    const Tensor* past_index_overlap_kv = ctx->Input<Tensor>(42);
    const Tensor* past_index_overlap_gate = ctx->Input<Tensor>(43);

    ORT_RETURN_IF_NOT(hidden_states && position_ids && past_key && past_value &&
                          seqlens_k && cos_cache && sin_cache &&
                          q_a_weight && q_a_norm_weight && q_b_weight &&
                          kv_weight && kv_norm_weight && o_a_weight && o_b_weight && head_sink,
                      "DeepSeekV4Attention: required input is missing.");
    if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF_NOT(hca_kv_weight && hca_gate_weight && hca_position_bias && hca_kv_norm_weight &&
                            past_hca_pending_kv && past_hca_pending_gate && past_hca_entries,
                        "DeepSeekV4Attention(hca): all HCA inputs are required.");
    }
    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      ORT_RETURN_IF_NOT(csa_kv_weight && csa_gate_weight && csa_position_bias && csa_kv_norm_weight &&
                            index_kv_weight && index_gate_weight && index_position_bias && index_kv_norm_weight &&
                            index_q_b_weight && index_weights_proj_weight && past_csa_pending_kv &&
                            past_csa_pending_gate && past_csa_entries && past_csa_overlap_kv &&
                            past_csa_overlap_gate && past_index_pending_kv && past_index_pending_gate &&
                            past_index_entries && past_index_overlap_kv && past_index_overlap_gate,
                        "DeepSeekV4Attention(csa): all CSA inputs are required.");
    }

    // ------------------------------------------------------------------
    // 3. Extract and validate dimensions.
    // ------------------------------------------------------------------
    const auto& hidden_shape = hidden_states->Shape();
    ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must be rank 3.");
    const int64_t batch_size      = hidden_shape[0];
    const int64_t sequence_length = hidden_shape[1];
    const int64_t hidden_size     = hidden_shape[2];
    ORT_RETURN_IF_NOT(hidden_size == num_heads_ * head_size_,
                      "hidden size must match num_heads * head_size.");

    const auto& past_key_shape = past_key->Shape();
    ORT_RETURN_IF_NOT(past_key_shape.NumDimensions() == 4 &&
                          past_key_shape[0] == batch_size &&
                          past_key_shape[1] == 1 &&
                          past_key_shape[3] == head_size_,
                      "past_key shape must be (B, 1, cache_capacity, head_size).");
    const int64_t cache_capacity = past_key_shape[2];

    const int64_t kv_width = kv_weight->Shape()[1];
    ORT_RETURN_IF_NOT(kv_width >= head_size_ * 2, "kv_weight width must be at least 2 * head_size.");

    const int64_t head_sink_count =
        (head_sink->Shape().NumDimensions() == 0) ? 1 : head_sink->Shape()[0];
    ORT_RETURN_IF_NOT(head_sink_count == 1 || head_sink_count == num_heads_,
                      "head_sink must have 1 or num_heads entries.");

    const int64_t cos_rows = cos_cache->Shape()[0];

    // ------------------------------------------------------------------
    // 4. Allocate outputs and copy KV cache.
    // ------------------------------------------------------------------
    Tensor* output        = ctx->Output(0, hidden_shape);
    Tensor* present_key   = ctx->Output(1, past_key_shape);
    Tensor* present_value = ctx->Output(2, past_key_shape);

    cudaStream_t stream = Stream(ctx);
    const size_t kv_cache_bytes = static_cast<size_t>(past_key_shape.Size()) * sizeof(T);

    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_key->MutableData<T>(), past_key->Data<T>(),
        kv_cache_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_value->MutableData<T>(), past_value->Data<T>(),
        kv_cache_bytes, cudaMemcpyDeviceToDevice, stream));

    // ------------------------------------------------------------------
    // 5. Type aliases and cuBLAS setup.
    // ------------------------------------------------------------------
    typedef typename ToCudaType<T>::MappedType CudaT;

    cublasHandle_t cublas         = GetCublasHandle(ctx);
    const cudaDeviceProp& dev_prop = GetDeviceProp();
    const int max_threads          = dev_prop.maxThreadsPerBlock;

    CudaT one  = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    const int BS  = narrow<int>(batch_size * sequence_length);
    const int H   = narrow<int>(hidden_size);
    const int NH  = narrow<int>(num_heads_);
    const int HS  = narrow<int>(head_size_);
    const int Rq  = narrow<int>(q_lora_rank_);
    const int Kw  = narrow<int>(kv_width);
    const int Go  = narrow<int>(o_groups_);
    const int Ro  = narrow<int>(o_lora_rank_);
    const int Rdim = narrow<int>(rotary_dim_);
    const int CosCacheRows = narrow<int>(cos_rows);

    const auto* hidden_cu = reinterpret_cast<const CudaT*>(hidden_states->Data<T>());
    const auto* pos_ids   = position_ids->Data<int64_t>();
    const auto* cos_cu    = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    const auto* sin_cu    = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());

    // ------------------------------------------------------------------
    // 6. Scratch buffers.
    // ------------------------------------------------------------------
    auto q_a_buf      = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * Rq, stream);
    auto kv_row_buf   = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * Kw, stream);
    auto q_full_buf   = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * H, stream);
    auto new_key_buf  = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * HS, stream);
    auto new_val_buf  = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * HS, stream);
    auto context_buf  = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * H, stream);
    auto o_a_buf      = GetScratchBuffer<CudaT>(static_cast<size_t>(BS) * Go * Ro, stream);

    CudaT* q_a     = q_a_buf.get();
    CudaT* kv_row  = kv_row_buf.get();
    CudaT* q_full  = q_full_buf.get();
    CudaT* new_key = new_key_buf.get();
    CudaT* new_val = new_val_buf.get();
    CudaT* context = context_buf.get();
    CudaT* o_a     = o_a_buf.get();

    const auto* qa_w  = reinterpret_cast<const CudaT*>(q_a_weight->Data<T>());
    const auto* qan_w = reinterpret_cast<const CudaT*>(q_a_norm_weight->Data<T>());
    const auto* qb_w  = reinterpret_cast<const CudaT*>(q_b_weight->Data<T>());
    const auto* kv_w  = reinterpret_cast<const CudaT*>(kv_weight->Data<T>());
    const auto* kvn_w = reinterpret_cast<const CudaT*>(kv_norm_weight->Data<T>());
    const auto* oa_w  = reinterpret_cast<const CudaT*>(o_a_weight->Data<T>());
    const auto* ob_w  = reinterpret_cast<const CudaT*>(o_b_weight->Data<T>());
    const auto* sink  = reinterpret_cast<const CudaT*>(head_sink->Data<T>());

    // ------------------------------------------------------------------
    // 7. Q projection: hidden × q_a_weight  →  q_a  [BS, Rq]
    //    Row-major formula: cublasGemmHelper(NN, n, m, k, alpha, B, n, A, k, ...)
    //    computes A [m, k] × B [k, n] → C [m, n].
    // ------------------------------------------------------------------
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, Rq, BS, H,
        &one, qa_w, Rq, hidden_cu, H, &zero, q_a, Rq, dev_prop, UseTF32()));

    // 8. RMS-norm q_a in-place.
    HostApplyLayerNorm<CudaT, float, CudaT, true>(
        dev_prop, stream,
        q_a, nullptr, nullptr, q_a,
        BS, Rq, rms_norm_epsilon_,
        qan_w, nullptr);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    // 9. Q projection: q_a × q_b_weight  →  q_full  [BS, H]
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, H, BS, Rq,
        &one, qb_w, H, q_a, Rq, &zero, q_full, H, dev_prop, UseTF32()));

    // ------------------------------------------------------------------
    // 10. KV projection: hidden × kv_weight  →  kv_row  [BS, Kw]
    // ------------------------------------------------------------------
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, Kw, BS, H,
        &one, kv_w, Kw, hidden_cu, H, &zero, kv_row, Kw, dev_prop, UseTF32()));

    // 11. RMS-norm kv_row in-place.
    HostApplyLayerNorm<CudaT, float, CudaT, true>(
        dev_prop, stream,
        kv_row, nullptr, nullptr, kv_row,
        BS, Kw, rms_norm_epsilon_,
        kvn_w, nullptr);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    // 12. Split kv_row → new_key [BS, HS] and new_val [BS, HS].
    ORT_RETURN_IF_ERROR(LaunchSplitKVRowKernel<CudaT>(
        stream, new_key, new_val, kv_row, BS, HS, Kw, max_threads));

    // ------------------------------------------------------------------
    // 13. Apply interleaved trailing RoPE to all query heads (in-place).
    //     q_full is [B, S, NH, HS] in BSNH layout.
    //     position_ids_format = 1  (each entry is the explicit token position).
    // ------------------------------------------------------------------
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<CudaT>(
        stream,
        q_full, q_full,
        pos_ids, /*past_sequence_lengths=*/nullptr,
        cos_cu, sin_cu,
        narrow<int>(batch_size), narrow<int>(sequence_length),
        NH, HS, Rdim, CosCacheRows,
        /*position_ids_format=*/1,
        /*interleaved=*/true, max_threads,
        /*is_input_bnsh_format=*/false,
        /*trailing=*/true, /*negate_sin=*/false));

    // 14. Apply RoPE to the key head (in-place).
    //     new_key is [B, S, 1, HS] treated as BSNH with num_heads=1.
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<CudaT>(
        stream,
        new_key, new_key,
        pos_ids, /*past_sequence_lengths=*/nullptr,
        cos_cu, sin_cu,
        narrow<int>(batch_size), narrow<int>(sequence_length),
        /*num_heads=*/1, HS, Rdim, CosCacheRows,
        /*position_ids_format=*/1,
        /*interleaved=*/true, max_threads,
        /*is_input_bnsh_format=*/false,
        /*trailing=*/true, /*negate_sin=*/false));

    CudaT* compressed_entries = nullptr;
    int compressed_entry_count = 0;
    auto compressed_entries_buffer = GetScratchBuffer<CudaT>(1, stream);

    if (attention_mode_ != DeepSeekV4AttentionMode::kSliding) {
      const bool is_csa = attention_mode_ == DeepSeekV4AttentionMode::kCsa;
      const Tensor* compressor_kv_weight = is_csa ? csa_kv_weight : hca_kv_weight;
      const Tensor* compressor_gate_weight = is_csa ? csa_gate_weight : hca_gate_weight;
      const Tensor* compressor_position_bias = is_csa ? csa_position_bias : hca_position_bias;
      const Tensor* compressor_norm_weight = is_csa ? csa_kv_norm_weight : hca_kv_norm_weight;
      const Tensor* past_pending_kv = is_csa ? past_csa_pending_kv : past_hca_pending_kv;
      const Tensor* past_pending_gate = is_csa ? past_csa_pending_gate : past_hca_pending_gate;
      const Tensor* past_entries = is_csa ? past_csa_entries : past_hca_entries;
      const int output_base = is_csa ? 6 : 3;
      const int compressor_width = is_csa ? 2 * HS : HS;

      ORT_RETURN_IF_NOT(compressor_kv_weight->Shape() == TensorShape({hidden_size, compressor_width}) &&
                            compressor_gate_weight->Shape() == compressor_kv_weight->Shape(),
                        "compressor projection weight shape mismatch.");
      ORT_RETURN_IF_NOT(compressor_position_bias->Shape() == TensorShape({compress_rate_, compressor_width}),
                        "compressor position_bias shape mismatch.");
      ORT_RETURN_IF_NOT(compressor_norm_weight->Shape() == TensorShape({head_size_}),
                        "compressor norm weight shape mismatch.");
      ORT_RETURN_IF_NOT(past_pending_kv->Shape().NumDimensions() == 3 &&
                            past_pending_kv->Shape()[0] == batch_size &&
                            past_pending_kv->Shape()[2] == compressor_width &&
                            past_pending_gate->Shape() == past_pending_kv->Shape(),
                        "compressor pending state shape mismatch.");
      ORT_RETURN_IF_NOT((past_entries->Shape().NumDimensions() == 3 ||
                         past_entries->Shape().NumDimensions() == 4) &&
                            past_entries->Shape()[0] == batch_size &&
                            past_entries->Shape()[past_entries->Shape().NumDimensions() - 1] == head_size_,
                        "compressor entries state shape mismatch.");

      const int pending_token_count = narrow<int>(past_pending_kv->Shape()[1]);
      const int total_compressor_tokens = pending_token_count + narrow<int>(sequence_length);
      const int new_entry_count = total_compressor_tokens / narrow<int>(compress_rate_);
      const int output_pending_count = total_compressor_tokens - new_entry_count * narrow<int>(compress_rate_);
      const bool entries_rank4 = past_entries->Shape().NumDimensions() == 4;
      ORT_RETURN_IF(entries_rank4 && past_entries->Shape()[1] != 1,
                    "rank-4 compressor entries must have shape (B, 1, E, head_size).");
      const int old_entry_count = narrow<int>(entries_rank4 ? past_entries->Shape()[2] : past_entries->Shape()[1]);
      compressed_entry_count = old_entry_count + new_entry_count;
      ORT_RETURN_IF(new_entry_count > 0 &&
                        static_cast<int64_t>(compressed_entry_count - 1) * compress_rate_ >= cos_rows,
                    "compressor position out of range for cos/sin cache.");

      auto compressor_kv_buffer = GetScratchBuffer<CudaT>(
          static_cast<size_t>(BS) * compressor_width, stream);
      auto compressor_gate_buffer = GetScratchBuffer<CudaT>(
          static_cast<size_t>(BS) * compressor_width, stream);
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, compressor_width, BS, H,
          &one, reinterpret_cast<const CudaT*>(compressor_kv_weight->Data<T>()), compressor_width,
          hidden_cu, H, &zero, compressor_kv_buffer.get(), compressor_width, dev_prop, UseTF32()));
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, compressor_width, BS, H,
          &one, reinterpret_cast<const CudaT*>(compressor_gate_weight->Data<T>()), compressor_width,
          hidden_cu, H, &zero, compressor_gate_buffer.get(), compressor_width, dev_prop, UseTF32()));

        Tensor* pending_kv_output = output_base < ctx->OutputCount()
                    ? ctx->Output(output_base, TensorShape({batch_size, output_pending_count,
                                   compressor_width}))
                    : nullptr;
        Tensor* pending_gate_output = output_base + 1 < ctx->OutputCount()
                      ? ctx->Output(output_base + 1,
                        TensorShape({batch_size, output_pending_count,
                             compressor_width}))
                      : nullptr;
          const size_t pending_element_count = std::max<size_t>(
            1, static_cast<size_t>(batch_size) * output_pending_count * compressor_width);
          auto pending_kv_buffer = GetScratchBuffer<CudaT>(pending_element_count, stream);
          auto pending_gate_buffer = GetScratchBuffer<CudaT>(pending_element_count, stream);
          CudaT* pending_kv_output_data = pending_kv_output == nullptr
                            ? pending_kv_buffer.get()
                            : reinterpret_cast<CudaT*>(pending_kv_output->MutableData<T>());
          CudaT* pending_gate_output_data = pending_gate_output == nullptr
                            ? pending_gate_buffer.get()
                            : reinterpret_cast<CudaT*>(pending_gate_output->MutableData<T>());
      const TensorShape entries_shape = entries_rank4
                                            ? TensorShape({batch_size, 1, compressed_entry_count, head_size_})
                                            : TensorShape({batch_size, compressed_entry_count, head_size_});
      Tensor* entries_output = output_base + 2 < ctx->OutputCount()
                   ? ctx->Output(output_base + 2, entries_shape)
                   : nullptr;
      compressed_entries_buffer = GetScratchBuffer<CudaT>(
          std::max<size_t>(1, static_cast<size_t>(batch_size) * compressed_entry_count * HS), stream);
      compressed_entries = entries_output != nullptr
                               ? reinterpret_cast<CudaT*>(entries_output->MutableData<T>())
                               : compressed_entries_buffer.get();

      const CudaT* old_entries = reinterpret_cast<const CudaT*>(past_entries->Data<T>());
      if (old_entry_count > 0) {
        for (int batch = 0; batch < narrow<int>(batch_size); ++batch) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
              compressed_entries + static_cast<size_t>(batch) * compressed_entry_count * HS,
              old_entries + static_cast<size_t>(batch) * old_entry_count * HS,
              static_cast<size_t>(old_entry_count) * HS * sizeof(CudaT),
              cudaMemcpyDeviceToDevice, stream));
        }
      }

      auto overlap_kv_buffer = GetScratchBuffer<CudaT>(
          std::max<size_t>(1, static_cast<size_t>(batch_size) * compress_rate_ * HS), stream);
      auto overlap_gate_buffer = GetScratchBuffer<CudaT>(
          std::max<size_t>(1, static_cast<size_t>(batch_size) * compress_rate_ * HS), stream);
      CudaT* overlap_kv_output_data = overlap_kv_buffer.get();
      CudaT* overlap_gate_output_data = overlap_gate_buffer.get();
      if (is_csa) {
        ORT_RETURN_IF_NOT(past_csa_overlap_kv->Shape() == TensorShape({batch_size, compress_rate_, head_size_}) &&
                              past_csa_overlap_gate->Shape() == past_csa_overlap_kv->Shape(),
                          "CSA overlap state shape mismatch.");
        Tensor* overlap_kv_output = 9 < ctx->OutputCount()
                ? ctx->Output(9, past_csa_overlap_kv->Shape())
                : nullptr;
        Tensor* overlap_gate_output = 10 < ctx->OutputCount()
                  ? ctx->Output(10, past_csa_overlap_gate->Shape())
                  : nullptr;
        if (overlap_kv_output != nullptr) {
          overlap_kv_output_data = reinterpret_cast<CudaT*>(overlap_kv_output->MutableData<T>());
        }
        if (overlap_gate_output != nullptr) {
          overlap_gate_output_data = reinterpret_cast<CudaT*>(overlap_gate_output->MutableData<T>());
        }
        if (new_entry_count == 0) {
          const size_t overlap_bytes = static_cast<size_t>(past_csa_overlap_kv->Shape().Size()) * sizeof(T);
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_kv_output_data, past_csa_overlap_kv->Data<T>(),
                                               overlap_bytes, cudaMemcpyDeviceToDevice, stream));
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_gate_output_data, past_csa_overlap_gate->Data<T>(),
                                               overlap_bytes, cudaMemcpyDeviceToDevice, stream));
        }
      }

      ORT_RETURN_IF_ERROR(LaunchDeepSeekV4CompressorKernel<CudaT>(
          stream, compressed_entries,
          pending_kv_output_data, pending_gate_output_data,
          overlap_kv_output_data, overlap_gate_output_data,
          compressor_kv_buffer.get(), compressor_gate_buffer.get(),
          reinterpret_cast<const CudaT*>(past_pending_kv->Data<T>()),
          reinterpret_cast<const CudaT*>(past_pending_gate->Data<T>()),
          is_csa ? reinterpret_cast<const CudaT*>(past_csa_overlap_kv->Data<T>()) : nullptr,
          is_csa ? reinterpret_cast<const CudaT*>(past_csa_overlap_gate->Data<T>()) : nullptr,
          reinterpret_cast<const CudaT*>(compressor_position_bias->Data<T>()),
          reinterpret_cast<const CudaT*>(compressor_norm_weight->Data<T>()),
          cos_cu, sin_cu, narrow<int>(batch_size), narrow<int>(sequence_length),
          pending_token_count, old_entry_count, new_entry_count,
          compressor_width, HS, narrow<int>(compress_rate_), Rdim,
          narrow<int>(cos_cache->Shape()[1]), rms_norm_epsilon_, is_csa, max_threads));

      if (is_csa) {
        static constexpr std::pair<int, int> kIndexPassthrough[] = {
            {39, 11}, {40, 12}, {41, 13}, {42, 14}, {43, 15}};
        for (const auto& [input_index, output_index] : kIndexPassthrough) {
          if (output_index < ctx->OutputCount()) {
            const Tensor* input = ctx->Input<Tensor>(input_index);
            Tensor* state_output = ctx->Output(output_index, input->Shape());
            if (state_output != nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
                  state_output->MutableDataRaw(), input->DataRaw(),
                  static_cast<size_t>(input->Shape().Size()) * sizeof(T),
                  cudaMemcpyDeviceToDevice, stream));
            }
          }
        }
      }
    }

    // ------------------------------------------------------------------
    // 15. KV-cache update + sliding-window attention → context [BS, H].
    // ------------------------------------------------------------------
    int64_t bias_b = 0, bias_h = 0, bias_q = 0, bias_k = 0;
    const CudaT* bias_cu = nullptr;
    if (attention_bias != nullptr) {
      const auto& bs = attention_bias->Shape();
      ORT_RETURN_IF_NOT(bs.NumDimensions() == 4, "attention_bias must be rank 4.");
      bias_b  = bs[0];
      bias_h  = bs[1];
      bias_q  = bs[2];
      bias_k  = bs[3];
      bias_cu = reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
    }

    auto* pk_cu = reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    auto* pv_cu = reinterpret_cast<CudaT*>(present_value->MutableData<T>());
    const int max_selected_entries = attention_mode_ == DeepSeekV4AttentionMode::kCsa
                                         ? std::min(narrow<int>(index_topk_), compressed_entry_count)
                                         : compressed_entry_count;
    const size_t attention_workspace_stride =
        static_cast<size_t>(2 * (narrow<int>(cache_capacity) + max_selected_entries) + max_threads);
    auto attention_workspace = GetScratchBuffer<float>(
        static_cast<size_t>(batch_size) * attention_workspace_stride, stream);

    ORT_RETURN_IF_ERROR(LaunchDeepSeekV4CacheAndAttentionKernel<CudaT>(
        stream,
        context, attention_workspace.get(), pk_cu, pv_cu,
        q_full, new_key, new_val,
        compressed_entries,
        bias_cu, bias_b, bias_h, bias_q, bias_k,
        sink, seqlens_k->Data<int32_t>(), pos_ids,
        narrow<int>(batch_size), narrow<int>(sequence_length),
        NH, HS, narrow<int>(cache_capacity),
        narrow<int>(local_window_size_),
        compressed_entry_count, narrow<int>(compress_rate_), narrow<int>(index_topk_),
        static_cast<int>(attention_mode_),
        scale_, narrow<int>(head_sink_count),
        max_threads));

    // ------------------------------------------------------------------
    // 16. Output projection: context × o_a_weight  →  o_a  [BS, Go*Ro]
    // ------------------------------------------------------------------
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, Go * Ro, BS, H,
        &one, oa_w, Go * Ro, context, H, &zero, o_a, Go * Ro, dev_prop, UseTF32()));

    // 17. Output projection: o_a × o_b_weight  →  output  [BS, H]
    auto* out_cu = reinterpret_cast<CudaT*>(output->MutableData<T>());
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, H, BS, Go * Ro,
        &one, ob_w, H, o_a, Go * Ro, &zero, out_cu, H, dev_prop, UseTF32()));

    // 18. Zero-fill optional debug output (output index 16).
    if (ctx->OutputCount() > 16) {
      Tensor* output_qk = ctx->Output(16, hidden_shape);
      if (output_qk != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
            output_qk->MutableDataRaw(), 0,
            static_cast<size_t>(hidden_shape.Size()) * sizeof(T),
            stream));
      }
    }

    return Status::OK();
  }

 private:
  int64_t num_heads_{};
  int64_t head_size_{};
  int64_t q_lora_rank_{};
  int64_t o_groups_{};
  int64_t o_lora_rank_{};
  int64_t rotary_dim_{};
  int64_t local_window_size_{};
  int64_t compress_rate_{1};
  int64_t index_topk_{};
  int64_t index_num_heads_{};
  int64_t index_head_dim_{};
  float   rms_norm_epsilon_{};
  float   scale_{};
  DeepSeekV4AttentionMode attention_mode_{};
};

}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      DeepSeekV4Attention,                                                  \
      kMSDomain,                                                            \
      1,                                                                    \
      T,                                                                    \
      kCudaExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())      \
          .TypeConstraint("P", DataTypeImpl::GetTensorType<int64_t>()),     \
      DeepSeekV4Attention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
