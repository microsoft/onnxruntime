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
    }

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      int64_t index_topk = 0;
      int64_t index_num_heads = 0;
      int64_t index_head_dim = 0;
      ORT_ENFORCE(info.GetAttr("index_topk", &index_topk).IsOK() && index_topk > 0,
                  "DeepSeekV4Attention: index_topk must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_num_heads", &index_num_heads).IsOK() && index_num_heads > 0,
                  "DeepSeekV4Attention: index_num_heads must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_head_dim", &index_head_dim).IsOK() && index_head_dim > 0,
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

    ORT_RETURN_IF_NOT(hidden_states && position_ids && past_key && past_value &&
                          seqlens_k && cos_cache && sin_cache &&
                          q_a_weight && q_a_norm_weight && q_b_weight &&
                          kv_weight && kv_norm_weight && o_a_weight && o_b_weight && head_sink,
                      "DeepSeekV4Attention: required input is missing.");

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

    ORT_RETURN_IF_ERROR(LaunchDeepSeekV4CacheAndAttentionKernel<CudaT>(
        stream,
        context, pk_cu, pv_cu,
        q_full, new_key, new_val,
        bias_cu, bias_b, bias_h, bias_q, bias_k,
        sink, seqlens_k->Data<int32_t>(),
        narrow<int>(batch_size), narrow<int>(sequence_length),
        NH, HS, narrow<int>(cache_capacity),
        narrow<int>(local_window_size_),
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

    // ------------------------------------------------------------------
    // 18. Passthrough outputs (optional compressor state inputs → outputs).
    // ------------------------------------------------------------------
    static constexpr std::pair<int, int> kPassthrough[] = {
        {21, 3},  {22, 4},  {23, 5},  {34, 6},  {35, 7},  {36, 8},  {37, 9},
        {38, 10}, {39, 11}, {40, 12}, {41, 13}, {42, 14}, {43, 15}};

    for (const auto& [in_idx, out_idx] : kPassthrough) {
      if (out_idx < ctx->OutputCount() && HasInput(ctx, in_idx)) {
        const Tensor* in_t  = ctx->Input<Tensor>(in_idx);
        Tensor*       out_t = ctx->Output(out_idx, in_t->Shape());
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
            out_t->MutableDataRaw(), in_t->DataRaw(),
            static_cast<size_t>(in_t->Shape().Size()) * sizeof(T),
            cudaMemcpyDeviceToDevice, stream));
      }
    }

    // 19. Zero-fill optional debug output (output index 16).
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
  float   rms_norm_epsilon_{};
  float   scale_{};
  DeepSeekV4AttentionMode attention_mode_{};
};

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

}  // namespace

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
