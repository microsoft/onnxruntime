/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include <cutlass/numeric_types.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/flash_attention/flash.h"
#include "contrib_ops/cuda/bert/flash_attention/static_switch.h"

namespace onnxruntime {
namespace flash {

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      size_t batch_size,
                      size_t seqlen_q,
                      size_t seqlen_k,
                      size_t seqlen_q_rounded,
                      size_t seqlen_k_rounded,
                      size_t num_heads,
                      size_t num_heads_k,
                      size_t head_size,
                      size_t head_size_rounded,
                      // device pointers
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* p_d,
                      void* softmax_lse_d,
                      float softmax_scale,
                      bool is_causal,
                      bool kv_bsnh = true) {
  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;

  params.is_bf16 = false;

  // All stride are in elements, not bytes.
  if (kv_bsnh) {
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = num_heads_k * head_size;
    params.v_row_stride = num_heads_k * head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;
    params.o_row_stride = num_heads * head_size;
    params.o_head_stride = head_size;
  } else {
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = head_size;
    params.v_row_stride = head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = seqlen_k * head_size;
    params.v_head_stride = seqlen_k * head_size;
    params.o_row_stride = num_heads * head_size;
    params.o_head_stride = head_size;
  }

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
    params.k_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.v_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.o_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
  } else {
    params.q_batch_stride = 0;
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.o_batch_stride = 0;
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.is_causal = is_causal;
  params.is_seqlens_k_cumulative = true;
}

size_t get_softmax_lse_size(int seqlen, int batch_size, int num_heads) {
  size_t bytes = sizeof(float) * batch_size * num_heads * seqlen;
  return bytes;
}

size_t get_softmax_lse_accum_size(int num_splits, int batch_size, int num_heads, int seqlen_q) {
  size_t bytes = sizeof(float) * num_splits * batch_size * seqlen_q * num_heads;
  return bytes;
}

size_t get_out_accum_size(int num_splits, int batch_size, int num_heads, int seqlen_q, int head_size_rounded) {
  size_t bytes = sizeof(float) * num_splits * batch_size * seqlen_q * num_heads * head_size_rounded;
  return bytes;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.d, [&] {
      if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int batch_size, int seqlen_q, int seqlen_k, int num_heads, int head_size, int num_SMs,
                         int max_splits, bool new_kv, bool is_sm8x) {
  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = is_sm8x ? (head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64))
                              : (head_size <= 64 ? 256 : (head_size <= 160 ? 128 : 64));
  const int num_n_blocks = (seqlen_k + (!new_kv ? 0 : seqlen_q) + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
  // In any case we don't expect seqlen_q to be larger than 64 for inference.
  const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
  int batch_nheads_mblocks = batch_size * num_heads * num_m_blocks;
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
  // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
  // (i.e. it's 11 splits anyway).
  // So we check if the number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

Status mha_fwd(const cudaDeviceProp& dprops,
               cudaStream_t stream,
               void* q,            // batch_size x seqlen_q x num_heads x head_size
               void* k,            // batch_size x seqlen_k x num_heads_k x head_size
               void* v,            // batch_size x seqlen_k x num_heads_k x head_size
               void* out,          // batch_size x seqlen_q x num_heads x head_size
               void* softmax_lse,  // batch_size x num_heads x seqlen_q
               int batch_size,
               int num_heads,
               int num_heads_k,
               int head_size,
               int seqlen_q,
               int seqlen_k,
               float softmax_scale,
               bool is_causal,
               int num_splits,
               void* softmax_lse_accum,  // num_splits x batch_size x seqlen_q x num_heads
               void* out_accum,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
               bool kv_bsnh) {
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k,
                   head_size, head_size_rounded,
                   q, k, v, out,
                   /*cu_seqlens_q*/ nullptr,
                   /*cu_seqlens_k*/ nullptr,
                   nullptr,
                   softmax_lse,
                   softmax_scale,
                   is_causal,
                   kv_bsnh);
  params.dprops = &dprops;
  params.knew_ptr = nullptr;
  params.vnew_ptr = nullptr;
  params.knew_batch_stride = 0;
  params.vnew_batch_stride = 0;
  params.knew_row_stride = 0;
  params.vnew_row_stride = 0;
  params.knew_head_stride = 0;
  params.vnew_head_stride = 0;

  params.num_splits = num_splits;
  if (params.num_splits > 1 && softmax_lse_accum != nullptr && out_accum != nullptr) {
    params.softmax_lseaccum_ptr = softmax_lse_accum;
    params.oaccum_ptr = out_accum;
  } else {
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
  }

  run_mha_fwd(params, stream);
  return Status::OK();
}

Status mha_varlen_fwd(const cudaDeviceProp& dprops,
                      cudaStream_t stream,
                      void* q,            // half (total_q, num_heads, head_size)
                      void* k,            // half (total_k, num_heads, head_size)
                      void* v,            // half (total_k, num_heads, head_size)
                      void* out,          // half (total_q, num_heads, head_size)
                      int* cu_seqlens_q,  // int (batch_size + 1)
                      int* cu_seqlens_k,  // int (batch_size + 1)
                      void* softmax_lse,  // float (batch_size, num_heads, max_seqlen_q)
                      int batch_size,
                      int num_heads,
                      int num_heads_k,
                      int head_size,
                      int max_seqlen_q,
                      int max_seqlen_k,
                      float softmax_scale,
                      bool is_causal) {
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   max_seqlen_q, max_seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k,
                   head_size, head_size_rounded,
                   q, k, v, out,
                   cu_seqlens_q,
                   cu_seqlens_k,
                   nullptr,
                   softmax_lse,
                   softmax_scale,
                   is_causal);
  params.dprops = &dprops;
  params.num_splits = 0;
  params.softmax_lseaccum_ptr = nullptr;
  params.oaccum_ptr = nullptr;
  params.knew_ptr = nullptr;
  params.vnew_ptr = nullptr;
  run_mha_fwd(params, stream);
  return Status::OK();
}

bool is_supported(const cudaDeviceProp& dprops, int head_size, int num_heads, int num_heads_k) {
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  return (is_sm8x || is_sm90) && (head_size % 8 == 0) && (head_size <= 256) && (num_heads % num_heads_k == 0);
}

// This API is used when past key and value are present... since cached, these are assumed to have sequence length
// of max_sequence_length, so seqlen_k == max_sequence_length. The actual past sequence length is held in seqlens_k_.
Status mha_fwd_kvcache(const cudaDeviceProp& dprops,
                       cudaStream_t stream,
                       void* q,            // batch_size x seqlen_q x num_heads x head_size
                       void* kcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x head_size
                       void* vcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x head_size
                       void* k,            // (optional) batch_size x seqlen_k_new x num_heads_k x head_size
                       void* v,            // (optional) batch_size x seqlen_k_new x num_heads_k x head_size
                       void* out,          // batch_size x seqlen_q x num_heads x head_size
                       void* softmax_lse,  // batch_size x num_heads x seqlen_q
                       void* seqlens_k_,   // batch_size
                       int batch_size,
                       int num_heads,
                       int num_heads_k,
                       int head_size,
                       int seqlen_q,
                       int seqlen_k,
                       int seqlen_k_new,
                       const float softmax_scale,
                       bool is_causal,
                       bool past_bsnh,  // otherwise bnsh
                       int num_splits,
                       void* softmax_lse_accum,  // num_splits x batch_size x seqlen_q x num_heads
                       void* out_accum           // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
) {
  if (seqlen_q == 1) {
    is_causal = false;
  }  // causal=true is the same as causal=false in this case

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k,
                   head_size, head_size_rounded,
                   q, kcache, vcache, out,
                   /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr,
                   /*p_ptr=*/nullptr,
                   softmax_lse,
                   softmax_scale,
                   is_causal,
                   past_bsnh);
  params.dprops = &dprops;

  if (k != nullptr && v != nullptr) {
    params.seqlen_knew = seqlen_k_new;
    params.knew_ptr = k;
    params.vnew_ptr = v;
    // All stride are in elements, not bytes.
    params.knew_batch_stride = seqlen_k_new * num_heads_k * head_size;
    params.vnew_batch_stride = seqlen_k_new * num_heads_k * head_size;
    params.knew_row_stride = num_heads_k * head_size;
    params.vnew_row_stride = num_heads_k * head_size;
    params.knew_head_stride = head_size;
    params.vnew_head_stride = head_size;
  } else {
    params.seqlen_knew = 0;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;
  }

  params.is_seqlens_k_cumulative = seqlens_k_ == nullptr;
  if (seqlens_k_ != nullptr) {
    params.cu_seqlens_k = static_cast<int*>(seqlens_k_);
  }

  params.num_splits = num_splits;
  if (params.num_splits > 1 && softmax_lse_accum != nullptr && out_accum != nullptr) {
    params.softmax_lseaccum_ptr = softmax_lse_accum;
    params.oaccum_ptr = out_accum;
  } else {
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
  }

  // Only split kernel supports appending to KV cache
  run_mha_fwd(params, stream, /*force_split_kernel=*/k != nullptr);

  return Status::OK();
}

}  // namespace flash
}  // namespace onnxruntime

#endif  // USE_FLASH_ATTENTION
