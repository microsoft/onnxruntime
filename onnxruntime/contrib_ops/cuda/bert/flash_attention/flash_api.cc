/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
#if USE_FLASH_ATTENTION

#include "core/providers/cuda/cuda_common.h"
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace flash {

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t batch_size,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t num_heads,
                      const size_t num_heads_k,
                      const size_t head_size,
                      const size_t head_size_rounded,
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
                      bool is_causal) {
  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.p_ptr = p_d;
  // All stride are in elements, not bytes.
  params.q_row_stride = num_heads * head_size;
  params.k_row_stride = num_heads_k * head_size;
  params.v_row_stride = num_heads * head_size;
  params.q_head_stride = head_size;
  params.k_head_stride = head_size;
  params.v_head_stride = head_size;
  params.o_row_stride = num_heads * head_size;
  params.o_head_stride = head_size;
  params.is_bf16 = false;

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
    params.k_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.v_batch_stride = seqlen_k * num_heads * head_size;    // stride(0)
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
  params.h_h_k_ratio = head_size / num_heads_k;
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
}

size_t get_softmax_lse_size(int seqlen, int batch_size, int num_heads) {
  size_t bytes = sizeof(float) * batch_size * num_heads * seqlen;
  return bytes;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.d, [&] {
      run_mha_fwd_<elem_type, kHeadDim>(params, stream);
    });
  });
}

Status mha_fwd(const cudaDeviceProp& dprops,
               cudaStream_t stream,
               void* q,             // batch_size x seqlen_q x num_heads x head_size
               void* k,             // batch_size x seqlen_k x num_heads_k x head_size
               void* v,             // batch_size x seqlen_k x num_heads_k x head_size
               void* out,           // batch_size x seqlen_q x num_heads x head_size
               float* softmax_lse,  // batch_size x num_heads x seqlen_q
               const int batch_size,
               const int num_heads,
               const int num_heads_k,
               const int head_size,
               const int total_q,
               const int seqlen_q,
               const int seqlen_k,
               const float softmax_scale,
               const bool is_causal) {
  ORT_UNUSED_PARAMETER(total_q);

  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  ORT_ENFORCE(is_sm8x || is_sm90);

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE(num_heads % num_heads_k == 0);  // Number of heads in key/value must divide number of heads in query
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 256));

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params{};
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
                   is_causal);
  run_mha_fwd(params, stream);
  return Status::OK();
}

Status mha_varlen_fwd(const cudaDeviceProp& dprops,
                      cudaStream_t stream,
                      void* q,                   // half (total_q, num_heads, head_size)
                      void* k,                   // half (total_k, num_heads, head_size)
                      void* v,                   // half (total_k, num_heads, head_size)
                      void* out,                 // half (total_q, num_heads, head_size)
                      int* cu_seqlens_q,         // int (batch_size + 1)
                      int* cu_seqlens_k,         // int (batch_size + 1)
                      void* softmax_lse_buffer,  // float (batch_size, num_heads, max_seqlen_q)
                      const int batch_size,
                      const int num_heads,
                      const int num_heads_k,
                      const int head_size,
                      const int total_q,
                      const int max_seqlen_q_,
                      const int max_seqlen_k_,
                      const float softmax_scale,
                      const bool is_causal) {
  ORT_UNUSED_PARAMETER(total_q);

  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  ORT_ENFORCE(is_sm8x || is_sm90);

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 128));

  int max_seqlen_k = max_seqlen_k_;
  int max_seqlen_q = max_seqlen_q_;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q_, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k_, 128);

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
                   nullptr,  // o_tmp_buffer : nullptr,
                   softmax_lse_buffer,
                   softmax_scale,
                   is_causal);
  run_mha_fwd(params, stream);
  return Status::OK();
}

}  // namespace flash
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_FLASH_ATTENTION
