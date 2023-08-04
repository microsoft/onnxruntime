/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

//#include <torch/extension.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <c10/cuda/CUDAGuard.h>

//#include "core/common/common.h"
#include <cutlass/numeric_types.h>
#include "core/providers/cuda/cuda_common.h"

#include "flash.h"
#include "static_switch.h"

//#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
                      const size_t v_head_size,
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
  // Reset the parameters
  //memset(&params, 0, sizeof(params));

  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.p_ptr = p_d;
  // All stride are in elements, not bytes.
  params.q_row_stride = num_heads * head_size;
  params.k_row_stride = num_heads_k * head_size;
  params.v_row_stride = num_heads * v_head_size;
  params.q_head_stride = head_size;
  params.k_head_stride = head_size;
  params.v_head_stride = v_head_size;
  params.o_row_stride = num_heads * v_head_size;
  params.o_head_stride = v_head_size;
  params.is_bf16 = false; // TODO: how can i determine this value?

  if (cu_seqlens_q_d == nullptr) {
    // TODO: confirm batch stride
    params.q_batch_stride = batch_size * num_heads * head_size; // stride(0)
    params.k_batch_stride = batch_size * num_heads_k * head_size;  // stride(0)
    params.v_batch_stride = batch_size * num_heads * v_head_size;  // stride(0)
    params.o_batch_stride = batch_size * num_heads * v_head_size;  // stride(0)
  }
  else {
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

  // TODO: idk what blockmask is but everything gets zeroed by memset so...
  params.blockmask = 0;

  // Set this to probability of keeping an element to simplify things.
  //params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  //params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  //params.rp_dropout = 1.f / params.p_dropout;
  //params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  //TORCH_CHECK(p_dropout < 1.f);

  params.is_causal = is_causal;
}

int get_max_seqlen_k(int max_seqlen_k_, int head_size, bool& loop) {
  int blocksize_c = head_size > 64 ? 128 : 256;
  // Need to round max_seqlen_k to multiples of blocksize_c
  int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
  if (max_seqlen_k <= 128) {
    max_seqlen_k = 128;
  } else if (max_seqlen_k <= 256) {
    max_seqlen_k = 256;
  }
  loop = max_seqlen_k > blocksize_c;
  return max_seqlen_k;
}

int get_max_seqlen_q(int max_seqlen_q_) {
  return ((max_seqlen_q_ + 16 - 1) / 16) * 16;
}

// TODO: is this correct still?
size_t get_softmax_lse_size(int max_seqlen_q_, int batch_size, int num_heads) {
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);
  size_t bytes = sizeof(float) * batch_size * num_heads * max_seqlen_q;

  return bytes;
}

// TODO: is this correct still?
size_t get_o_tmp_size(int max_seqlen_k_, int total_q, int num_heads, int head_size, int v_head_size) {
  bool loop = false;
  get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  return loop ? (sizeof(float) * total_q * num_heads * v_head_size) : 0;
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
               void* q,                          // batch_size x seqlen_q x num_heads x head_size
               void* k,                          // batch_size x seqlen_k x num_heads_k x head_size
               void* v,                          // batch_size x seqlen_k x num_heads_k x head_size
               void* out,                        // batch_size x seqlen_q x num_heads x head_size
               const int batch_size,
               const int num_heads,
               const int num_heads_k,
               const int head_size,
               const int v_head_size,
               const int total_q, //huh
               const int max_seqlen_q_,
               const int max_seqlen_k_,
               const float softmax_scale,
               const bool is_causal) {
  
  ORT_UNUSED_PARAMETER(total_q);
  // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  ORT_ENFORCE(is_sm8x || is_sm90);

  //TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
  // We will support Turing in the near future
  // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

  //auto q_dtype = q.dtype();
  //TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
  //            "FlashAttention only support fp16 and bf16 data type");
  //if (q_dtype == torch::kBFloat16) {
  //  TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
  //}
  //TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  //TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  //TORCH_CHECK(q.is_cuda(), "Input tensor must be on CUDA device");
  //TORCH_CHECK(k.is_cuda(), "Input tensor must be on CUDA device");
  //TORCH_CHECK(v.is_cuda(), "Input tensor must be on CUDA device");

  //TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  //TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  //TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  //TORCH_CHECK(batch_size > 0, "batch size must be postive");
  //TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
  //TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  //CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  //CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  //CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

  // TODO: is this necessary?
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  //at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 128));

  bool loop = false;
  int max_seqlen_k = get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);

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
                   head_size, v_head_size, head_size_rounded,
                   q, k, v, out,
                   /*cu_seqlens_q*/nullptr,
                   /*cu_seqlens_k*/nullptr,
                   nullptr,
                   nullptr,
                   softmax_scale,
                   is_causal);

  run_mha_fwd(params, stream);
  return Status::OK(); // TODO: return from inside run_mha_fwd to make sure status is actually ok

  //Tensor out_padded = out;
  //if (head_size_og % 8 != 0) {
  //  out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}); // TODO: huh?
  //  if (out_.has_value()) {
  //    out_.value().copy_(out);
  //  }
  //}

  //return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
}

Status mha_varlen_fwd(const cudaDeviceProp& dprops,
               cudaStream_t stream,
               void* q,                // half (total_q, num_heads, head_size)
               void* k,                // half (total_k, num_heads, head_size)
               void* v,                // half (total_k, num_heads, v_head_size)
               void* out,              // half (total_q, num_heads, v_head_size)
               int32_t* cu_seqlens_q,  // int (batch_size + 1)
               int32_t* cu_seqlens_k,  // int (batch_size + 1)
               void* softmax_lse_buffer,  // float (batch_size, num_heads, max_seqlen_q)
               void* o_tmp_buffer,        // NULL or float (total_q, num_heads, v_head_size)
               const int batch_size,
               const int num_heads,
               const int num_heads_k,
               const int head_size,
               const int v_head_size,
               const int total_q,
               const int max_seqlen_q_,
               const int max_seqlen_k_,
               const float softmax_scale,
               const bool is_causal/*,
               const int num_splits,
               const bool zero_tensors*/) {
  ORT_UNUSED_PARAMETER(total_q);
  // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  ORT_ENFORCE(is_sm8x || is_sm90);
  //TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
  // We will support Turing in the near future
  // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

  //auto q_dtype = q.dtype();
  //TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
  //            "FlashAttention only support fp16 and bf16 data type");
  //if (q_dtype == torch::kBFloat16) {
  //  TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
  //}
  //TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  //TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  //TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  //TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  //TORCH_CHECK(q.is_cuda(), "Input tensor must be on CUDA device");
  //TORCH_CHECK(k.is_cuda(), "Input tensor must be on CUDA device");
  //TORCH_CHECK(v.is_cuda(), "Input tensor must be on CUDA device");
  //TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be on CUDA device");
  //TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be on CUDA device");

  //TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  //TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  //TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  //TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  //TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");

  constexpr bool return_softmax = false;

  //TORCH_CHECK(batch_size > 0, "batch size must be positive");
  //TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
  //TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  //CHECK_SHAPE(q, total_q, num_heads, head_size_og);
  //CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
  //CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
  //CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  //CHECK_SHAPE(cu_seqlens_k, batch_size + 1

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 128));

  bool loop = false;
  int max_seqlen_k = get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);

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
                   head_size, v_head_size, head_size_rounded,
                   q, k, v, out,
                   cu_seqlens_q,
                   cu_seqlens_k,
                   return_softmax ? o_tmp_buffer : nullptr,
                   softmax_lse_buffer,
                   softmax_scale,
                   is_causal);
  run_mha_fwd(params, stream);
  return Status::OK();  // TODO: return from inside run_mha_fwd to make sure status is actually ok

  //Tensor out_padded = out;
  //if (head_size_og % 8 != 0) {
  //  out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
  //  if (out_.has_value()) {
  //    out_.value().copy_(out);
  //  }
  //}

  //return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
