// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/rocm/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/rocm/bert/rotary_embedding_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

#ifdef USE_COMPOSABLE_KERNEL_CK_TILE
#include "ck_tile/core/numeric/integer.hpp"
#include "fmha_fwd.hpp"
#endif

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      GroupQueryAttention,                                             \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kRocmExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()) \
          .MayInplace(3, 1)                                            \
          .MayInplace(4, 2)                                            \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                     \
      GroupQueryAttention<T>);

// REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
std::string GetCkFmhaDataTypeString();

template <>
std::string GetCkFmhaDataTypeString<MLFloat16>() {
  return "fp16";
}

template <>
std::string GetCkFmhaDataTypeString<BFloat16>() {
  return "bf16";
}

__global__ void seqlens_inc_kernel(const int* seqlens, int* out, int num_elems, int inc) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num_elems) {
    out[idx] = seqlens[idx] + inc;
  }
}

Status LaunchSeqlensInc(hipStream_t stream, const int* seqlens, int* out, int num_elems, int inc) {
  constexpr int NumThreads = 128;
  int num_blks = CeilDiv(num_elems, NumThreads);
  seqlens_inc_kernel<<<num_blks, NumThreads, 0, stream>>>(seqlens, out, num_elems, inc);
  return HIP_CALL(hipGetLastError());
}

__global__ void seqstart_init_kernel(int* out, int num_elems, int length_per_seq) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num_elems) {
    out[idx] = idx * length_per_seq;
  }
  if (idx == 0) {
    out[num_elems] = num_elems * length_per_seq;
  }
}

Status LaunchSeqStartInit(hipStream_t stream, int* out, int num_elems, int length_per_seq) {
  constexpr int NumThreads = 128;
  int num_blks = CeilDiv(num_elems, NumThreads);
  seqstart_init_kernel<<<num_blks, NumThreads, 0, stream>>>(out, num_elems, length_per_seq);
  return HIP_CALL(hipGetLastError());
}

// Kernel to convert seqlens_k to position_ids
__global__ void SeqlensToPosIdsPrompt(const int32_t* seqlens_k, int64_t* position_ids, const int seqlen,
                                      const int batch_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int b = tid / seqlen;
  int s = tid % seqlen;
  if (b < batch_size) {
    if (s < seqlens_k[b] + 1) {
      position_ids[tid] = s;
    } else {
      position_ids[tid] = 1;
    }
  }
}

// Kernel to convert seqlens_k to position_ids
__global__ void SeqlensToPosIdsToken(const int32_t* seqlens_k, int64_t* position_ids, const int batch_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < batch_size) {
    position_ids[tid] = seqlens_k[tid];
  }
}

// Convert seqlens_k to position_ids
Status LaunchSeqlensToPosIds(contrib::GroupQueryAttentionParameters& parameters, const int32_t* seqlens_k,
                             int64_t* position_ids, hipStream_t stream, const int max_threads_per_block) {
  const int seqlen = parameters.sequence_length;
  const int batch_size = parameters.batch_size;
  const int threads = max_threads_per_block;
  const int blocks = (batch_size * seqlen + threads - 1) / threads;
  if (parameters.is_first_prompt) {
    SeqlensToPosIdsPrompt<<<blocks, threads, 0, stream>>>(seqlens_k, position_ids, seqlen, batch_size);
  } else {
    SeqlensToPosIdsToken<<<blocks, threads, 0, stream>>>(seqlens_k, position_ids, batch_size);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : RocmKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  is_unidirectional_ = true;
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <>
std::once_flag GroupQueryAttention<MLFloat16>::arch_checking_{};

template <>
std::once_flag GroupQueryAttention<BFloat16>::arch_checking_{};

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* ctx) const {
#if USE_COMPOSABLE_KERNEL_CK_TILE
  auto hip_stream = static_cast<hipStream_t>(ctx->GetComputeStream()->GetHandle());
  const Tensor* query = ctx->Input<Tensor>(0);
  const Tensor* key = ctx->Input<Tensor>(1);
  const Tensor* value = ctx->Input<Tensor>(2);
  const Tensor* past_key = ctx->Input<Tensor>(3);
  const Tensor* past_value = ctx->Input<Tensor>(4);
  const Tensor* seqlens_k = ctx->Input<Tensor>(5);
  const Tensor* total_seqlen = ctx->Input<Tensor>(6);
  const Tensor* cos_cache = ctx->Input<Tensor>(7);
  const Tensor* sin_cache = ctx->Input<Tensor>(8);

  auto& device_prop = GetDeviceProp();
  std::call_once(
      arch_checking_,
      [](const hipDeviceProp_t& device_prop) {
        if (std::string_view(device_prop.gcnArchName).find("gfx90a") == std::string_view::npos &&
            std::string_view(device_prop.gcnArchName).find("gfx942") == std::string_view::npos) {
          LOGS_DEFAULT(WARNING)
              << "GroupQueryAttention currently only supports ck_tile fmha backend which only supports "
              << "CDNA2 and CDNA3 archs.";
          LOGS_DEFAULT(WARNING)
              << "GroupQueryAttention running on an unsuppoted GPU may result in "
              << "hipErrorNoBinaryForGpu or hipErrorSharedObjectInitFailedshared error.";
        }
      },
      device_prop);

  GroupQueryAttentionParameters parameters;
  using HipT = typename ToHipType<T>::MappedType;

  const int max_thr_per_blk = device_prop.maxThreadsPerBlock;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlens_k,
                                                                total_seqlen,
                                                                is_past_bsnh_,
                                                                scale_,
                                                                max_thr_per_blk));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  parameters.local_window_size = local_window_size_;
  parameters.is_unidirectional = is_unidirectional_;
  // parameters.zeros_count = kZerosCount;
  // parameters.zero_ptr = zeros_.get();
  // parameters.left_padding = left_padding_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = ctx->Output(0, output_shape);
  Strides output_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads, head_size);

  int4 past_shape;
  std::vector<int64_t> present_dims;
  Strides present_strides;
  Strides past_strides;
  if (past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    past_shape = {
        batch_size, parameters.seqlen_past_kv_cache, kv_num_heads, head_size};
    past_strides = Strides::BSNHMemory(
        batch_size, parameters.seqlen_past_kv_cache, kv_num_heads, head_size);
    present_dims = {
        batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size};
    present_strides = Strides::BSNHMemory(
        batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size);
  } else {  // BNSH
    past_shape = {
        batch_size, kv_num_heads, parameters.seqlen_past_kv_cache, head_size};
    past_strides = Strides::BNSHMemory(
        batch_size, kv_num_heads, parameters.seqlen_past_kv_cache, head_size);
    present_dims = {
        batch_size, kv_num_heads, parameters.seqlen_present_kv_cache, head_size};
    present_strides = Strides::BNSHMemory(
        batch_size, kv_num_heads, parameters.seqlen_present_kv_cache, head_size);
  }
  TensorShape present_shape(present_dims);
  Tensor* present_key = ctx->Output(1, present_shape);
  Tensor* present_value = ctx->Output(2, present_shape);

  Strides query_strides;
  Strides key_strides;
  Strides value_strides;
  int4 kv_shape{batch_size, kv_num_heads, kv_sequence_length, head_size};  // BNSH coord
  const HipT* query_ptr = reinterpret_cast<const HipT*>(query->DataRaw());
  const HipT* key_ptr;
  const HipT* value_ptr;
  if (!parameters.is_packed_qkv) {
    query_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads, head_size);
    key_strides = Strides::BSNHMemory(batch_size, kv_sequence_length, kv_num_heads, head_size);
    value_strides = key_strides;
    key_ptr = reinterpret_cast<const HipT*>(key->DataRaw());
    value_ptr = reinterpret_cast<const HipT*>(value->DataRaw());
  } else {
    query_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size);
    key_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size);
    value_strides = query_strides;
    const size_t key_offset = static_cast<size_t>(num_heads * head_size);
    const size_t value_offset = static_cast<size_t>(kv_num_heads * head_size);
    key_ptr = query_ptr + key_offset;
    value_ptr = key_ptr + value_offset;
  }

  IAllocatorUniquePtr<HipT> rotary_q_tmp;
  IAllocatorUniquePtr<HipT> rotary_k_tmp;
  if (parameters.do_rotary) {
    size_t q_size = static_cast<size_t>(batch_size * sequence_length * num_heads * head_size);
    size_t k_size = static_cast<size_t>(batch_size * sequence_length * kv_num_heads * head_size);
    auto rotary_q_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads, head_size);
    auto rotary_k_strides = Strides::BSNHMemory(batch_size, sequence_length, kv_num_heads, head_size);

    rotary_q_tmp = GetScratchBuffer<HipT>(q_size, ctx->GetComputeStream());
    rotary_k_tmp = GetScratchBuffer<HipT>(k_size, ctx->GetComputeStream());
    auto rotary_position_ids_tmp = GetScratchBuffer<int64_t>(sequence_length * batch_size, ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchSeqlensToPosIds(parameters,
                                              reinterpret_cast<const int32_t*>(seqlens_k->DataRaw()),
                                              reinterpret_cast<int64_t*>(rotary_position_ids_tmp.get()),
                                              hip_stream, max_thr_per_blk));
    // Launch rotary embedding kernel
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<HipT>(hip_stream, rotary_q_tmp.get(), query_ptr,
                                                          reinterpret_cast<int64_t*>(rotary_position_ids_tmp.get()),
                                                          reinterpret_cast<const HipT*>(cos_cache->DataRaw()),
                                                          reinterpret_cast<const HipT*>(sin_cache->DataRaw()),
                                                          parameters.batch_size, parameters.sequence_length,
                                                          parameters.num_heads, parameters.head_size,
                                                          parameters.rotary_dim, parameters.seqlen_present_kv_cache,
                                                          /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                          max_thr_per_blk,
                                                          query_strides.ForBNSHCoord<int4>(),
                                                          rotary_q_strides.ForBNSHCoord<int4>()));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<HipT>(hip_stream, rotary_k_tmp.get(), key_ptr,
                                                          reinterpret_cast<int64_t*>(rotary_position_ids_tmp.get()),
                                                          reinterpret_cast<const HipT*>(cos_cache->DataRaw()),
                                                          reinterpret_cast<const HipT*>(sin_cache->DataRaw()),
                                                          parameters.batch_size, parameters.sequence_length,
                                                          parameters.kv_num_heads, parameters.head_size,
                                                          parameters.rotary_dim, parameters.seqlen_present_kv_cache,
                                                          /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                          max_thr_per_blk,
                                                          key_strides.ForBNSHCoord<int4>(),
                                                          rotary_k_strides.ForBNSHCoord<int4>()));
    query_ptr = reinterpret_cast<const HipT*>(rotary_q_tmp.get());
    key_ptr = reinterpret_cast<const HipT*>(rotary_k_tmp.get());
    query_strides = rotary_q_strides;
    key_strides = rotary_k_strides;
  }

  const int* seqlens_k_ptr = seqlens_k ? reinterpret_cast<const int*>(seqlens_k->DataRaw()) : nullptr;
  IAllocatorUniquePtr<int> seqlens_k_tmp;

  // build present kv cache
  auto* present_key_ptr = reinterpret_cast<HipT*>(present_key->MutableDataRaw());
  auto* present_value_ptr = reinterpret_cast<HipT*>(present_value->MutableDataRaw());
  if (parameters.is_first_prompt) {
    // copy prompt kv to present kv
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, key_ptr, kv_shape, key_strides.ForBNSHCoord(),
                                          present_key_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, value_ptr, kv_shape, value_strides.ForBNSHCoord(),
                                          present_value_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
  } else {
    const auto* past_key_ptr = past_key == nullptr ? nullptr : reinterpret_cast<const HipT*>(past_key->DataRaw());
    const auto* past_value_ptr = past_key == nullptr ? nullptr : reinterpret_cast<const HipT*>(past_value->DataRaw());
    parameters.kv_share_buffer = past_key_ptr == present_key_ptr;  // FIXME:
    if (!parameters.kv_share_buffer) {
      // copy past to present,
      // NOTE: we do a low perf full buffer copy due to the seqlens_k indicate the seqlen of different seqs are
      // not the same, aka, can not be as simple as strided
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, past_key_ptr, past_shape, past_strides.ForBNSHCoord(),
                                            present_key_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, past_value_ptr, past_shape, past_strides.ForBNSHCoord(),
                                            present_value_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
    } else {
      // In the case of share buffer
      ORT_ENFORCE(past_key_ptr == nullptr || past_key_ptr == present_key_ptr);
      ORT_ENFORCE(past_key_ptr == nullptr || past_value_ptr == present_value_ptr);
    }
    // then append new kv to present
    size_t buffer_offset = seqlens_k ? 0 : present_strides.OffsetAt(0, 0, kv_sequence_length, 0);
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(
        hip_stream, key_ptr, kv_shape, key_strides.ForBNSHCoord(), /*in_seqlens_offset=*/nullptr,
        present_key_ptr + buffer_offset, present_strides.ForBNSHCoord(), seqlens_k_ptr,
        max_thr_per_blk));
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(
        hip_stream, value_ptr, kv_shape, value_strides.ForBNSHCoord(), /*in_seqlens_offset=*/nullptr,
        present_value_ptr + buffer_offset, present_strides.ForBNSHCoord(), seqlens_k_ptr,
        max_thr_per_blk));

    // NOTE: ORT: seqlens_k Indicates past sequence lengths for token generation case.
    // we should call fmha with total sequence lengths
    seqlens_k_tmp = GetScratchBuffer<int>(batch_size * sizeof(int), ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchSeqlensInc(hip_stream, seqlens_k_ptr, seqlens_k_tmp.get(), batch_size, sequence_length));
    seqlens_k_ptr = seqlens_k_tmp.get();
  }
  static_assert(std::is_same_v<ck_tile::index_t, int32_t>);

  const float scale = parameters.scale == 0.0f
                          ? 1.f / sqrt(static_cast<float>(parameters.head_size))
                          : parameters.scale;
  bias_enum bias_type = bias_enum::no_bias;

  mask_info mask = [&]() {
    if (local_window_size_ != -1) {
      mask_info ret;
      ret.type = mask_enum::window_generic;
      ret.left = local_window_size_;
      ret.right = parameters.is_unidirectional ? 0 : -1;
      // ret.x = kv_sequence_length - (sequence_length - ret.left);
      // ret.y = sequence_length + (ret.right - kv_sequence_length);
      return ret;
    }

    if (parameters.is_first_prompt && is_unidirectional_) {
      return mask_info::decode("t", sequence_length, kv_sequence_length);
    }

    return mask_info::decode("0", sequence_length, kv_sequence_length);
  }();

  auto seqstart_q_tmp = GetScratchBuffer<int>((batch_size + 1) * sizeof(int), ctx->GetComputeStream());
  auto seqstart_k_tmp = GetScratchBuffer<int>((batch_size + 1) * sizeof(int), ctx->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchSeqStartInit(
      hip_stream, seqstart_q_tmp.get(), batch_size,
      query_strides.strides_for_bnsh_coord.x / query_strides.strides_for_bnsh_coord.z));
  ORT_RETURN_IF_ERROR(LaunchSeqStartInit(
      hip_stream, seqstart_k_tmp.get(), batch_size,
      present_strides.strides_for_bnsh_coord.x / present_strides.strides_for_bnsh_coord.z));

  fmha_fwd_args args{
      query_ptr,
      present_key->DataRaw(),
      present_value->DataRaw(),
      nullptr,  // bias, alibi/element
      nullptr,  // lse, logsumexp buffer
      output->MutableDataRaw(),
      seqstart_q_tmp.get(),        // seqstart_q_ptr, for group mode
      seqstart_k_tmp.get(),        // seqstart_k_ptr, for group mode
      seqlens_k_ptr,               // seqlen_k_ptr, for group mode
      sequence_length,             // seqlen_q, for batch mode
      kv_sequence_length,          // seqlen_k, for batch mode
      parameters.batch_size,       // batch
      parameters.sequence_length,  // max_seqlen_q
      parameters.head_size,        // hdim_q
      parameters.head_size,        // hdim_v
      parameters.num_heads,
      parameters.kv_num_heads,
      scale,
      1.0f,                                                                     // scale_p of squant, useless
      1.0f,                                                                     // scale_o of squant, useless
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.z),    // stride_q, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.z),  // stride_k, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.z),  // stride_v, to be regarded as stride of dim S
      batch_size,                                                               // stride_bias, if alibi, b*h need set this to h, 1*h need set this to 0
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.z),   // stride_o, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.y),    // nhead_stride_q, to be regarded as stride of dim N
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.y),  // nhead_stride_k, to be regarded as stride of dim N
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.y),  // nhead_stride_v, to be regarded as stride of dim N
      0,                                                                        // nhead_stride_bias
      batch_size,                                                               // nhead_stride_lse
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.y),   // batch_stride_o, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.x),    // batch_stride_q, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.x),  // batch_stride_k, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.x),  // batch_stride_v, to be regarded as stride of dim B
      0,                                                                        // batch_stride_bias
      num_heads * batch_size,                                                   // batch_stride_lse
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.x),   // batch_stride_o, to be regarded as stride of dim B
      mask.left,                                                                // window_size_left
      mask.right,                                                               // window_size_right
      static_cast<ck_tile::index_t>(mask.type)};

#if 0
  std::cout
      << "\n  sequence_length:" << sequence_length
      << "\n  kv_sequence_length:" << kv_sequence_length
      << "\n  seqlen_past_kv_cache:" << parameters.seqlen_past_kv_cache
      << "\n  seqlen_present_kv_cache:" << parameters.seqlen_present_kv_cache << std::endl;

  std::cout
      << "\n  q_ptr:" << args.q_ptr
      << "\n  k_ptr:" << args.k_ptr
      << "\n  v_ptr:" << args.v_ptr
      << "\n  bias_ptr:" << args.bias_ptr
      << "\n  lse_ptr:" << args.lse_ptr
      << "\n  o_ptr:" << args.o_ptr
      << "\n  seqstart_q_ptr:" << args.seqstart_q_ptr
      << "\n  seqstart_k_ptr:" << args.seqstart_k_ptr
      << "\n  seqlen_k_ptr:" << args.seqlen_k_ptr
      << "\n  seqlen_q:" << args.seqlen_q
      << "\n  seqlen_k:" << args.seqlen_k
      << "\n  batch:" << args.batch
      << "\n  max_seqlen_q:" << args.max_seqlen_q
      << "\n  hdim_q:" << args.hdim_q
      << "\n  hdim_v:" << args.hdim_v
      << "\n  nhead_q:" << args.nhead_q
      << "\n  nhead_k:" << args.nhead_k
      << "\n  scale_s:" << args.scale_s
      << "\n  scale_p:" << args.scale_p
      << "\n  scale_o:" << args.scale_o
      << "\n  stride_q:" << args.stride_q
      << "\n  stride_k:" << args.stride_k
      << "\n  stride_v:" << args.stride_v
      << "\n  stride_bias:" << args.stride_bias
      << "\n  stride_o:" << args.stride_o
      << "\n  nhead_stride_q:" << args.nhead_stride_q
      << "\n  nhead_stride_k:" << args.nhead_stride_k
      << "\n  nhead_stride_v:" << args.nhead_stride_v
      << "\n  nhead_stride_bias:" << args.nhead_stride_bias
      << "\n  nhead_stride_lse:" << args.nhead_stride_lse
      << "\n  nhead_stride_o:" << args.nhead_stride_o
      << "\n  batch_stride_q:" << args.batch_stride_q
      << "\n  batch_stride_k:" << args.batch_stride_k
      << "\n  batch_stride_v:" << args.batch_stride_v
      << "\n  batch_stride_bias:" << args.batch_stride_bias
      << "\n  batch_stride_lse:" << args.batch_stride_lse
      << "\n  batch_stride_o:" << args.batch_stride_o
      << "\n  window_size_left:" << args.window_size_left
      << "\n  window_size_right:" << args.window_size_right
      << "\n  mask_type:" << args.mask_type
      << std::endl;
#endif

  fmha_fwd_traits traits{
      parameters.head_size,
      parameters.head_size,  // v head size
      GetCkFmhaDataTypeString<T>(),
      !parameters.is_first_prompt,  // true,  // is_group_mode
      true,                   // is_v_rowmajor ? dim is fastest : seq is fastest
      mask.type,
      bias_type,
      false,  // has_lse
      false,  // do_fp8_static_quant, aka, squant
  };

  ck_tile::stream_config stream_config{
      hip_stream,
      false  // time_kernel
  };

  auto duration = fmha_fwd(traits, args, stream_config);
  if (duration < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "fmha_fwd internal error");
  }
  HIP_RETURN_IF_ERROR(hipGetLastError());

  return Status::OK();
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "GroupQueryAttention requires ck_tile to be enabled");
#endif
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
