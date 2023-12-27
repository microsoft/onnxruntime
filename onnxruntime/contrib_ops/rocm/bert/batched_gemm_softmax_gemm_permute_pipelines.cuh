// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/* About Computing in these Pipelines

B: batch size of Attention Op. NOTE: To be disambiguated with batch size of GEMMs
S: sequence length
T: total sequence length
N: num of heads
H: head dimension

The following use qkv_format == Q_K_V_BNSH (mode == BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE) as a example:

BN: B*N, which is the batch size of GEMMs. NOTE: To be disambiguated with batch size of Attention Op

In QKV projection (prior to this pipeline):
     /-> Q [B,S,N*H] ->Reshape-> [B,S,N,H] ->Permute0213-> [B,N,S,H]
X --o--> K [B,T,N*H] ->Reshape-> [B,T,N,H] ->Permute0213-> [B,N,T,H]
     \-> V [B,T,N*H] ->Reshape-> [B,T,N,H] ->Permute0213-> [B,N,T,H]

pre_softmax_attn_scores        = Q*K' = [B,N,S,H] * [BxNxTxH]' = [B,N,S,T]               Batched GEMM1
pre_softmax_attn_scores_masked = pre_softmax_attn_scores * scale +? bias +? mask         Scale Add Bias, +? is optional
attn_scores                    = softmax(pre_softmax_attn_scores_masked) = [B,N,S,T]     Softmax
scaled_multi_head_attn         = attn_scores * V = [B,N,S,T] * [B,N,T,H] = [B,N,S,H]     Batched GEMM2

Op outputs scaled_multi_head_attn:
[B,N,S,H] ->Permute0213-> [B,S,N,H] ->Reshape-> [B,S,N*H]


For the computing of pre_softmax_attn_scores +? mask +? bias:

GemmSoftmaxGemmPermuteGenericPipeline handles it in specialized softmax. TODO: remove it!

CK in GemmSoftmaxGemmPermuteTunablePipeline

   Q*K' ---> scale ---> [B,N,S,T] -------+?--> masked
   bias --------------> [B,N,S,T] --+?--/
mask_2d ---> [B,T] ---> [B,1,1,T] -/

   Q*K' ---> scale ---> [B,N,S,T] -------+?--> masked
   bias --------------> [B,N,S,T] --+?--/
mask_3d --> [B,S,T] --> [B,1,S,T] -/

   Q*K' ---> scale ---> [B,N,S,T] -------+?--> masked
   bias --------------> [B,N,S,T] --+?--/
mask_4d -> [B,1,M,M] -> [B,1,S,T] -/                      M is max_sequence_length from megatron, we will create a
                                                          **sub-view** from original mask buffer

For CK implementation, there will be four cases combined:
non-biased, non-masked, no special processing.
    biased, non-masked, no special processing, add the mask directly.
non-biased,     masked, convert the mask to [B,1,1_or_S,T] and perform broadcast add with scaled Q*K'.
    biased,     masked, convert the mask to [B,1,1_or_S,T] and perform broadcast add with bias and scaled Q*K'.

Broadcast add is not actually perform the broadcasting, just broadcast the load operation from memory. The impl details
are in composable kernels. The scale and add logic is performed via Acc0ElementOp

# Classified modes:

| Q    | K    | V    | past(K)| pastV | present(K)| presentV | Op, desc
| ---- | ---- | ---- | ------ | ----- | --------- | -------- | ---------
| QFMT | KFMT | VFMT | -      | -     | -         | -        | A, basic, qkv is impl dependent by qkv_format
| QFMT | KFMT | VFMT | 2BNPH  | -     | 2BNTH *^  | -        | A, past_present_share_buffer = false, qkv is impl dependent by qkv_format
| QFMT | KFMT | VFMT | 2BNMH  | -     | 2BNMH *^  | -        | A, past_present_share_buffer = true,  qkv is impl dependent by qkv_format
| BSNH | BLNH*| BLNH^| -      | -     | -         | -        | MHA basic
| BSNH | BNLH*| BNLH^| -      | -     | -         | -        | MHA cross, pass_past_in_kv = true
| BSNH | -    | -    | -      | -     | BNLH *    | BNLH ^   | MHA cross, pass_past_in_kv = false
| BSNH | BLNH | BLNH | -      | -     | BNTH *    | BNTH ^   | MHA cross, past_present_share_buffer = false
| BSNH | BNLH | BNLH | -      | -     | BNTH *    | BNTH ^   | MHA cross, past_present_share_buffer = false
| BSNH | BLNH | BLNH | -      | -     | BNMH *    | BNMH ^   | MHA cross, past_present_share_buffer = true
| BSNH | BNLH | BNLH | -      | -     | BNMH *    | BNMH ^   | MHA cross, past_present_share_buffer = true
| BSNH | BLNH | BLNH | BNPH   | BNPH  | BNTH *    | BNTH ^   | MHA self, past_present_share_buffer = false
| BSNH | BNLH | BNLH | BNPH   | BNPH  | BNTH *    | BNTH ^   | MHA self, past_present_share_buffer = false
| BSNH | BLNH | BLNH | BNMH   | BNMH  | BNMH *    | BNMH ^   | MHA self, past_present_share_buffer = true
| BSNH | BNLH | BNLH | BNMH   | BNMH  | BNMH *    | BNMH ^   | MHA self, past_present_share_buffer = true
| BLN3H*^| -  | -    | -      | -     | -         | -        | MHA basic, qkv_packed
| BSNH | BLN2H*^| -  | -      | -     | -         | -        | MHA basic, kv_packed

Q, K, V, past(K), pastV, present(K), presentV is the Input of the contrib OpKernel

About k_buffer and v_buffer, we always explicitly concat past to present and use present_k for k_buffer and present_v for v_buffer

- Marked with `*` indicate the Tensor is used for k_buffer passing.
- Marked with `^` indicate the Tensor is used for v_buffer passing.

# Supported Op

- A: Attention
- MHA: MultiHeadAttention

# Dim Value

- B: batch_size
- N: num_heads
- H: head_size

- S: sequence_length
- L: kv_sequence_length
- P: past_sequence_length
- T: total_sequence_length = P + L
- M: max_sequence_length
*/

#include "core/framework/tensor_shape.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"
#ifdef USE_COMPOSABLE_KERNEL
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_ck_impl/impl.cuh"
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif  // USE_COMPOSABLE_KERNEL

#include <array>
#include <vector>

namespace blas = onnxruntime::rocm::tunable::blas;

namespace onnxruntime {
namespace contrib {
namespace rocm {

inline int3 Get2DMaskStrides(int total_sequence_length) {
  // stride == 0 indicate broadcasting
  return {total_sequence_length, 0, 1};
}

// A stride maps from natural coordinate to physical offset of underlying memory storage buffer offset. We need to
// specify both of the natural coordinate order, say (b,n,s,h), (b,s,n,h) or (b,n,h,s), and memory order, say BNSH or
// BSNH, to determain the strides. To obtain the offset, we just do the inner product of coordinate with the strides.
// This wrapper create the stride vector from the physical dimension (or physical shape).
struct Strides {
  // Create the strides for BNSH physically indexed memory buffer
  static Strides BNSHMemory(int batch_dim,
                            int num_head_dim,
                            int seqlen_dim,
                            int head_size_dim) {
    ORT_UNUSED_PARAMETER(batch_dim);
    return Strides{longlong4{
        static_cast<int64_t>(num_head_dim) * seqlen_dim * head_size_dim,
        static_cast<int64_t>(seqlen_dim) * head_size_dim,
        static_cast<int64_t>(head_size_dim),
        static_cast<int64_t>(1),
    }};
  }

  // Create the strides for BSNH physically indexed memory buffer
  static Strides BSNHMemory(int batch_dim,
                            int seqlen_dim,
                            int num_head_dim,
                            int head_size_dim) {
    ORT_UNUSED_PARAMETER(batch_dim);
    return Strides{longlong4{
        static_cast<int64_t>(seqlen_dim) * num_head_dim * head_size_dim,
        static_cast<int64_t>(head_size_dim),
        static_cast<int64_t>(num_head_dim) * head_size_dim,
        static_cast<int64_t>(1),
    }};
  }

  template <typename T = longlong4>
  T ForBNSHCoord() const {
    using E = typename T::value_type;
    return T{static_cast<E>(strides_for_bnsh_coord.x),
             static_cast<E>(strides_for_bnsh_coord.y),
             static_cast<E>(strides_for_bnsh_coord.z),
             static_cast<E>(strides_for_bnsh_coord.w)};
  }

  template <typename T = longlong4>
  T ForBSNHCoord() const {
    using E = typename T::value_type;
    return T{static_cast<E>(strides_for_bnsh_coord.x),
             static_cast<E>(strides_for_bnsh_coord.z),
             static_cast<E>(strides_for_bnsh_coord.y),
             static_cast<E>(strides_for_bnsh_coord.w)};
  }

  template <typename T = longlong4>
  T ForBNHSCoord() const {
    using E = typename T::value_type;
    return T{static_cast<E>(strides_for_bnsh_coord.x),
             static_cast<E>(strides_for_bnsh_coord.y),
             static_cast<E>(strides_for_bnsh_coord.w),
             static_cast<E>(strides_for_bnsh_coord.z)};
  }

  int64_t OffsetAt(int b, int n, int s, int h) const {
    return strides_for_bnsh_coord.x * b + strides_for_bnsh_coord.y * n +
           strides_for_bnsh_coord.z * s + strides_for_bnsh_coord.w * h;
  }

  // store intermediate strides in the canonical (b,n,s,h) coordinate order
  longlong4 strides_for_bnsh_coord;
};

template <typename HipT, typename T>
std::tuple<const HipT*, const HipT*, const HipT*> ConvertToOffsetedBufferViews(
    const RocmAttentionParameters* attn,
    const T* query = nullptr,    // q or packed_qkv
    const T* key = nullptr,      // k or packed kv
    const T* value = nullptr,    //
    const T* present = nullptr,  // present or present_k
    const T* present_v = nullptr) {
  switch (attn->mode) {
    case QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE:
    case BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE:
    case BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE: {
      return {reinterpret_cast<const HipT*>(query),
              reinterpret_cast<const HipT*>(key),
              reinterpret_cast<const HipT*>(value)};
    }
    case QFMT_KFMT_VFMT_NONE_NONE_2BNTH_NONE:
    case QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE: {
      auto offset = static_cast<int64_t>(attn->batch_size) * attn->num_heads * attn->total_sequence_length *
                    attn->head_size;
      return {reinterpret_cast<const HipT*>(query),
              reinterpret_cast<const HipT*>(present),
              reinterpret_cast<const HipT*>(present) + offset};
    }
    case QFMT_KFMT_VFMT_NONE_NONE_2BNMH_NONE:
    case QFMT_KFMT_VFMT_2BNMH_NONE_2BNMH_NONE: {
      auto offset = static_cast<int64_t>(attn->batch_size) * attn->num_heads * attn->max_sequence_length *
                    attn->head_size;
      return {reinterpret_cast<const HipT*>(query),
              reinterpret_cast<const HipT*>(present),
              reinterpret_cast<const HipT*>(present) + offset};
    }
    case BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH:
    case BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH:
    case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
    case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
    case BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH:
    case BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH:
    case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
    case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
      return {reinterpret_cast<const HipT*>(query),
              reinterpret_cast<const HipT*>(present),
              reinterpret_cast<const HipT*>(present_v)};
    case BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE: {
      auto packed_kv = reinterpret_cast<const HipT*>(key);
      return {reinterpret_cast<const HipT*>(query), packed_kv, packed_kv + attn->head_size};
    }
    case BLN3H_NONE_NONE_NONE_NONE_NONE_NONE: {
      auto packed_qkv = reinterpret_cast<const HipT*>(query);
      return {packed_qkv, packed_qkv + 1 * attn->head_size, packed_qkv + 2 * attn->head_size};
    }
    default:
      ORT_ENFORCE("unreachable");
      return {};
  }
}

inline std::tuple<Strides, Strides, Strides> GetQkvStrides(const RocmAttentionParameters* attn) {
  // G0 not used, because it is the slowest dimension
  const int& B = attn->batch_size;
  const int& N = attn->num_heads;
  const int& S = attn->sequence_length;
  const int& L = attn->kv_sequence_length;
  const int& T = attn->total_sequence_length;
  const int& M = attn->max_sequence_length;
  const int& H = attn->head_size;
  const int& Hv = attn->v_head_size;

  switch (attn->mode) {
    case QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE:
      if (attn->qkv_format == Q_K_V_BNSH) {
        return {
            Strides::BNSHMemory(B, N, S, H),
            Strides::BNSHMemory(B, N, L, H),
            Strides::BNSHMemory(B, N, L, Hv),
        };
      } else if (attn->qkv_format == Q_K_V_BSNH) {
        return {
            Strides::BSNHMemory(B, S, N, H),
            Strides::BSNHMemory(B, L, N, H),
            Strides::BSNHMemory(B, L, N, Hv),
        };
      }
    case BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH:
    case BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH:
    case BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH:
    case BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH:
      return {
          Strides::BSNHMemory(B, S, N, H),
          Strides::BNSHMemory(B, N, T, H),
          Strides::BNSHMemory(B, N, T, Hv),
      };
    case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
    case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
    case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
    case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
      return {
          Strides::BSNHMemory(B, S, N, H),
          Strides::BNSHMemory(B, N, M, H),
          Strides::BNSHMemory(B, N, M, Hv),
      };
    case BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE:
      return {
          Strides::BSNHMemory(B, S, N, H),
          Strides::BSNHMemory(B, L, N, H),
          Strides::BSNHMemory(B, L, N, Hv),
      };
    case BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE:
      return {
          Strides::BSNHMemory(B, S, N, H),
          Strides::BNSHMemory(B, N, L, H),
          Strides::BNSHMemory(B, N, L, Hv),
      };
    case BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE:
      return {
          Strides::BSNHMemory(B, S, N, H),
          Strides::BSNHMemory(B, L, N, 2 * H),
          Strides::BSNHMemory(B, L, N, 2 * Hv),
      };
    case BLN3H_NONE_NONE_NONE_NONE_NONE_NONE:
      return {
          Strides::BSNHMemory(B, L, N, 3 * H),
          Strides::BSNHMemory(B, L, N, 3 * H),
          Strides::BSNHMemory(B, L, N, 3 * Hv),
      };
    default:
      ORT_ENFORCE("unreachable");
      return {};
  }
}

inline std::tuple<const int*, int3, int3> GetRawMaskBufferAddrSizesAndStrides(
    const int* buffer, const RocmAttentionParameters* attn) {
  const int* offseted_buffer{buffer};  // how to view the mask buffer
  int3 sizes{0, 0, 0};                 // the logical shape of the view
  int3 strides{-1, -1, -1};            // the physical memory layout
  switch (attn->mask_type) {
    case MASK_NONE:
    case MASK_2D_DUMMY:
      break;  // No mask
    case MASK_2D_KEY_PADDING:
      sizes = {attn->batch_size, 1, attn->total_sequence_length};
      strides = Get2DMaskStrides(attn->total_sequence_length);
      break;
    case MASK_3D_ATTENTION:
      sizes = {attn->batch_size, attn->sequence_length, attn->total_sequence_length};
      strides = {attn->sequence_length * attn->total_sequence_length, attn->total_sequence_length, 1};
      break;
    case MASK_4D_MEGATRON:
      // offset to skip past sequence part, so that we can index it with [batch_index, sequence_index, token_index]
      offseted_buffer = buffer + attn->past_sequence_length * attn->max_sequence_length;
      sizes = {attn->batch_size, attn->sequence_length, attn->total_sequence_length};
      strides = {attn->max_sequence_length * attn->max_sequence_length, attn->max_sequence_length, 1};
      break;
    default:
      LOGS_DEFAULT(FATAL) << "unsupported mask type: " << attn->mask_type;
      throw std::runtime_error("unsupported mask type");
  }
  return {offseted_buffer, sizes, strides};
}

template <typename T>
struct GemmSoftmaxGemmPermuteParams : onnxruntime::rocm::tunable::OpParams {
  std::string Signature() const override {
    return MakeString(
        "B", attention->batch_size,
        "_S", attention->sequence_length,
        "_T", attention->total_sequence_length,
        "_N", attention->num_heads,
        "_H", attention->head_size,
        "_Hv", attention->v_head_size,
        bias_buffer != nullptr ? "_B" : "_NB",
        "_M", mask_index_dims.size(),
        "_QKV", attention->qkv_format,
        "_MODE", attention->mode);
  }

  std::tuple<int, int, int, int, int> GetGemmsMNKOBatch() const {
    ORT_ENFORCE(attention != nullptr);
    auto m = attention->sequence_length;
    auto n = attention->total_sequence_length;
    auto k = attention->head_size;
    auto o = attention->v_head_size;
    auto batch = attention->batch_size * attention->num_heads;
    return {m, n, k, o, batch};
  }

  rocblas_handle handle;
  const RocmAttentionParameters* attention;
  const hipDeviceProp_t* device_prop;

  float scale;
  const T* q_buffer;
  const T* k_buffer;
  const T* v_buffer;
  T* out_buffer;

  // optional, bias [B,N,S,T]
  const T* bias_buffer{nullptr};

  // optional, mask value
  const int* mask_index_buffer{nullptr};
  TensorShapeVector mask_index_dims{};

  // optional, internal
  void* workspace_buffer{nullptr};
};

inline bool IsKVBNMH(AttentionMode mode) {
  switch (mode) {
    case QFMT_KFMT_VFMT_NONE_NONE_2BNMH_NONE:
    case QFMT_KFMT_VFMT_2BNMH_NONE_2BNMH_NONE:
    case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
    case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
    case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
    case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
      return true;
    default:
      return false;
  }
}

template <typename T>
struct GemmSoftmaxGemmPermuteGenericPipeline {
  static bool UseRawAttentionMask(const GemmSoftmaxGemmPermuteParams<T>* params) {
    return params->mask_index_buffer != nullptr && params->mask_index_dims.size() >= 2;
  }

  static std::tuple<T*, T*, T*> GetWorkspacePlan(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto bytes = GetAttentionScratchSize(
        sizeof(T),
        params->attention->batch_size,
        params->attention->num_heads,
        params->attention->sequence_length,
        params->attention->total_sequence_length);
    auto gemm1_out = reinterpret_cast<T*>(params->workspace_buffer);
    auto softmax_out = gemm1_out + (bytes / sizeof(T));
    auto gemm2_out = softmax_out + (bytes / sizeof(T));
    return {gemm1_out, softmax_out, gemm2_out};
  }

  inline static size_t GetWorkspaceNumBytes(const RocmAttentionParameters* attn) {
    return GetAttentionWorkspaceSize(
        sizeof(T),
        attn->batch_size,
        attn->num_heads,
        attn->head_size,
        attn->sequence_length,
        attn->total_sequence_length);
  }

  inline static Status Gemm1(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto [m, n, k, o, batch] = params->GetGemmsMNKOBatch();
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);

    int k_buffer_stride = n * k;
    if (IsKVBNMH(params->attention->mode)) {
      k_buffer_stride = params->attention->max_sequence_length * params->attention->head_size;
    }

    // GEMM1 [m,k] * [n,k]' -> [m,n]
    return blas::row_major::StridedBatchedGemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::Trans,
        m, n, k,
        // For raw attention mask, the scalar is moved to softmax computation.
        /*alpha=*/UseRawAttentionMask(params) ? 1.0f : params->scale,
        params->q_buffer, k, m * k,
        params->k_buffer, k, k_buffer_stride,
        /*beta=*/0.0f,
        gemm1_out, n, m * n,
        batch);
  }

  inline static Status SoftmaxRawMask(const GemmSoftmaxGemmPermuteParams<T>* params, bool use_persistent_softmax) {
    // Softmax on [m,n] along the n dimension.
    // Raw attention mask could be 2D (B,S) or 3D (B,S,T) or 4D(B,1,M,M), where M is the max sequence length.
    auto attn = params->attention;
    auto [buffer, sizes, strides] = GetRawMaskBufferAddrSizesAndStrides(params->mask_index_buffer, attn);
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    T* persistent_softmax_workspace = gemm1_out;  // replace Q*K' in place if persistent softmax is selected.
    return ComputeSoftmaxWithRawMask<T>(
        params->Stream(), attn->total_sequence_length, attn->sequence_length, attn->batch_size, attn->num_heads,
        strides, buffer, nullptr, params->bias_buffer, gemm1_out, softmax_out,
        attn->is_unidirectional, /* FIXME: this must not be attn.scale! */ params->scale,
        use_persistent_softmax, persistent_softmax_workspace, attn->mask_filter_value);
  }

  inline static Status Softmax1DIndexMask(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto mask_1d = params->mask_index_buffer;
    auto mask_1d_size = params->mask_index_dims[0];
    // Softmax on [m,n] along the n dimension.
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    auto attn = params->attention;
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    const int* mask_start = (mask_1d_size > attn->batch_size) ? mask_1d + attn->batch_size : nullptr;
    return ComputeSoftmaxWithMask1D<T>(
        params->StreamHandle(), attn->total_sequence_length, attn->sequence_length, attn->batch_size, attn->num_heads,
        mask_1d, mask_start, params->bias_buffer, gemm1_out, softmax_out, attn->is_unidirectional);
  }

  inline static Status SoftmaxNoMask(const GemmSoftmaxGemmPermuteParams<T>* params) {
    // Softmax on [m,n] along the n dimension.
    auto attn = params->attention;
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    return ComputeSoftmax<T>(
        params->StreamHandle(), attn->total_sequence_length, attn->sequence_length, attn->batch_size, attn->num_heads,
        params->bias_buffer, gemm1_out, softmax_out, attn->is_unidirectional);
  }

  inline static Status Gemm2(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto [m, n, k, o, batch] = params->GetGemmsMNKOBatch();
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);

    int v_buffer_stride = n * o;
    if (IsKVBNMH(params->attention->mode)) {
      v_buffer_stride = params->attention->max_sequence_length * params->attention->v_head_size;
    }

    // GEMM2 [m,n] * [n,o] -> [m,o]
    // semantically, the output buffer contains B*N matrices of shape [S,H], compactly, thus B,N,S,H.
    return blas::row_major::StridedBatchedGemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        m, o, n,
        /*alpha=*/1.0f,
        softmax_out, n, m * n,
        params->v_buffer, o, v_buffer_stride,
        /*beta=*/0.0f,
        gemm2_out, o, m * o,
        batch);
  }

  inline static Status Permute0213(const GemmSoftmaxGemmPermuteParams<T>* params) {
    // Permute 0213
    // gemm2_out is B,N,S,H, transpose to out_buffer as B,S,N,H
    auto attn = params->attention;
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    return LaunchTransCtx(
        params->StreamHandle(),
        attn->sequence_length, attn->batch_size, attn->head_size, attn->num_heads,
        params->device_prop->maxThreadsPerBlock, false, gemm2_out, params->out_buffer);
  }

  static Status GetSupportedStatus(const GemmSoftmaxGemmPermuteParams<T>* params) {
    const auto& attn = params->attention;
    // TODO: address the BNMH k,v strides
    switch (attn->mode) {
      case QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE:
      case QFMT_KFMT_VFMT_NONE_NONE_2BNTH_NONE:
      case QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE:
      case QFMT_KFMT_VFMT_NONE_NONE_2BNMH_NONE:
      case QFMT_KFMT_VFMT_2BNMH_NONE_2BNMH_NONE:
        if (attn->qkv_format == Q_K_V_BNSH) {
          return Status::OK();
        } else {
          return TUNABLE_OP_UNSUPPORTED("GenericPipeline only supports qkv_format as Q_K_V_BNSH, got ",
                                        attn->qkv_format);
        }
      case BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE:
        return TUNABLE_OP_UNSUPPORTED("GenericPipeline only supports qkv_format as Q_K_V_BNSH but k, v are BLNH");
      case BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE:
      case BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH:
      case BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
      case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
      case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
        // If sequence_length is 1, query of B1NH can be simply viewed as BN1H.
        if (attn->sequence_length == 1) {
          return Status::OK();
        } else {
          return TUNABLE_OP_UNSUPPORTED("GenericPipeline only supports qkv_format as Q_K_V_BNSH, ",
                                        "only if sequence_length is 1, query of BSNH can be viewed as BNSH");
        }
      case BLN3H_NONE_NONE_NONE_NONE_NONE_NONE:
      case BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE:
        return TUNABLE_OP_UNSUPPORTED("GenericPipeline only supports qkv_format as Q_K_V_BNSH");
      default:
        return TUNABLE_OP_UNSUPPORTED("unknonw");
    }
    return TUNABLE_OP_UNSUPPORTED("unknonw case");
  }

  static Status Run(const GemmSoftmaxGemmPermuteParams<T>* params, bool use_persistent_softmax) {
    auto supported_status = GetSupportedStatus(params);
    if (!supported_status.IsOK()) {
      return supported_status;
    }
    ORT_RETURN_IF_ERROR(Gemm1(params));

    if (UseRawAttentionMask(params)) {
      ORT_RETURN_IF_ERROR(SoftmaxRawMask(params, use_persistent_softmax));
    } else if (params->mask_index_dims.size() == 1) {  // 1d index mask
      ORT_RETURN_IF_ERROR(Softmax1DIndexMask(params));
    } else {
      ORT_RETURN_IF_ERROR(SoftmaxNoMask(params));
    }

    ORT_RETURN_IF_ERROR(Gemm2(params));
    ORT_RETURN_IF_ERROR(Permute0213(params));
    return Status::OK();
  }
};

template <typename T>
class GemmSoftmaxGemmPermuteTunableOp : public tunable::TunableOp<GemmSoftmaxGemmPermuteParams<T>> {
 public:
  GemmSoftmaxGemmPermuteTunableOp();

  inline static bool IsSupportedMode(const RocmAttentionParameters* attn) {
    switch (attn->mode) {
      case QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE:
      case QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE:
        // depends on qkv format
        if (attn->qkv_format == Q_K_V_BNSH || attn->qkv_format == Q_K_V_BSNH) {
          return true;
        } else {
          return false;
        }
      case BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
      case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
      case BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE:
      case BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE:
      case BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH:
      case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
      case BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE:
      case BLN3H_NONE_NONE_NONE_NONE_NONE_NONE:
        return true;
      default:
        return false;
    }
  }

  inline static bool IsSupportedMaskType(const RocmAttentionParameters* attn) {
    switch (attn->mask_type) {
      case MASK_NONE:
      case MASK_2D_DUMMY:
      case MASK_2D_KEY_PADDING:
      case MASK_3D_ATTENTION:
      case MASK_4D_MEGATRON:
        return true;
      default:
        return false;
    }
  }

  inline static size_t GetWorkspaceNumBytes(const RocmAttentionParameters* attn) {
    size_t num_bytes = GemmSoftmaxGemmPermuteGenericPipeline<T>::GetWorkspaceNumBytes(attn);

#ifdef USE_COMPOSABLE_KERNEL
    if (IsSupportedMaskType(attn)) {
      auto [buffer, sizes, strides] = GetRawMaskBufferAddrSizesAndStrides(nullptr, attn);
      num_bytes = std::max(num_bytes, sizeof(T) * sizes.x * sizes.y * sizes.z);
    }
#endif

    return num_bytes;
  }

  template <int VecSize, typename Converter>
  __global__ static void ConvertToFilledMaskValue(
      T* __restrict__ out,
      const int3 out_strides,
      const int* __restrict__ mask_buffer,
      const int3 mask_lengths,  // [B,S,T]
      const int3 mask_strides,
      Converter cvt) {
    const int64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx >= mask_lengths.x * mask_lengths.y * CeilDiv(mask_lengths.z, VecSize)) {
      return;
    }

    const int tidx = (global_idx % CeilDiv(mask_lengths.z, VecSize)) * VecSize;
    const int bs_idx = global_idx / CeilDiv(mask_lengths.z, VecSize);
    const int sidx = bs_idx % mask_lengths.y;
    const int bidx = bs_idx / mask_lengths.y;

    int64_t in_offset = mask_strides.x * bidx + mask_strides.y * sidx + mask_strides.z * tidx;
    int64_t out_offset = out_strides.x * bidx + out_strides.y * sidx + out_strides.z * tidx;

    if (tidx + VecSize <= mask_lengths.z) {
      using LoadT = const aligned_vector<int, VecSize>;
      using StoreT = aligned_vector<T, VecSize>;
      LoadT load = *reinterpret_cast<LoadT*>(mask_buffer + in_offset);
      StoreT store;

#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        store.val[i] = cvt(load.val[i]);
      }
      *reinterpret_cast<StoreT*>(out + out_offset) = store;
    } else {
#pragma unroll
      for (int i = 0; i < mask_lengths.z - tidx; i++) {
        out[out_offset + i] = cvt(mask_buffer[in_offset + i]);
      }
    }
  }

  static Status LaunchConvertToFilledMaskValue(const GemmSoftmaxGemmPermuteParams<T>* params) {
    constexpr const int kThreadPerBlock = 256;
    constexpr const int kVecSize = 4;

    auto attn = params->attention;
    auto [buffer, lengths, strides] = GetRawMaskBufferAddrSizesAndStrides(params->mask_index_buffer, attn);
    int64_t total_threads = lengths.x * lengths.y * CeilDiv(lengths.z, kVecSize);
    auto num_blocks = CeilDiv(total_threads, kThreadPerBlock);

    auto mask_filter_value = attn->mask_filter_value;
    auto cvt = [=] __device__(int v) -> T {
      return v == 1 ? 0 : mask_filter_value;
    };

    ConvertToFilledMaskValue<kVecSize><<<num_blocks, kThreadPerBlock, 0, params->StreamHandle()>>>(
        reinterpret_cast<T*>(params->workspace_buffer), {lengths.y * lengths.z, lengths.z, 1},  // out desc
        buffer, lengths, strides,                                                               // mask desc
        cvt);

    return HIP_CALL(hipGetLastError());
  }
};

#ifdef USE_COMPOSABLE_KERNEL

template <typename U, typename V, typename T, bool USE_BIAS, bool USE_MASK>
auto GetArgAndRunInvoker(const U& impl, const V& invoker, const GemmSoftmaxGemmPermuteParams<T>* params) {
  constexpr const int kNumBiasBuffer = static_cast<int>(USE_BIAS) + static_cast<int>(USE_MASK);

  using Nop = ck::tensor_operation::element_wise::PassThrough;
  using Acc0ElementOp = internal::PreSoftmaxAttentionScoreOp;

  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      !GemmSoftmaxGemmPermuteTunableOp<T>::IsSupportedMode(params->attention),
      "attention mode is not supported, got ", params->attention->mode);
  if constexpr (USE_BIAS) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        params->bias_buffer == nullptr, "biased version only support input with bias");
  } else {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        params->bias_buffer != nullptr, "non-biased version only support input without bias");
  }
  if constexpr (USE_MASK) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !GemmSoftmaxGemmPermuteTunableOp<T>::IsSupportedMaskType(params->attention),
        "mask type is not supported, got ", params->attention->mask_type);
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        params->mask_index_buffer == nullptr, "masked version only support input with mask");
  } else {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        params->mask_index_buffer != nullptr, "non-masked version only support input without mask");
  }

  auto attn = params->attention;
  const int& G0 = attn->batch_size;
  const int& G1 = attn->num_heads;
  const int& M = attn->sequence_length;
  const int& N = attn->total_sequence_length;
  const int& K = attn->head_size;
  const int& O = attn->v_head_size;
  {
    auto [m, n, k, o, batch] = params->GetGemmsMNKOBatch();
    ORT_ENFORCE(M == m && N == n && K == k && O == o && G0 * G1 == batch, "semantic mismatch");
  }

  auto [qs, ks, vs] = GetQkvStrides(attn);
  std::vector<ck::index_t> q_buffer_lengths = {G0, G1, M, K};
  std::vector<ck::index_t> q_buffer_strides = qs.template ForBNSHCoord<std::vector<ck::index_t>>();
  std::vector<ck::index_t> k_buffer_lengths = {G0, G1, N, K};
  std::vector<ck::index_t> k_buffer_strides = ks.template ForBNSHCoord<std::vector<ck::index_t>>();
  std::vector<ck::index_t> v_buffer_lengths = {G0, G1, O, N};
  std::vector<ck::index_t> v_buffer_strides = vs.template ForBNHSCoord<std::vector<ck::index_t>>();
  std::vector<ck::index_t> out_buffer_lengths = {G0, G1, M, O};
  std::vector<ck::index_t> out_buffer_strides = {M * G1 * O, O, G1 * O, 1};  // permute 0213

  std::array<void*, kNumBiasBuffer> bias_buffers{};
  std::array<std::vector<ck::index_t>, kNumBiasBuffer> bias_lengths{};
  std::array<std::vector<ck::index_t>, kNumBiasBuffer> bias_strides{};
  if constexpr (USE_BIAS) {
    bias_buffers[0] = const_cast<T*>(params->bias_buffer);
    bias_lengths[0] = {G0, G1, M, N};  // BN(G0*G1), S(M), T(N)
    bias_strides[0] = {G1 * M * N, M * N, N, 1};
  }
  if constexpr (USE_MASK) {
    bias_buffers[kNumBiasBuffer - 1] = params->workspace_buffer;
    bias_lengths[kNumBiasBuffer - 1] = {G0, G1, M, N};  // BN(G0*G1), S(M), T(N)
    if (params->mask_index_dims.size() == 2) {          // [B,T]
      bias_strides[kNumBiasBuffer - 1] = {N, 0, 0, 1};
    } else if (params->mask_index_dims.size() == 3) {  // [B,S,T]
      bias_strides[kNumBiasBuffer - 1] = {M * N, 0, N, 1};
    } else if (params->mask_index_dims.size() == 4) {  // [B,1,max_seq_len,max_seq_len] -->convert--> [B,S,T]
      bias_strides[kNumBiasBuffer - 1] = {M * N, 0, N, 1};
    } else {
      ORT_ENFORCE(false, "Unreachable");
    }
  }

  auto arg = impl->MakeArgumentPointer(
      params->q_buffer, params->k_buffer, params->v_buffer, params->out_buffer,
      bias_buffers,  // Gemm1 bias, as attention mask
      {},            // Gemm2 bias
      q_buffer_lengths, q_buffer_strides,
      k_buffer_lengths, k_buffer_strides,
      v_buffer_lengths, v_buffer_strides,
      out_buffer_lengths, out_buffer_strides,
      bias_lengths, bias_strides,
      {},
      {},
      Nop{},
      Nop{},
      Acc0ElementOp{params->scale},
      Nop{},
      Nop{});

  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                            impl->GetTypeString(), " does not support the params");

  if constexpr (USE_MASK) {
    ORT_RETURN_IF_ERROR(GemmSoftmaxGemmPermuteTunableOp<T>::LaunchConvertToFilledMaskValue(params));
  }

  invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
  return Status::OK();
}

template <typename T, bool USE_BIAS, bool USE_MASK>
auto GetCKGemmSoftmaxGemmPermuteTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using D0DataType = typename ck::detail::tuple_concat<
      std::conditional_t<USE_BIAS, ck::Tuple<CKDataType>, ck::Tuple<>>,
      std::conditional_t<USE_MASK, ck::Tuple<CKDataType>, ck::Tuple<>>>::type;

  constexpr static auto MaskingSpecMaskDisabled =
      ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
  constexpr static auto MaskingSpecMaskOutUpperTriangle =
      ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle;

  std::vector<std::pair<std::string, Op<GemmSoftmaxGemmPermuteParams<T>>>>
      ret;

  for (auto&& impl : internal::GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
           CKDataType, D0DataType, internal::F32, internal::PreSoftmaxAttentionScoreOp, MaskingSpecMaskDisabled>()) {
    auto type_string = impl->GetTypeString();

    auto invoker = impl->MakeInvokerPointer();
    auto op = [impl = std::move(impl), invoker = std::move(invoker)](
                  const GemmSoftmaxGemmPermuteParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->attention->is_unidirectional, "unidirectional attention is not supported with MaskingSpecMaskDisabled");

      return GetArgAndRunInvoker<decltype(impl), decltype(invoker), T, USE_BIAS, USE_MASK>(impl, invoker, params);
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(op)));
  }

  for (auto&& impl : internal::GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
           CKDataType, D0DataType, internal::F32, internal::PreSoftmaxAttentionScoreOp, MaskingSpecMaskOutUpperTriangle>()) {
    auto type_string = impl->GetTypeString();

    auto invoker = impl->MakeInvokerPointer();
    auto op = [impl = std::move(impl), invoker = std::move(invoker)](
                  const GemmSoftmaxGemmPermuteParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          !params->attention->is_unidirectional, "bidirectional attention is not supported with MaskingSpecMaskOutUpperTriangle");
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->attention->sequence_length != params->attention->total_sequence_length,
          "seqence_length != total_seqence_length is not supported with MaskingSpecMaskOutUpperTriangle");

      return GetArgAndRunInvoker<decltype(impl), decltype(invoker), T, USE_BIAS, USE_MASK>(impl, invoker, params);
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(op)));
  }

  return ret;
}
#endif  // USE_COMPOSABLE_KERNEL

template <typename T>
GemmSoftmaxGemmPermuteTunableOp<T>::GemmSoftmaxGemmPermuteTunableOp() {
  this->RegisterOp([](const GemmSoftmaxGemmPermuteParams<T>* params) {
    return GemmSoftmaxGemmPermuteGenericPipeline<T>::Run(params, false);
  });

#ifdef USE_COMPOSABLE_KERNEL
  for (auto&& [_, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_BIAS=*/false, /*USE_MASK=*/false>()) {
    this->RegisterOp(std::move(op));
  }

  for (auto&& [_, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_BIAS=*/true, /*USE_MASK=*/false>()) {
    this->RegisterOp(std::move(op));
  }

  for (auto&& [_, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_BIAS=*/false, /*USE_MASK=*/true>()) {
    this->RegisterOp(std::move(op));
  }

  for (auto&& [_, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, /*USE_BIAS=*/true, /*USE_MASK=*/true>()) {
    this->RegisterOp(std::move(op));
  }
#endif
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
