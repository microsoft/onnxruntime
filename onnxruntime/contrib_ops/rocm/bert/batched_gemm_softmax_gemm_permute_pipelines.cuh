// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/* About Computing in these Pipelines

B: batch size of Attention Op. NOTE: To be disambiguated with batch size of GEMMs
S: sequence length
T: total sequence length
N: num of heads
H: head dimension

BN: B*N, which is the batch size of GEMMs. NOTE: To be disambiguated with batch size of Attention Op

In QKV projection (prior to this pipeline):
     /-> Q [B,S,N*H] ->Reshape-> [B,S,N,H] ->Permute0213-> [B,N,S,H]
X --o--> K [B,T,N*H] ->Reshape-> [B,T,N,H] ->Permute0213-> [B,N,T,H]
     \-> V [B,T,N*H] ->Reshape-> [B,T,N,H] ->Permute0213-> [B,N,T,H]

pre_softmax_attn_scores        = Q*K' = [B,N,S,H] * [BxNxTxH]' = [B,N,S,T]                      Batched GEMM1
pre_softmax_attn_scores_masked = pre_softmax_attn_scores +? bias +? mask                        Add Bias, +? is optional
attn_scores                    = softmax(pre_softmax_attn_scores_masked * scale) = [B,N,S,T]    Scale then Softmax
scaled_multi_head_attn         = attn_scores * V = [B,N,S,T] * [B,N,T,H] = [B,N,S,H]            Batched GEMM2

Op outputs scaled_multi_head_attn:
[B,N,S,H] ->Permute0213-> [B,S,N,H] ->Reshape-> [B,S,N*H]


For the computing of pre_softmax_attn_scores +? mask +? bias:

GemmSoftmaxGemmPermuteGenericPipeline handles it in specialized softmax. TODO: remove it!

*/

#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"

namespace blas = onnxruntime::rocm::tunable::blas;

namespace onnxruntime {
namespace contrib {
namespace rocm {

inline int3 Get2DMaskStrides(int total_sequence_length) {
  // stride == 0 indicate broadcasting
  return {total_sequence_length, 0, 1};
}

inline std::tuple<const int*, int3, int3> GetRawMaskBufferAddrSizesAndStrides(
    const int* buffer, const AttentionParameters* attn) {
  const int* offseted_buffer{buffer};  // how to view the mask buffer
  int3 sizes{-1, -1, -1};              // the logical shape of the view
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
      throw std::runtime_error("unsupported mask type");
  }
  return {offseted_buffer, sizes, strides};
}

template <typename T>
struct GemmSoftmaxGemmPermuteParams : onnxruntime::rocm::tunable::OpParams {
  std::string Signature() const override {
    auto [m, n, k, o, batch] = GetGemmsMNKOBatch();
    return MakeString("M", m, "_N", n, "_K", k, "_O", o, "_B", batch);
  }

  std::tuple<int, int, int, int, int> GetGemmsMNKOBatch() const {
    ORT_ENFORCE(attention != nullptr);
    auto m = attention->sequence_length;
    auto n = attention->total_sequence_length;
    auto k = attention->head_size;
    auto o = attention->head_size;
    auto batch = attention->batch_size * attention->num_heads;
    return {m, n, k, o, batch};
  }

  rocblas_handle handle;
  const AttentionParameters* attention;
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
  gsl::span<const int64_t> mask_index_dims{};

  // optional, internal
  T* workspace_buffer{nullptr};
};

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
    auto gemm1_out = params->workspace_buffer;
    auto softmax_out = gemm1_out + (bytes / sizeof(T));
    auto gemm2_out = softmax_out + (bytes / sizeof(T));
    return {gemm1_out, softmax_out, gemm2_out};
  }

  inline static size_t GetWorkspaceNumBytes(const AttentionParameters* attn) {
    return GetAttentionWorkspaceSize(
        sizeof(T),
        attn->batch_size,
        attn->num_heads,
        attn->head_size,
        attn->sequence_length,
        attn->past_sequence_length);
  }

  inline static Status Gemm1(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto [m, n, k, o, batch] = params->GetGemmsMNKOBatch();
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    // GEMM1 [m,k] * [n,k]' -> [m,n]
    return blas::row_major::StridedBatchedGemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::Trans,
        m, n, k,
        // For raw attention mask, the scalar is moved to softmax computation.
        /*alpha=*/UseRawAttentionMask(params) ? 1.0f : params->scale,
        params->q_buffer, k, m * k,
        params->k_buffer, k, n * k,
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
        params->Stream(), attn->total_sequence_length, attn->sequence_length, attn->batch_size, attn->num_heads,
        mask_1d, mask_start, params->bias_buffer, gemm1_out, softmax_out, attn->is_unidirectional);
  }

  inline static Status SoftmaxNoMask(const GemmSoftmaxGemmPermuteParams<T>* params) {
    // Softmax on [m,n] along the n dimension.
    auto attn = params->attention;
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    return ComputeSoftmax<T>(
        params->Stream(), attn->total_sequence_length, attn->sequence_length, attn->batch_size, attn->num_heads,
        params->bias_buffer, gemm1_out, softmax_out, attn->is_unidirectional);
  }

  inline static Status Gemm2(const GemmSoftmaxGemmPermuteParams<T>* params) {
    auto [m, n, k, o, batch] = params->GetGemmsMNKOBatch();
    auto [gemm1_out, softmax_out, gemm2_out] = GetWorkspacePlan(params);
    // GEMM2 [m,n] * [n,o] -> [m,o]
    // semantically, the output buffer contains B*N matrices of shape [S,H], compactly, thus B,N,S,H.
    return blas::row_major::StridedBatchedGemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        m, o, n,
        /*alpha=*/1.0f,
        softmax_out, n, m * n,
        params->v_buffer, o, n * o,
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
        params->Stream(),
        attn->sequence_length, attn->batch_size, attn->head_size, attn->num_heads,
        params->device_prop->maxThreadsPerBlock, false, gemm2_out, params->out_buffer);
  }

  static Status Run(const GemmSoftmaxGemmPermuteParams<T>* params, bool use_persistent_softmax) {
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


template<typename T>
class GemmSoftmaxGemmPermuteTunableOp : public tunable::TunableOp<GemmSoftmaxGemmPermuteParams<T>> {
 public:
  GemmSoftmaxGemmPermuteTunableOp() {
    this->RegisterOp([](const GemmSoftmaxGemmPermuteParams<T>* params) {
      return GemmSoftmaxGemmPermuteGenericPipeline<T>::Run(params, nullptr, false);
    });
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
