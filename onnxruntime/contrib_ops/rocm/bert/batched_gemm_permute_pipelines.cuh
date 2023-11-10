// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

namespace blas = onnxruntime::rocm::tunable::blas;

namespace {
std::tuple<int, int, int, int> GetQkvProjectGemmMNKBatch(const AttentionParameters* attention) {
  int m = attention->sequence_length;
  int n = (attention->hidden_size + attention->hidden_size + attention->v_hidden_size);  // q + k + v
  int k = attention->input_hidden_size;
  int batch = attention->batch_size;
  return {m, n, k, batch};
}
}  // namespace

template <typename T>
struct GemmPermuteParams : onnxruntime::rocm::tunable::OpParams {
  std::string Signature() const override {
    auto [m, n, k, batch] = GetQkvProjectGemmMNKBatch(attention);
    return MakeString("M", m, "_N", n, "_K", k, "_B", batch);
  }

  rocblas_handle handle;
  const AttentionParameters* attention;
  const hipDeviceProp_t* device_prop;

  const T* input_buffer;
  const T* weight_buffer;
  const T* bias_buffer;
  T* out_buffer;

  int3 bias_strides;

  const T* ones; // used for broadcasting bias if the underlying algorithm does not support strides
  T* workspace_buffer;
};

template <typename T>
struct GemmPermuteGenericPipeline {
  inline static size_t GetOutputNumBytes(const AttentionParameters* attn) {
    auto [m, n, _, batch] = GetQkvProjectGemmMNKBatch(attn);
    return sizeof(T) * m * n * batch;
  }

  inline static size_t GetWorkspaceNumBytes(const AttentionParameters* attn) {
    return GetOutputNumBytes(attn);
  }

  inline static std::tuple<int, int, int> GetGemmMNK(const GemmPermuteParams<T>* params) {
    auto [m, n, k, batch] = GetQkvProjectGemmMNKBatch(params->attention);
    return {batch * m, n, k};
  }

  inline static std::tuple<const T*, const T*, const T*> UnspliceOutputQKV(const GemmPermuteParams<T>* params) {
    auto* attn = params->attention;
    int64_t batch = attn->batch_size * attn->num_heads;
    int64_t num_elems_per_batch = attn->sequence_length * attn->head_size;
    int64_t num_elems = batch * num_elems_per_batch;
    auto q = params->out_buffer + 0 * num_elems;
    auto k = params->out_buffer + 1 * num_elems;
    auto v = params->out_buffer + 2 * num_elems;
    return {q, k, v};
  }

  inline static Status BroadcastBias(const GemmPermuteParams<T>* params) {
    auto [m, n, k] = GetGemmMNK(params);
    // Bias shape is (N), broadcast using B(M, N) = ones(M, 1) x bias(1, N).
    // TODO: use custom kernel of expand to improve the performance.
    return blas::row_major::Gemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        m, n, 1,
        /*alpha=*/1.0f,
        params->ones, 1,
        params->bias_buffer, n,
        /*beta=*/0.0f,
        params->workspace_buffer, n);
  }

  inline static Status Gemm(const GemmPermuteParams<T>* params) {
    auto [m, n, k] = GetGemmMNK(params);
    // result(M, N) = input x weights  + bias.
    return blas::row_major::Gemm(
        params->TuningContext(), params->Stream(), params->handle,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        m, n, k,
        /*alpha=*/1.0f,
        params->input_buffer, k,
        params->weight_buffer, n,
        /*beta=*/1.0f,
        params->workspace_buffer, n);
  }

  inline static Status Permute0213(const GemmPermuteParams<T>* params) {
    auto* attn = params->attention;
    // input should be BxSx3xNxH => gemm_buffer: 3xBxNxSxH
    return LaunchTransQkv(
        params->Stream(), 3, attn->sequence_length, attn->batch_size, attn->head_size, attn->num_heads,
        params->device_prop->maxThreadsPerBlock, false, params->workspace_buffer, params->out_buffer);
  }

  static Status Run(const GemmPermuteParams<T>* params) {
    ORT_RETURN_IF_ERROR(BroadcastBias(params));
    ORT_RETURN_IF_ERROR(Gemm(params));
    ORT_RETURN_IF_ERROR(Permute0213(params));
    return Status::OK();
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
