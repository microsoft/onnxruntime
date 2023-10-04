/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"
#include "contrib_ops/rocm/bert/decoder_attention_impl.h"

using namespace onnxruntime::rocm;

namespace blas = onnxruntime::rocm::tunable::blas;

#define CHECK_ROCM(expr) HIP_RETURN_IF_ERROR(expr)

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t GetAttentionScratchSize(size_t element_size,
                               int batch_size,
                               int num_heads,
                               int sequence_length,
                               int total_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * total_sequence_length;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int total_sequence_length) {
  size_t qkv_size = element_size * 3 * batch_size * sequence_length * num_heads * head_size;
  return qkv_size + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads,
                                                sequence_length, total_sequence_length);
}

inline int3 Get2DMaskStrides(int total_sequence_length) {
  // stride == 0 indicate broadcasting
  return {total_sequence_length, 0, 1};
}

Status ClassifyAttentionMode(
    AttentionType attn_type,
    RocmAttentionParameters* attn,
    const std::vector<const Tensor*>& qkv,
    const std::vector<const Tensor*>& past,
    const std::vector<Tensor*>& present) {
  size_t num_qkv = std::count_if(qkv.cbegin(), qkv.cend(), [](auto it) { return it != nullptr; });
  size_t num_past = std::count_if(past.cbegin(), past.cend(), [](auto it) { return it != nullptr; });
  size_t num_present = std::count_if(present.cbegin(), present.cend(), [](auto it) { return it != nullptr; });

  auto hint = MakeString(num_qkv, " qkv inputs, ", num_past, " past inputs and ", num_present, " present inputs");
  LOGS_DEFAULT(VERBOSE) << hint;

  if (attn_type == kAttention) {
    ORT_ENFORCE(num_qkv == 0);
    if (num_past == 0 && num_present == 0) {
      attn->mode = QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE;
      return Status::OK();
    } else if (num_past == 0 && num_present == 1) {
      if (attn->past_present_share_buffer == false) {
        attn->mode = QFMT_KFMT_VFMT_NONE_NONE_2BNTH_NONE;
        return Status::OK();
      } else {
        attn->mode = QFMT_KFMT_VFMT_NONE_NONE_2BNMH_NONE;
        return Status::OK();
      }
    } else if (num_past == 1 && num_present == 1) {
      if (attn->past_present_share_buffer == false) {
        attn->mode = QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE;
        return Status::OK();
      } else {
        attn->mode = QFMT_KFMT_VFMT_2BNMH_NONE_2BNMH_NONE;
        return Status::OK();
      }
    }
  } else if (attn_type == kMultiHeadAttention || attn_type == kDecoderMaskedMultiHeadAttention) {
    if (num_qkv == 3 && num_past == 0 && num_present == 0) {
      if (attn->qkv_format == Q_K_V_BSNH) {
        attn->mode = BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE;
        return Status::OK();
      } else if (attn->pass_past_in_kv) {
        attn->mode = BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE;
        return Status::OK();
      }
    } else if (num_qkv == 3 && num_past == 0 && num_present == 2) {
      if (attn->past_present_share_buffer == false) {
        if (attn->qkv_format == Q_K_V_BSNH) {
          attn->mode = BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH;
          return Status::OK();
        } else if (attn->pass_past_in_kv) {
          attn->mode = BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH;
          return Status::OK();
        }
      } else {
        if (attn->qkv_format == Q_K_V_BSNH) {
          attn->mode = BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH;
          return Status::OK();
        } else if (attn->pass_past_in_kv) {
          attn->mode = BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH;
          return Status::OK();
        }
      }
    } else if (num_qkv == 3 && num_past == 2 && num_present == 2) {
      if (attn->past_present_share_buffer == false) {
        if (attn->qkv_format == Q_K_V_BSNH) {
          attn->mode = BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH;
          return Status::OK();
        } else if (attn->pass_past_in_kv) {
          attn->mode = BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH;
          return Status::OK();
        }
      } else {
        if (attn->qkv_format == Q_K_V_BSNH) {
          attn->mode = BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH;
          return Status::OK();
        } else if (attn->pass_past_in_kv) {
          attn->mode = BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH;
          return Status::OK();
        }
      }
    } else if (num_qkv == 1 && num_past == 0 && num_present == 0) {
      if (attn->qkv_format == QKV_BSN3H) {
        attn->mode = BLN3H_NONE_NONE_NONE_NONE_NONE_NONE;
        return Status::OK();
      }
    } else if (num_qkv == 2 && num_past == 0 && num_present == 0) {
      if (attn->qkv_format == Q_KV_BSNH_BSN2H) {
        attn->mode = BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE;
        return Status::OK();
      }
    }
  }
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, INVALID_ARGUMENT,
      "Unsupported AttentionMode for ", attn_type, ". Got qkv format ", attn->qkv_format,
      ". Got ", hint);
}

template <typename T>
Status DecoderQkvToContext(
    const hipDeviceProp_t& prop,
    RocmTuningContext* tuning_ctx,
    Stream* ort_stream,
    rocblas_handle& rocblas,
    const size_t element_size,
    const int batch_size,
    const int sequence_length,
    const int kv_sequence_length,
    const int num_heads,
    const int head_size,
    const bool static_kv,
    const bool use_past,
    const bool has_layer_state,
    const bool has_key_padding_mask,
    const float mask_filter_value,
    const T* gemm_query_buffer,
    const T* gemm_kv_buffer,
    const bool* key_padding_mask,
    const T* key_cache,
    const T* value_cache,
    T* qkv_buffer,
    T* workspace_buffer,
    T* output,
    T* new_key_cache,
    T* new_value_cache) {
  const int max_threads_per_block = prop.maxThreadsPerBlock;
  const int BN = batch_size * num_heads;
  const int BHN = BN * head_size;
  const int BNS = BN * sequence_length;
  const int k_buffer_offset = sequence_length * BHN;
  const int v_buffer_offset = (sequence_length + kv_sequence_length) * BHN;

  T* temp_qkv_buffer = workspace_buffer;
  auto stream = static_cast<hipStream_t>(ort_stream->GetHandle());

  const T* q = qkv_buffer;
  // transpose q and copy them to qkv_buffer
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, head_size,
                                     num_heads, max_threads_per_block, true, gemm_query_buffer, qkv_buffer));

  const T* k = qkv_buffer + k_buffer_offset;
  const T* v = qkv_buffer + v_buffer_offset;
  if (!has_layer_state || !use_past) {
    if (!static_kv) {
      // transpose kv and copy them to qkv_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, qkv_buffer + k_buffer_offset));
    } else {
      // transpose kv and copy them to qkv_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, kv_sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, qkv_buffer + k_buffer_offset));
    }
  } else {
    if (!static_kv) {
      // transpose kv and copy them to temp_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, temp_qkv_buffer));
      // concat cache-k with k and copy to qkv_buffer
      if (nullptr != key_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length, sequence_length,
                                                       batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1, key_cache,
                                                       temp_qkv_buffer, qkv_buffer + k_buffer_offset));
      }
      // concat cache-v with v and copy to qkv_buffer
      if (nullptr != value_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length, sequence_length,
                                                       batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1, value_cache,
                                                       temp_qkv_buffer + k_buffer_offset,
                                                       qkv_buffer + v_buffer_offset));
      }
    }
  }

  if (has_layer_state) {
    if (use_past && static_kv) {
      CHECK_ROCM(hipMemcpyAsync(new_key_cache, key_cache,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
      CHECK_ROCM(hipMemcpyAsync(new_value_cache, value_cache,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
    } else {
      CHECK_ROCM(hipMemcpyAsync(new_key_cache, k,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
      CHECK_ROCM(hipMemcpyAsync(new_value_cache, v,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
    }
  }

  // scratch1: BxNxSxS* buffer
  // scratch2: BxNxSxS* buffer
  // scratch3: BxNxSxH  buffer
  T* scratch1 = temp_qkv_buffer + 3 * BHN * sequence_length;
  T* scratch2 = scratch1 + BNS * kv_sequence_length;
  T* scratch3 = scratch2 + BNS * kv_sequence_length;

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * kv_sequence_length;

  const int strideA = kv_sequence_length * head_size;
  const int strideB = sequence_length * head_size;
  if (use_past && static_kv) {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning_ctx, ort_stream, rocblas,
        blas::BlasOp::Trans, blas::BlasOp::NonTrans,
        kv_sequence_length, sequence_length, head_size,
        /*alpha=*/rsqrt_head_size,
        key_cache, head_size, strideA,
        q, head_size, strideB,
        /*beta=*/0.0f,
        scratch1, kv_sequence_length, temp_matrix_size,
        BN));
  } else {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning_ctx, ort_stream, rocblas,
        blas::BlasOp::Trans, blas::BlasOp::NonTrans,
        kv_sequence_length, sequence_length, head_size,
        /*alpha=*/rsqrt_head_size,
        k, head_size, strideA,
        q, head_size, strideB,
        /*beta=*/0.0f,
        scratch1, kv_sequence_length, temp_matrix_size,
        BN));
  }

  if (has_key_padding_mask) {
    int3 strides = Get2DMaskStrides(kv_sequence_length);
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(
        ort_stream, kv_sequence_length, sequence_length, batch_size, num_heads,
        strides, nullptr, key_padding_mask, nullptr, scratch1, scratch2,
        false, 1.0f, false, nullptr, mask_filter_value));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(stream, kv_sequence_length, sequence_length, batch_size,
                                          num_heads, nullptr, scratch1, scratch2, false));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning_ctx, ort_stream, rocblas,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        head_size, sequence_length, kv_sequence_length,
        /*alpha=*/1.0f,
        value_cache, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        /*beta=*/0.0f,
        scratch3, head_size, strideB,
        BN));
  } else {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning_ctx, ort_stream, rocblas,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        head_size, sequence_length, kv_sequence_length,
        /*alpha=*/1.0f,
        v, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        /*beta=*/0.0f,
        scratch3, head_size, strideB,
        BN));
  }

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size,
                        num_heads, max_threads_per_block, true, scratch3, output);
}

Status LaunchDecoderAttentionKernel(
    const hipDeviceProp_t& prop,
    RocmTuningContext* tuning_ctx,
    Stream* stream,
    rocblas_handle& rocblas,
    const size_t element_size,
    const int batch_size,
    const int sequence_length,
    const int kv_sequence_length,
    const int num_heads,
    const int head_size,
    const bool static_kv,
    const bool use_past,
    const bool has_layer_state,
    const bool has_key_padding_mask,
    const float mask_filter_value,
    const void* gemm_query_buffer,
    const void* gemm_kv_buffer,
    const bool* key_padding_mask,
    const void* key_cache,
    const void* value_cache,
    void* qkv_buffer,
    void* workspace_buffer,
    void* output,
    void* new_key_cache,
    void* new_value_cache) {
  if (element_size == 2) {
    return DecoderQkvToContext(
        prop,
        tuning_ctx,
        stream,
        rocblas,
        element_size,
        batch_size,
        sequence_length,
        kv_sequence_length,
        num_heads,
        head_size,
        static_kv,
        use_past,
        has_layer_state,
        has_key_padding_mask,
        mask_filter_value,
        reinterpret_cast<const half*>(gemm_query_buffer),
        reinterpret_cast<const half*>(gemm_kv_buffer),
        key_padding_mask,
        reinterpret_cast<const half*>(key_cache),
        reinterpret_cast<const half*>(value_cache),
        reinterpret_cast<half*>(qkv_buffer),
        reinterpret_cast<half*>(workspace_buffer),
        reinterpret_cast<half*>(output),
        reinterpret_cast<half*>(new_key_cache),
        reinterpret_cast<half*>(new_value_cache));
  } else {
    return DecoderQkvToContext(
        prop,
        tuning_ctx,
        stream,
        rocblas,
        element_size,
        batch_size,
        sequence_length,
        kv_sequence_length,
        num_heads,
        head_size,
        static_kv,
        use_past,
        has_layer_state,
        has_key_padding_mask,
        mask_filter_value,
        reinterpret_cast<const float*>(gemm_query_buffer),
        reinterpret_cast<const float*>(gemm_kv_buffer),
        key_padding_mask,
        reinterpret_cast<const float*>(key_cache),
        reinterpret_cast<const float*>(value_cache),
        reinterpret_cast<float*>(qkv_buffer),
        reinterpret_cast<float*>(workspace_buffer),
        reinterpret_cast<float*>(output),
        reinterpret_cast<float*>(new_key_cache),
        reinterpret_cast<float*>(new_value_cache));
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
