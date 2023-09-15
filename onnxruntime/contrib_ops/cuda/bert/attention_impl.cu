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

// Modifications:
// (1) support GPT-2 past state, unidirectional mask and 4D attention mask from Megatron
// (2) support 2D attention mask
// (3) allow persistent softmax from PyTorch for debugging purpose.
// (4) support different input hidden size and model hidden size for pruned model
// (5) support different hidden sizes of Q/K and V
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::attention_softmax_cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kMemoryAlignment = 256;

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t AlignSize(size_t bytes) {
  const size_t bytesAligned = AlignTo(bytes, kMemoryAlignment);
  return bytesAligned;
}

void CumulatedSequenceLengthCache::Initialize(int32_t sequence_length, cudaStream_t stream) {
  if (this->sequence_length != sequence_length) {
    ORT_ENFORCE(buffer.get() != nullptr && this->max_batch_size > 0);
    LaunchTrtSequenceOffset(reinterpret_cast<int32_t*>(buffer.get()), nullptr,
                            this->max_batch_size, sequence_length, stream);
    this->sequence_length = sequence_length;
  }
}

int* GetCumulatedSequenceLength(CumulatedSequenceLengthCache* cache,
                                const int* mask_index,
                                int batch_size,
                                int sequence_length,
                                cudaStream_t stream,
                                void* scratch_buffer) {
  if (mask_index == nullptr && cache != nullptr) {
    if (batch_size <= cache->max_batch_size) {
      cache->Initialize(sequence_length, stream);
      return reinterpret_cast<int*>(cache->buffer.get());
    }
  }

  int* sequence_offset = reinterpret_cast<int*>(scratch_buffer);
  LaunchTrtSequenceOffset(sequence_offset, mask_index, batch_size, sequence_length, stream);
  return sequence_offset;
}

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t total_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * total_sequence_length;
  return AlignSize(bytes);
}

size_t GetSequenceOffsetSize(int batch_size, bool has_padding) {
  // There are batch_size + 1 offsets Without padding (or padding removed), and 2 * batch_size + 1 with padding.
  size_t bytes = sizeof(int) * ((has_padding ? 2 * batch_size : batch_size) + 1);
  return AlignSize(bytes);
  ;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    void* fused_runner,
    bool use_flash_attention,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_bytes = element_size * batch_size * num_heads *
                           ((sequence_length + kv_sequence_length) * qk_head_size + kv_sequence_length * v_head_size);

#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    return qkv_bytes + onnxruntime::flash::get_softmax_lse_size(sequence_length, batch_size, num_heads);
  }
#else
  ORT_UNUSED_PARAMETER(use_flash_attention);
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (use_memory_efficient_attention) {
    size_t fmha_buffer_bytes = 0;
    if (MemoryEfficientAttentionParams::need_workspace(v_head_size, element_size == sizeof(float))) {
      fmha_buffer_bytes = batch_size * sequence_length * num_heads * v_head_size * sizeof(float);
    }

    return qkv_bytes + fmha_buffer_bytes;
  }
#else
  ORT_UNUSED_PARAMETER(use_memory_efficient_attention);
#endif

  if (fused_runner != nullptr) {
    return qkv_bytes + GetSequenceOffsetSize(static_cast<int>(batch_size), true);
  }

  if (use_fused_cross_attention) {
    return qkv_bytes + 2 * GetSequenceOffsetSize(static_cast<int>(batch_size), true);
  }

  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length,
                                                 total_sequence_length);
}

template <typename T>
__global__ void AddBiasTransAppendKvToPresentSmall(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  constexpr int M = 3;           // Matrix count in qkv
  const int m = blockIdx.z + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

template <typename T>
__global__ void AddBiasTransAppendKvToPresent(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = blockIdx.x;
  const int s = blockIdx.y;
  const int b = (blockIdx.z >> 1);
  const int N = gridDim.x;
  const int S = gridDim.y;
  const int B = (gridDim.z >> 1);

  constexpr int M = 3;                   // Matrix count in qkv
  const int m = (blockIdx.z & 0x1) + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

// qkv buffer is merged tensor of shape (B,S,3,N,H), k v is the second/third of the 3.
// bias is of shape (3, NxH) or nullptr
// append to present of (2, B, N, (P..T) of M, H),
template <typename T>
Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                           const int max_sequence_length,
                                           const int past_sequence_length,
                                           const int sequence_length,
                                           const int batch_size,
                                           const int head_size,
                                           const int num_heads,
                                           const int max_threads_per_block,
                                           const T* biases,
                                           const T* qkv_buffer,
                                           T* present) {
  assert(head_size <= (1 << 30));

  int64_t nh = (int64_t)head_size * num_heads;
  if (nh <= max_threads_per_block) {
    const dim3 grid(sequence_length, batch_size, 2);  // 2 for k and v
    const dim3 block(max_threads_per_block / num_heads, num_heads, 1);

    AddBiasTransAppendKvToPresentSmall<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  } else {
    const dim3 grid(num_heads, sequence_length, batch_size * 2);  // 2 for k and v
    const dim3 block(std::min(head_size, max_threads_per_block), 1, 1);
    AddBiasTransAppendKvToPresent<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const float* bias,
                                                    const float* qkv_buffer,
                                                    float* present);

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const half* bias,
                                                    const half* qkv_buffer,
                                                    half* present);

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  constexpr size_t element_size = sizeof(T);
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  const bool past_present_share_buffer = parameters.past_present_share_buffer;
  const float mask_filter_value = parameters.mask_filter_value;
  void* fused_runner = data.fused_runner;

  // At most one fused kernel is enabled.
  assert((int(data.use_flash_attention) +
          int(data.use_memory_efficient_attention) +
          int(fused_runner != nullptr) +
          int(data.fused_cross_attention_kernel != nullptr)) <= 1);

  const int batches = batch_size * num_heads;

  bool use_fused_kernel = (nullptr != fused_runner && !parameters.is_unidirectional);
  bool use_fused_causal = (nullptr != fused_runner && parameters.is_unidirectional);

  QkvData<T> qkv;
  ORT_RETURN_IF_ERROR(PrepareQkv<T>(parameters, data, stream, max_threads_per_block, qkv));
  T* scratch1 = data.has_qkv_workspace ? qkv.after_v : data.workspace;

  int present_size_per_batch_k = 0;
  int present_size_per_batch_v = 0;
  if (!past_present_share_buffer) {
    present_size_per_batch_k = total_sequence_length * qk_head_size;
    present_size_per_batch_v = total_sequence_length * v_head_size;
    ORT_RETURN_IF_ERROR(ConcatPastToPresent(batch_size, num_heads, qk_head_size, v_head_size,
                                            sequence_length, total_sequence_length, parameters.pass_past_in_kv,
                                            stream, max_threads_per_block, data, qkv));

  } else {  // past_present_share_buffer
    assert(qk_head_size == v_head_size);
    assert(data.fused_cross_attention_kernel == nullptr);
    assert(!use_fused_kernel);
    assert(data.gemm_buffer != nullptr);
    assert(!data.use_memory_efficient_attention);
    assert(!data.use_flash_attention);
    assert(data.has_qkv_workspace);

    if (nullptr != data.past_key || nullptr != data.present_key) {
      // TODO: support this case.
      ORT_THROW("buffer sharing for no bias case between past and present is not supported yet.");
    }

    if (data.present != data.past) {
      // For easy testing. Production should better avoid this path.
      int64_t kv_size = 2LL * (int64_t)batch_size * num_heads * parameters.max_sequence_length * qk_head_size;
      cudaMemcpyAsync(data.present, data.past, kv_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    // append last k v to present
    ORT_RETURN_IF_ERROR(LaunchAddBiasTransAppendKvToPresent(
        stream, parameters.max_sequence_length, parameters.past_sequence_length, sequence_length,
        batch_size, qk_head_size, num_heads, max_threads_per_block,
        use_fused_causal ? nullptr : data.bias,  // For fused causal, bias has been added to gemm_buffer
        data.gemm_buffer, data.present));

    present_size_per_batch_k = parameters.max_sequence_length * qk_head_size;
    present_size_per_batch_v = present_size_per_batch_k;
    qkv.k = data.present;
    qkv.v = data.present + batches * present_size_per_batch_k;
  }

  // Q, K and V are ready now
  DUMP_TENSOR_INIT();

  if (data.fused_cross_attention_kernel != nullptr) {
    assert(qkv.format == AttentionQkvFormat::Q_KV_BSNH_BSN2H);

    // We only enable fused cross attention when there is no key padding mask.
    // Otherwise, key have effective batch size 2 * batch_size, which is different from batch_size of query.
    assert(data.mask_index == nullptr);

    int* q_sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_q_cache,
                                                        data.mask_index, batch_size, sequence_length, stream,
                                                        scratch1);

    DUMP_TENSOR_D("q_sequence_offset", q_sequence_offset, 1, batch_size + 1);

    int* kv_sequence_offset = q_sequence_offset + (GetSequenceOffsetSize(batch_size, false) / sizeof(int));
    kv_sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_kv_cache,
                                                    data.mask_index, batch_size, kv_sequence_length, stream,
                                                    kv_sequence_offset);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    DUMP_TENSOR_D("kv_sequence_offset", kv_sequence_offset, 1, batch_size + 1);

    FusedMultiHeadCrossAttentionKernel const* cross_attention_kernel =
        reinterpret_cast<FusedMultiHeadCrossAttentionKernel const*>(data.fused_cross_attention_kernel);

    // When there is no bias, we can directly use q and packed kv from inputs.
    void const* query = qkv.q;
    void const* packed_kv = qkv.k;
    if (data.value == nullptr && data.bias == nullptr) {
      query = data.query;
      packed_kv = data.key;
    }

    run_fused_cross_attention(
        query,                   // Q
        packed_kv,               // packed KV
        q_sequence_offset,       // cumulated sequence length of Q
        kv_sequence_offset,      // cumulated sequence length of KV
        data.output,             // output
        cross_attention_kernel,  // kernels
        batch_size,              // batch size
        num_heads,               // number of heads
        qk_head_size,            // head size of Q/K/V
        sequence_length,         // sequence length of Q
        kv_sequence_length,      // sequence length of KV
        stream);

    DUMP_TENSOR("trt cross output", data.output, batch_size, sequence_length, num_heads, v_head_size);
    return Status::OK();
  }

  // Run TRT fused attention.
  if (use_fused_kernel || use_fused_causal) {
    int* sequence_offset = reinterpret_cast<int*>(scratch1);
    if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
      DUMP_TENSOR_D("mask", reinterpret_cast<const int*>(data.mask_index), batch_size, sequence_length);
      LaunchTrtSequenceOffset2d(sequence_offset, data.mask_index, batch_size, sequence_length, stream);
    } else {
      sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_q_cache,
                                                   data.mask_index, batch_size, sequence_length, stream,
                                                   sequence_offset);
    }
    DUMP_TENSOR_D("sequence_offset", sequence_offset, 1, (data.mask_index != nullptr ? 2 : 1) * batch_size + 1);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(fused_runner);

    const int S = use_fused_causal ? sequence_length : fused_fp16_runner->getSFromMaxSeqLen(sequence_length);

    // B = 2 * batch_size when there is padding in input, and B = batch_size when padding is removed.
    const int B = (nullptr == data.mask_index ? batch_size : 2 * batch_size);

    fused_fp16_runner->setup(S, B);

    if (use_fused_kernel) {
      assert(qkv.format == AttentionQkvFormat::QKV_BSN3H);

      // When there is no bias, we can directly use packed qkv from inputs.
      void const* packed_qkv = qkv.q;
      if (data.query != nullptr && data.key == nullptr && data.bias == nullptr) {
        packed_qkv = data.query;
      }

      fused_fp16_runner->run(packed_qkv, sequence_offset, data.output, stream);
      DUMP_TENSOR("fused output", data.output, batch_size, sequence_length, num_heads, v_head_size);
    } else {
      assert(qkv.format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);
      fused_fp16_runner->run(data.gemm_buffer, sequence_offset, data.output, stream);
      DUMP_TENSOR("fused causal output", data.output, batch_size, sequence_length, num_heads, v_head_size);
    }
    return Status::OK();
  }

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                               : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    assert(qkv.format == AttentionQkvFormat::Q_K_V_BSNH);
    assert(nullptr == data.mask_index);
    assert(nullptr == data.relative_position_bias);
    assert(parameters.head_size == parameters.v_head_size);

    void* query = reinterpret_cast<void*>(qkv.q);
    void* key = reinterpret_cast<void*>(qkv.k);
    void* value = reinterpret_cast<void*>(qkv.v);
    // For packed KV, we can use query input directly.
    if (data.gemm_buffer == nullptr && data.key != nullptr && data.value == nullptr && data.bias == nullptr) {
      query = reinterpret_cast<void*>(const_cast<T*>(data.query));
    }

    DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query), batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", qkv.k, batch_size, parameters.total_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", qkv.v, batch_size, parameters.total_sequence_length, num_heads, v_head_size);

    constexpr bool is_causal = false;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(scratch1),
        parameters.batch_size, parameters.num_heads, parameters.num_heads, parameters.head_size,
        parameters.sequence_length, parameters.total_sequence_length, scale, is_causal));

    DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, v_head_size);

    return Status::OK();
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    // We only enable fused cross attention when there is no key padding mask.
    // Otherwise, key have effective batch size 2 * batch_size, which is different from batch_size of query.
    assert(qkv.format == AttentionQkvFormat::Q_K_V_BSNH);

    const void* query = qkv.q;
    const void* key = qkv.k;
    const void* value = qkv.v;
    // For packed KV, we can use query input directly.
    if (data.gemm_buffer == nullptr && data.key != nullptr && data.value == nullptr) {
      assert(data.bias == nullptr);
      query = data.query;
    }

    DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query), batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", qkv.k, batch_size, parameters.total_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", qkv.v, batch_size, parameters.total_sequence_length, num_heads, v_head_size);

    MemoryEfficientAttentionParams p;
    p.sm = device_prop.major * 10 + device_prop.minor;
    p.is_half = sizeof(T) == 2;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.num_heads;
    p.sequence_length = parameters.sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_unidirectional;
    p.scale = scale;
    p.seqlen_k_ptr = nullptr == data.mask_index
                         ? nullptr
                         : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(data.mask_index));
    p.seqstart_q_ptr = nullptr == data.mask_index
                           ? nullptr
                           : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(data.mask_index + batch_size));
    p.seqstart_k_ptr = nullptr == data.mask_index
                           ? nullptr
                           : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(data.mask_index + 2 * batch_size + 1));
    p.query = query;
    p.key = key;
    p.value = value;
    p.attn_bias = nullptr == data.relative_position_bias ? nullptr : data.relative_position_bias;
    p.is_attn_bias_batched = !parameters.broadcast_res_pos_bias;
    p.output = data.output;
    p.workspace = MemoryEfficientAttentionParams::need_workspace(v_head_size, sizeof(T) == sizeof(float))
                      ? scratch1
                      : nullptr;
    p.stream = stream;
    run_memory_efficient_attention(p);
    DUMP_TENSOR("efficient attention output", data.output, batch_size, sequence_length, num_heads, v_head_size);

    return Status::OK();
  }
#endif

  // The following are unfused attention.
  assert(qkv.format == AttentionQkvFormat::Q_K_V_BNSH);
  const int* mask_index = data.mask_index;
  gsl::span<const int64_t>& mask_index_dims = data.mask_index_dims;

  // Raw attention mask could be 2D (BxT) or 3D (BxSxT) or 4D(Bx1xMxM), where M is the max sequence length.
  bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxT
  // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  float one = 1.0f;
  float zero = 0.f;

  float alpha = use_raw_attention_mask ? one : scale;

  cublasSetStream(cublas, stream);

  DUMP_TENSOR_D("q[BNSH]", q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("k[BNSH]", k, batch_size, num_heads, total_sequence_length, qk_head_size);
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N,
      total_sequence_length, sequence_length, qk_head_size,
      &alpha, qkv.k, qk_head_size, present_size_per_batch_k,
      qkv.q, qk_head_size, sequence_length * qk_head_size,
      &zero, scratch1, total_sequence_length, sequence_length * total_sequence_length, batches, device_prop));

  DUMP_TENSOR_D("Q", qkv.q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("K", qkv.k, batch_size, num_heads, qk_head_size, sequence_length);
  DUMP_TENSOR_D("QK", scratch1, batch_size, num_heads, sequence_length, total_sequence_length);

  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, total_sequence_length);
  T* scratch2 = scratch1 + (bytes / element_size);

  // Apply softmax and store result R to scratch2: BxNxSxT
  if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
    const int mask_dimension = static_cast<int>(mask_index_dims.size());

    // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax used in Torch.
    const TransformerOptions* options = TransformerOptions::GetInstance();
    bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

    T* persistent_softmax_workspace = scratch1;  // replace Q*K' in place with masked score for persistent softmax.
    ORT_RETURN_IF_ERROR(
        ComputeSoftmaxWithRawMask<T>(
            ort_stream, total_sequence_length, sequence_length, batch_size, num_heads,
            mask_index, nullptr, data.relative_position_bias, parameters.broadcast_res_pos_bias,
            scratch1, scratch2, parameters.is_unidirectional, scale, mask_dimension,
            parameters.max_sequence_length, use_persistent_softmax, persistent_softmax_workspace,
            mask_filter_value));
  } else if (nullptr != mask_index) {  // 1d mask index
    assert(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
        stream, total_sequence_length, sequence_length, batch_size, num_heads,
        mask_index, mask_start, data.relative_position_bias, parameters.broadcast_res_pos_bias,
        scratch1, scratch2, parameters.is_unidirectional));
  } else {  // no mask
    ORT_RETURN_IF_ERROR(
        ComputeSoftmax<T>(
            stream, total_sequence_length, sequence_length, batch_size, num_heads, data.relative_position_bias,
            parameters.broadcast_res_pos_bias, scratch1, scratch2, parameters.is_unidirectional));
  }

  DUMP_TENSOR_D("Softmax", scratch2, batch_size, num_heads, sequence_length, total_sequence_length);
  DUMP_TENSOR_D("V", qkv.v, batch_size, num_heads, sequence_length, v_head_size);

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  T* temp_output = qkv.q;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, total_sequence_length,
      &one, qkv.v, v_head_size, present_size_per_batch_v,
      scratch2, total_sequence_length, sequence_length * total_sequence_length,
      &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  Status result = LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
                                 max_threads_per_block, false, temp_output, data.output);
  DUMP_TENSOR("unfused output", data.output, batch_size, sequence_length, num_heads, v_head_size);
  return result;
}

template <typename T>
Status DecoderQkvToContext(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    cublasHandle_t& cublas,
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
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int BN = batch_size * num_heads;
  const int BHN = BN * head_size;
  const int BNS = BN * sequence_length;
  const int k_buffer_offset = sequence_length * BHN;
  const int v_buffer_offset = (sequence_length + kv_sequence_length) * BHN;

  T* temp_qkv_buffer = workspace_buffer;
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());

  const T* q = qkv_buffer;
  // transpose q and copy them to qkv_buffer
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, head_size, num_heads,
                                     max_threads_per_block, true, gemm_query_buffer, qkv_buffer));

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
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length,
                                                       sequence_length, batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1,
                                                       key_cache,
                                                       temp_qkv_buffer,
                                                       qkv_buffer + k_buffer_offset));
      }
      // concat cache-v with v and copy to qkv_buffer
      if (nullptr != value_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length,
                                                       sequence_length, batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1,
                                                       value_cache,
                                                       temp_qkv_buffer + k_buffer_offset,
                                                       qkv_buffer + v_buffer_offset));
      }
    }
  }

  if (has_layer_state) {
    if (use_past && static_kv) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_key_cache, key_cache, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_value_cache, value_cache, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_key_cache, k, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_value_cache, v, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
    }
  }

  // scratch1: BxNxSxL buffer
  // scratch2: BxNxSxL buffer
  // scratch3: BxNxSxH  buffer
  T* scratch1 = temp_qkv_buffer + 3 * BHN * sequence_length;
  T* scratch2 = scratch1 + BNS * kv_sequence_length;
  T* scratch3 = scratch2 + BNS * kv_sequence_length;

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxL
  // Q: BxNxSxH, K (present_k): BxNxLxH, Q*K': BxNxSxL
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * kv_sequence_length;
  float one = 1.0f;
  float zero = 0.f;

  float alpha = rsqrt_head_size;
  const int strideA = kv_sequence_length * head_size;
  const int strideB = sequence_length * head_size;
  if (use_past && static_kv) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        kv_sequence_length, sequence_length, head_size,
        &alpha, key_cache, head_size, strideA,
        q, head_size, strideB,
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, device_prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        kv_sequence_length, sequence_length, head_size,
        &alpha, k, head_size, strideA,
        q, head_size, strideB,
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, device_prop));
  }

  constexpr bool is_unidirectional = false;
  const T* add_before_softmax = nullptr;
  if (has_key_padding_mask) {
    constexpr int mask_dimension = 2;
    constexpr int max_sequence_length = 0;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(
        ort_stream, kv_sequence_length, sequence_length, batch_size,
        num_heads, nullptr, key_padding_mask, add_before_softmax,
        false /*broadcast rpb*/, scratch1, scratch2, is_unidirectional,
        1.0f, mask_dimension, max_sequence_length, false, nullptr,
        mask_filter_value));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(
        stream, kv_sequence_length, sequence_length, batch_size, num_heads,
        add_before_softmax, false /*broadcast rpb*/, scratch1, scratch2,
        is_unidirectional));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, value_cache, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, device_prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, v, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, device_prop));
  }

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads,
                        max_threads_per_block, true, scratch3, output);
}

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& device_prop,
    Stream* stream,
    cublasHandle_t& cublas,
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
        device_prop,
        stream,
        cublas,
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
        device_prop,
        stream,
        cublas,
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

// Template Instantiation
template struct AttentionData<float>;

template struct AttentionData<half>;

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
