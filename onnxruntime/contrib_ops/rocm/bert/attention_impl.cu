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
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

using namespace onnxruntime::rocm;
using namespace hipcub;

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
    int past_sequence_length) {
  size_t qkv_size = element_size * 3 * batch_size * sequence_length * num_heads * head_size;
  return qkv_size + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads,
                                                sequence_length, past_sequence_length + sequence_length);
}


template<typename T>
struct GemmSoftmaxGemmPermutePararms : onnxruntime::rocm::tunable::OpParams {
  // - GEMM1 [m,k] * [n,k]' -> [m,n]
  // - Apply softmax along n dimension
  // - GEMM2 [m,n] * [n,o] -> [m,o]
  // - Permute 0213

  std::string Signature() const override {
    return MakeString("M", m, "_N", n, "_K",k, "_O", o, "_B", batch);
  }

  void FillShape(const AttentionParameters& attention) {
    batch = attention.batch_size * attention.num_heads;
    m = attention.sequence_length;
    n = attention.total_sequence_length;
    k = attention.head_size;
    o = attention.head_size;
  }

  rocblas_handle handle;

  int batch;
  int m;
  int n;
  int k;
  int o;
  float scale;
  const T* q_buffer;
  const T* k_buffer;
  const T* v_buffer;
  T* out_buffer;
};

template <typename T>
Status GemmSoftmaxGemmPermuteGeneric(
    const GemmSoftmaxGemmPermutePararms<T>& params,
    const AttentionParameters& attn,
    const int max_threads_per_block,
    T* gemm1_out,
    T* softmax_out,
    T* gemm2_out,
    const int* mask_index,
    gsl::span<const int64_t> mask_index_dims,
    const T* relative_position_bias,
    bool use_persistent_softmax) {
  // Raw attention mask could be 2D (BxS) or 3D (BxSxS*) or 4D(Bx1xMxM), where M is the max sequence length.
  bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // GEMM1 [m,k] * [n,k]' -> [m,n]
  ORT_RETURN_IF_ERROR(blas::row_major::StridedBatchedGemm(
      params.TuningContext(), params.Stream(), params.handle,
      blas::BlasOp::NonTrans, blas::BlasOp::Trans,
      params.m, params.n, params.k,
      // For raw attention mask, the scalar is moved to softmax computation.
      /*alpha=*/use_raw_attention_mask ? 1.0f : params.scale,
      params.q_buffer, params.k, params.m * params.k,
      params.k_buffer, params.k, params.n * params.k,
      /*beta=*/0.0f,
      gemm1_out, params.n, params.m * params.n,
      params.batch));

  // Softmax on [m,n] along the n dimension.
  if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
    // Raw attention mask could be 2D (BxS) or 3D (BxSxS*) or 4D(Bx1xMxM), where M is the max sequence length.
    auto* mask = mask_index;
    int4 strides;
    const int mask_dimension = static_cast<int>(mask_index_dims.size());
    if (mask_dimension == 2) {
      strides = {attn.total_sequence_length, 0, 0, 1};
    } else if (mask_dimension == 3) {
      strides = {attn.sequence_length * attn.total_sequence_length, 0, attn.total_sequence_length, 1};
    } else if (mask_dimension == 4) {
      int max_sequence_length = mask_dimension == 4 ? static_cast<int>(mask_index_dims[3]) : 0;
      strides = {max_sequence_length * max_sequence_length, max_sequence_length, max_sequence_length, 1};
      // offset to skip past sequence part, so that we can index it with [batch_index, 0, sequence_index, token_index]
      mask = mask + attn.past_sequence_length * max_sequence_length;
    }

    T* persistent_softmax_workspace = gemm1_out;  // replace Q*K' in place if persistent softmax is selected.
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(
        params.Stream(), attn.total_sequence_length, attn.sequence_length, attn.batch_size, attn.num_heads,
        strides, mask, nullptr, relative_position_bias, gemm1_out, softmax_out,
        attn.is_unidirectional, /* FIXME: this must not be attn.scale! */params.scale,
        use_persistent_softmax, persistent_softmax_workspace, attn.mask_filter_value));
  } else if (nullptr != mask_index) {  // 1d mask index
    ORT_ENFORCE(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > attn.batch_size) ? mask_index + attn.batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
        params.Stream(), attn.total_sequence_length, attn.sequence_length, attn.batch_size, attn.num_heads,
        mask_index, mask_start, relative_position_bias, gemm1_out, softmax_out, attn.is_unidirectional));
  } else {  // no mask
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(
        params.Stream(), attn.total_sequence_length, attn.sequence_length, attn.batch_size, attn.num_heads,
        relative_position_bias, gemm1_out, softmax_out, attn.is_unidirectional));
  }

  // GEMM2 [m,n] * [n,o] -> [m,o]
  // semantically, the output buffer contains B*N matrices of shape [S,H], compactly, thus BxNxSxH.
  ORT_RETURN_IF_ERROR(blas::row_major::StridedBatchedGemm(
      params.TuningContext(), params.Stream(), params.handle,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      params.m, params.o, params.n,
      /*alpha=*/1.0f,
      softmax_out, params.n, params.m * params.n,
      params.v_buffer, params.o, params.n * params.o,
      /*beta=*/0.0f,
      gemm2_out, params.o, params.m * params.o,
      params.batch));

  // Permute 0213
  // gemm2_out is BxNxSxH, transpose to out_buffer as BxSxNxH
  return LaunchTransCtx(params.Stream(),
                        attn.sequence_length, attn.batch_size, attn.head_size, attn.num_heads,
                        max_threads_per_block, false, gemm2_out, params.out_buffer);
}

template <typename T>
Status DecoderQkvToContext(
    const hipDeviceProp_t& prop,
    RocmTuningContext* tuning_ctx,
    hipStream_t stream,
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
        tuning_ctx, stream, rocblas,
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
        tuning_ctx, stream, rocblas,
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
    int4 strides = {sequence_length, 0, 0, 1};
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(
        stream, kv_sequence_length, sequence_length, batch_size, num_heads,
        strides, nullptr, key_padding_mask, nullptr, scratch1, scratch2,
        false, 1.0f, false, nullptr, mask_filter_value));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(stream, kv_sequence_length, sequence_length, batch_size,
                           num_heads, nullptr, scratch1, scratch2, false));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning_ctx, stream, rocblas,
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
        tuning_ctx, stream, rocblas,
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
    hipStream_t stream,
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


constexpr int kPastSequenceLengthInputIndex = 6;
constexpr int kPastInputIndex = 4;
constexpr int kPresentOutputIndex = 1;

template <typename T>
class Attention final : public RocmKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
};

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Attention,                                                               \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kRocmExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : RocmKernel(info), AttentionBase(info, true) {}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  relative_position_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len));
  ORT_ENFORCE(parameters.sequence_length == parameters.kv_sequence_length);  // self attention

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);

  rocblas_handle rocblas = GetRocblasHandle(context);
  constexpr size_t element_size = sizeof(T);

  int m = parameters.batch_size * parameters.sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  typedef typename ToHipType<T>::MappedType HipT;
  namespace blas = tunable::blas;

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      GetTuningContext(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, 1,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(bias->Data<T>()), n,
      GetConstOnes<HipT>(m, Stream(context)), 1,
      /*beta=*/0.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // result(N, M) = 1 * weights x input + 1 x B.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      GetTuningContext(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, k,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(weights->Data<T>()), n,
      reinterpret_cast<const HipT*>(input->Data<T>()), k,
      /*beta=*/1.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  size_t workspace_size = GetAttentionWorkspaceSize(element_size,
                                                    parameters.batch_size,
                                                    parameters.num_heads,
                                                    parameters.head_size,
                                                    parameters.sequence_length,
                                                    parameters.past_sequence_length);

  auto workspace = GetScratchBuffer<void>(workspace_size, context->GetComputeStream());

  const size_t bytes = GetAttentionScratchSize(element_size, parameters.batch_size, parameters.num_heads,
                                               parameters.sequence_length, parameters.total_sequence_length);
  HipT* scratch1 = reinterpret_cast<HipT*>(workspace.get());
  HipT* scratch2 = scratch1 + (bytes / element_size);
  HipT* scratch3 = scratch2 + (bytes / element_size);

  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  ORT_RETURN_IF_ERROR(LaunchTransQkv(Stream(context), 3, parameters.sequence_length, parameters.batch_size, parameters.head_size, parameters.num_heads,
                      device_prop.maxThreadsPerBlock, false, reinterpret_cast<HipT*>(gemm_buffer.get()), scratch3));

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int batches = parameters.batch_size * parameters.num_heads;
  const int size_per_batch = parameters.sequence_length * parameters.head_size;
  const int total_size = batches * size_per_batch;

  const HipT* q_buffer = scratch3;
  const HipT* k_buffer = q_buffer + total_size;
  const HipT* v_buffer = k_buffer + total_size;

  rocblas_set_stream(rocblas, Stream(context));

  // Concat past (2xBxNxS'xH) to present (2xBxNxS*xH):
  // past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxS*xH)
  // past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxS*xH)
  const int present_size_per_batch = parameters.total_sequence_length * parameters.head_size;
  if (nullptr != present) {
    ORT_RETURN_IF_ERROR(
      LaunchConcatPastToPresent(Stream(context),
                                parameters.total_sequence_length,
                                parameters.sequence_length,
                                parameters.batch_size,
                                parameters.head_size,
                                parameters.num_heads,
                                device_prop.maxThreadsPerBlock,
                                nullptr == past ? nullptr : reinterpret_cast<const HipT*>(past->DataRaw()),
                                k_buffer,
                                reinterpret_cast<HipT*>(present->MutableDataRaw())));

    // update pointers to present_k and present_v.
    k_buffer = reinterpret_cast<HipT*>(present->MutableDataRaw());
    v_buffer = reinterpret_cast<HipT*>(present->MutableDataRaw()) + batches * present_size_per_batch;
  }

  // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax
  const TransformerOptions* options = TransformerOptions::GetInstance();
  bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

  GemmSoftmaxGemmPermutePararms<HipT> gemm_softmax_gemm_permute_params;
  {
    auto& params = gemm_softmax_gemm_permute_params;
    params.tuning_ctx = GetTuningContext();
    params.stream = Stream(context);
    params.handle = rocblas;
    params.FillShape(parameters);
    // FIXME: the params.scale seems to be different from AttentionParameters::scale;
    params.scale = 1.0f / sqrt(static_cast<float>(parameters.head_size));
    params.q_buffer = q_buffer;
    params.k_buffer = k_buffer;
    params.v_buffer = v_buffer;
    params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());
  }

  return GemmSoftmaxGemmPermuteGeneric(
      gemm_softmax_gemm_permute_params,
      parameters,
      device_prop.maxThreadsPerBlock,
      scratch1,
      scratch2,
      scratch3,
      nullptr == mask_index ? nullptr : mask_index->Data<int>(),
      nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
      nullptr == relative_position_bias ? nullptr : reinterpret_cast<const HipT*>(relative_position_bias->DataRaw()),
      use_persistent_softmax);
}


}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
