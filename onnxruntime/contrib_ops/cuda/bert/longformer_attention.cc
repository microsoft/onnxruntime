// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/longformer_global_impl.h"
#include "contrib_ops/cuda/bert/longformer_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "contrib_ops/cuda/bert/longformer_attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LongformerAttention,                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LongformerAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LongformerAttention<T>::LongformerAttention(const OpKernelInfo& info)
    : CudaKernel(info), LongformerAttentionBase(info) {
  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, true);
  use_half4_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseHalf4, true);
}

template <typename T>
Status LongformerAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* attention_mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention_mask = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), attention_mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention_mask->Shape()));
  // Input shapes:
  //   input                 : (batch_size, sequence_length, hidden_size)
  //   weights               : (hidden_size, 3 * hidden_size) -- format 1
  //                           (3, hidden_size, hidden_size)  -- format 0
  //   bias                  : (3 * hidden_size)              -- format 1 (bias for Q, K, V)
  //                           (5 * hidden_size)              -- format 0 (bias for Q, K, V, Global_K, Global_V)
  //   attention_mask        : (batch_size, sequence_length)
  //   global_weights        : (hidden_size, 3 * hidden_size) -- format 1
  //                           (3, hidden_size, hidden_size)  -- format 0
  //   global_bias           : (3 * hidden_size)              -- format 1 (bias for Global_Q, Global_K, Global_V)
  //                           (1 * hidden_size)              -- format 0 (bias for Global_Q)
  //   global_attention_mask : (batch_size, sequence_length)
  // Output shapes:
  //   output         : (batch_size, sequence_length, hidden_size)

  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = GetCublasHandle(context);
  cudaStream_t stream = Stream(context);

  constexpr size_t element_size = sizeof(T);

  // TODO(tianleiwu): only calculate global index once per model instead of once per LongformerAttention node.
  // Build Global Index
  auto global_index_buffer = GetScratchBuffer<int>(static_cast<size_t>(batch_size) * sequence_length, context->GetComputeStream());
  auto batch_global_num_buffer = GetScratchBuffer<int>(batch_size, context->GetComputeStream());

  size_t global_scratch_bytes = GetGlobalScratchSize(sequence_length);
  auto global_scratch_buffer = GetScratchBuffer<void>(global_scratch_bytes, context->GetComputeStream());

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(BuildGlobalIndex(
      device_prop,
      stream,
      global_attention_mask->Data<int>(),
      batch_size,
      sequence_length,
      global_index_buffer.get(),
      batch_global_num_buffer.get(),
      global_scratch_buffer.get(),
      global_scratch_bytes));

  // Copy batch_global_num to CPU
  size_t pinned_buffer_bytes = GetPinnedBufferSize(batch_size);
  auto pinned_buffer = AllocateBufferOnCPUPinned<void>(pinned_buffer_bytes);
  int* batch_global_num_pinned = reinterpret_cast<int*>(pinned_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(batch_global_num_pinned,
                                       batch_global_num_buffer.get(),
                                       batch_size * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& is_copy_done = new_event.Get();

  CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&is_copy_done, cudaEventDisableTiming));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(is_copy_done, stream));

  size_t qkv_size = static_cast<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size;
  // Buffer for GEMM outputs of q, k, v, global_q, global_k and global_v
  // TODO(tianleiwu): compact global_q only need batch_size * window * hidden_size * element_size buffer size.
  auto gemm_buffer = GetScratchBuffer<void>(qkv_size + qkv_size, context->GetComputeStream());

  bool use_merged_qkv_weights = (weights->Shape().NumDimensions() == 2);

  int m = batch_size * sequence_length;
  int n = use_merged_qkv_weights ? 3 * hidden_size : hidden_size;
  int k = hidden_size;
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* input_data = reinterpret_cast<const CudaT*>(input->Data<T>());
  const CudaT* weights_data = reinterpret_cast<const CudaT*>(weights->Data<T>());
  const CudaT* global_weights_data = reinterpret_cast<const CudaT*>(global_weights->Data<T>());

  float one = 1.0f;
  float zero = 0.0f;
  if (use_merged_qkv_weights) {
    // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 0 x B.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        weights_data, n,
        input_data, k,
        &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop, UseTF32()));
  } else {
    // q
    const CudaT* q_weight = weights_data;
    CudaT* q_data = reinterpret_cast<CudaT*>(gemm_buffer.get());
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        q_weight, n,
        input_data, k,
        &zero, q_data, n, device_prop, UseTF32()));
    // k
    const CudaT* k_weight = q_weight + static_cast<int64_t>(hidden_size) * hidden_size;
    CudaT* k_data = q_data + static_cast<int64_t>(batch_size) * sequence_length * hidden_size;
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        k_weight, n,
        input_data, k,
        &zero, k_data, n, device_prop, UseTF32()));

    // v
    const CudaT* v_weight = k_weight + static_cast<int64_t>(hidden_size) * hidden_size;
    CudaT* v_data = k_data + static_cast<int64_t>(batch_size) * sequence_length * hidden_size;
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        v_weight, n,
        input_data, k,
        &zero, v_data, n, device_prop, UseTF32()));
  }

  // Wait for async copy of batch_global_num
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(is_copy_done));

  // Find the maximum number of global tokens in all batches
  int max_num_global = 0;
  for (int i = 0; i < batch_size; ++i) {
    if (max_num_global < batch_global_num_pinned[i]) {
      max_num_global = batch_global_num_pinned[i];
    }
  }

  // Do not use compact memory kernel in the following situations:
  // (1) global tokens > windows size, compact memory kernel cannot be used due to its assumptions.
  // (2) sequence_length == 2 * attention_window, compact memory kernel has parity issue.
  // (3) user sets environment variable ORT_LONGFORMER_COMPACT_MEMORY=0
  bool disable_compact_memory = (max_num_global > window_ || sequence_length == 2 * window_ || !use_compact_memory_);

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run global GEMM.
  CudaT* global_gemm_buffer = nullptr;

  if (max_num_global > 0) {
    global_gemm_buffer = reinterpret_cast<CudaT*>(reinterpret_cast<char*>(gemm_buffer.get()) + qkv_size);

    if (use_merged_qkv_weights) {
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(global_weights->Data<T>()), n,
          input_data, k,
          &zero, global_gemm_buffer, n, device_prop, UseTF32()));
    } else {
      // global q
      const CudaT* global_q_weight = global_weights_data;
      CudaT* global_q = global_gemm_buffer + static_cast<int64_t>(2) * batch_size * sequence_length * hidden_size;
      if (disable_compact_memory) {
        CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
            global_q_weight, n,
            input_data, k,
            &zero, global_q, n, device_prop, UseTF32()));
      } else {
        CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
            cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            hidden_size,                                          // m
            max_num_global,                                       // n
            hidden_size,                                          // k
            &one,                                                 // alpha
            global_q_weight,                                      // A
            hidden_size,                                          // lda
            0,                                                    // strideA
            input_data,                                           // B
            hidden_size,                                          // ldb
            static_cast<int64_t>(sequence_length) * hidden_size,  // strideB
            &zero,                                                // beta
            global_q,                                             // C
            hidden_size,                                          // ldc
            static_cast<int64_t>(max_num_global) * hidden_size,   // strideC
            batch_size,                                           // batch count
            device_prop,
            UseTF32()));
      }
      // global k
      const CudaT* global_k_weight = global_weights_data + static_cast<int64_t>(hidden_size) * hidden_size;
      CudaT* global_k = global_gemm_buffer;
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          global_k_weight, n,
          input_data, k,
          &zero, global_k, n, device_prop, UseTF32()));

      // global v
      const CudaT* global_v_weight = global_k_weight + static_cast<int64_t>(hidden_size) * hidden_size;
      CudaT* global_v = global_gemm_buffer + static_cast<int64_t>(batch_size) * sequence_length * hidden_size;
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          global_v_weight, n,
          input_data, k,
          &zero, global_v, n, device_prop, UseTF32()));
    }
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size,
                                                             batch_size,
                                                             num_heads_,
                                                             head_size,
                                                             sequence_length,
                                                             max_num_global,
                                                             window_,
                                                             disable_compact_memory);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchLongformerAttentionKernel(
      device_prop,
      cublas,
      stream,
      reinterpret_cast<const CudaT*>(gemm_buffer.get()),
      reinterpret_cast<const CudaT*>(bias->Data<T>()),
      reinterpret_cast<const CudaT*>(attention_mask->Data<T>()),
      reinterpret_cast<const CudaT*>(global_gemm_buffer),
      reinterpret_cast<const CudaT*>(global_bias->Data<T>()),
      global_attention_mask->Data<int>(),
      global_index_buffer.get(),
      batch_global_num_buffer.get(),
      pinned_buffer.get(),
      workspace_buffer.get(),
      output->MutableData<T>(),
      batch_size,
      sequence_length,
      num_heads_,
      head_size,
      window_,
      max_num_global,
      element_size,
      disable_compact_memory,
      use_merged_qkv_weights,
      use_half4_));

  // Defer release of pinned memory since cudaStreamSynchronize is not used here and kernel need access the buffer.
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release(), context->GetComputeStream());

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
