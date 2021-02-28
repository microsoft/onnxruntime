// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "longformer_attention.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "longformer_global_impl.h"
#include "longformer_attention_impl.h"

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
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LongformerAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

// A wrapper class of cudaEvent_t to destroy the event automatically for avoiding memory leak.
class AutoDestoryCudaEvent {
 public:
  AutoDestoryCudaEvent() : cuda_event_(nullptr) {
  }

  ~AutoDestoryCudaEvent() {
    if (cuda_event_ != nullptr)
      cudaEventDestroy(cuda_event_);
  }

  cudaEvent_t& Get() {
    return cuda_event_;
  }
 private:
  cudaEvent_t cuda_event_;
};

template <typename T>
LongformerAttention<T>::LongformerAttention(const OpKernelInfo& info) : CudaKernel(info), LongformerAttentionBase(info) {
  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, false);
}

template <typename T>
Status LongformerAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention->Shape()));

  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Output 0 - output     : (batch_size, sequence_length, hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = CublasHandle();
  cudaStream_t stream = Stream();
  CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas, stream));

  constexpr size_t element_size = sizeof(T);

  // TODO: only calculate once per model.
  // Build Global Index
  auto global_index_buffer = GetScratchBuffer<int>(batch_size * sequence_length);
  auto batch_global_num_buffer = GetScratchBuffer<int>(batch_size);

  size_t global_scratch_bytes = GetGlobalScratchSize(batch_size, sequence_length);
  auto global_scratch_buffer = GetScratchBuffer<void>(global_scratch_bytes);

  BuildGlobalIndex(
      stream,
      global_attention->template Data<int>(),
      batch_size,
      sequence_length,
      global_index_buffer.get(),
      batch_global_num_buffer.get(),
      global_scratch_buffer.get(),
      global_scratch_bytes);

  // Copy batch_global_num to CPU
  size_t pinned_buffer_bytes = GetPinnedBufferSize(batch_size);
  auto pinned_buffer = AllocateBufferOnCPUPinned<void>(pinned_buffer_bytes);
  int* batch_global_num_pinned = reinterpret_cast<int*>(pinned_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(batch_global_num_pinned, batch_global_num_buffer.get(), batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();

  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  size_t qkv_size = batch_size * sequence_length * 3 * hidden_size * element_size;
  auto gemm_buffer = GetScratchBuffer<T>(qkv_size);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  auto& device_prop = GetDeviceProp();
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
      GetConstOnes<CudaT>(m), 1,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Wait for async copy of batch_global_num
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));

  // Find the maximum number of global tokens in all batches
  int max_num_global = 0;
  for (int i = 0; i < batch_size; ++i) {
    if (max_num_global < batch_global_num_pinned[i]) {
      max_num_global = batch_global_num_pinned[i];
    }
  }

  // Force to use fast kernel in two situations:
  // (1) global tokens > windows size. In that case, compact memory kernel cannot be used.
  // (2) sequence_length == 2 * attention_window. Use fast kernel to walk around parity issue of compact memory kernel.
  // In other case, we will choose according to user's environment variable setting (default is fast kernel).
  bool use_fast_kernel = (max_num_global > window_ || sequence_length == 2 * window_ || !use_compact_memory_);

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run glboal GEMM.
  auto global_gemm_buffer = GetScratchBuffer<T>(max_num_global > 0 ? qkv_size : 0);

  if (max_num_global > 0) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
        reinterpret_cast<const CudaT*>(global_bias->template Data<T>()), n,
        GetConstOnes<CudaT>(m), 1,
        &zero, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));

    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(global_weights->template Data<T>()), n,
        reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
        &one, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, max_num_global, window_, use_fast_kernel);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchLongformerAttentionKernel(
          device_prop,
          cublas,
          stream,
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          reinterpret_cast<const CudaT*>(mask->template Data<T>()),
          reinterpret_cast<const CudaT*>(global_gemm_buffer.get()),
          global_attention->template Data<int>(),
          global_index_buffer.get(),
          batch_global_num_buffer.get(),
          pinned_buffer.get(),
          workspace_buffer.get(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          window_,
          max_num_global,
          element_size,
          use_fast_kernel)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
