// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/longformer_attention.h"
#include "contrib_ops/cuda/bert/longformer_global_impl.h"
#include "contrib_ops/cuda/bert/longformer_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "contrib_ops/cuda/bert/transformer_common.h"

#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include "qorder_longformer_attention.h"
#include "qorder_common_impl.h"
#include "qorder_common.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedLongformerAttention,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("G", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // scale_input
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // scale_weight
        .InputMemoryType(OrtMemTypeCPUInput, 5)    // scale_bias
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // scale_qkv_gemm
        .InputMemoryType(OrtMemTypeCPUInput, 9)    // scale_global_weight
        .InputMemoryType(OrtMemTypeCPUInput, 11)   // scale_global_qkvgemm
        .InputMemoryType(OrtMemTypeCPUInput, 13),  // scale_output
    QOrderedLongformerAttention);

QOrderedLongformerAttention::QOrderedLongformerAttention(const OpKernelInfo& info) : CudaKernel(info), LongformerAttentionBase(info) {
  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, false);
  const cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
  const cublasLtOrder_t weight_tiles[2] = {CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4};
  order_input_ = GetCublasLtOrderAttr(info, "order_input", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight", 2, weight_tiles, "Only COL4_4R2_8C and COL32_2R_4R4 are supported for order_weght");
  order_global_weight_ = GetCublasLtOrderAttr(info, "order_global_weight", 2, weight_tiles, "Only COL4_4R2_8C and COL32_2R_4R4 are supported for order_global_weight");
  order_output_ = GetCublasLtOrderAttr(info, "order_output", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_output");
}

Status
QOrderedLongformerAttention::ComputeInternal(OpKernelContext* context) const {
  // For Debugging...
  DUBUG_PERF_CUDA_SYNC();

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* mask = context->Input<Tensor>(7);
  const Tensor* global_weights = context->Input<Tensor>(8);
  const Tensor* global_bias = context->Input<Tensor>(10);
  const Tensor* global_attention = context->Input<Tensor>(12);
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

  size_t output_elements = (size_t)shape.Size();
  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = CublasHandle();
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream();
  CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas, stream));

  constexpr size_t element_size = sizeof(MLFloat16);

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

  size_t qkv_count = (size_t)m * (size_t)n;
  size_t qkv_size = qkv_count * element_size;
  size_t qkv_3 = qkv_size + qkv_count * sizeof(int8_t);
  auto gemm_buffer = GetScratchBuffer<int8_t>(qkv_3 + 3 * element_size);  // extra half scale

  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;
  // CudaT one = ToCudaType<MLFloat16>::FromFloat(1.0f);
  // CudaT zero = ToCudaType<MLFloat16>::FromFloat(0.0f);

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  //   CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
  //       cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
  //       reinterpret_cast<const CudaT*>(bias->template Data<MLFloat16>()), n,
  //       GetConstOnes<CudaT>(m), 1,
  //       &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  //   // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  //   CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
  //       cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
  //       reinterpret_cast<const CudaT*>(weights->template Data<MLFloat16>()), n,
  //       reinterpret_cast<const CudaT*>(input->template Data<MLFloat16>()), k,
  //       &one, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  const float* scale_input = context->Input<Tensor>(1)->Data<float>();
  const float* scale_weight = context->Input<Tensor>(3)->Data<float>();
  const float* scale_qkvgemm = context->Input<Tensor>(6)->Data<float>();
  const float* scale_global_weight = context->Input<Tensor>(9)->Data<float>();
  const float* scale_global_qkvgemm = context->Input<Tensor>(11)->Data<float>();
  const float* scale_output = context->Input<Tensor>(13)->Data<float>();
  float alpha = (*scale_input * *scale_weight) / *scale_qkvgemm;

  // TODO: make it during operator construct
  // CudaT device_scales[3] = {ToCudaType<MLFloat16>::FromFloat(*scale_qkvgemm),
  //                           ToCudaType<MLFloat16>::FromFloat(*scale_output),
  //                           ToCudaType<MLFloat16>::FromFloat(*scale_global_qkvgemm)};
  // CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gemm_buffer.get() + qkv_3, device_scales, 3 * element_size, cudaMemcpyHostToDevice, stream));

  auto& device_prop = GetDeviceProp();

  // TODO: bias need pre-processing, i.e., / *scale_qkvgemm
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size, sequence_length, n, k,
                                      &alpha, input->Data<int8_t>(), weights->Data<int8_t>(),
                                      bias->Data<float>(), gemm_buffer.get() + qkv_size,
                                      (cublasLtOrder_t)order_weight_));

  QOrderDequantizeCol32ToRow(stream, device_prop, gemm_buffer.get() + qkv_size, (CudaT*)gemm_buffer.get(), *scale_qkvgemm, batch_size, sequence_length, n);

  // ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
  //                                     batch_size, sequence_length, n, k,
  //                                     &alpha, input->Data<int8_t>(), weights->Data<int8_t>(),
  //                                     bias->Data<float>(), gemm_buffer.get(),
  //                                     (cublasLtOrder_t)order_weight_));
  // ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, device_prop, batch_size, sequence_length, n, CUDA_R_8I,
  //                             gemm_buffer.get(), (cublasLtOrder_t)order_input_,
  //                             gemm_buffer.get() + qkv_size, CUBLASLT_ORDER_ROW));
  // const CudaT* half_scale = (const CudaT*)(gemm_buffer.get() + qkv_3);
  // ORT_RETURN_IF_ERROR(CudaDequantizeLinear(stream, gemm_buffer.get() + qkv_size,
  //                                          (CudaT*)gemm_buffer.get(), half_scale, (const int8_t*)nullptr, qkv_count));

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
  auto global_gemm_buffer = GetScratchBuffer<int8_t>(max_num_global > 0 ? qkv_3 : 0);

  if (max_num_global > 0) {
    // CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
    //     cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
    //     reinterpret_cast<const CudaT*>(global_bias->template Data<MLFloat16>()), n,
    //     GetConstOnes<CudaT>(m), 1,
    //     &zero, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));

    // CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
    //     cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
    //     reinterpret_cast<const CudaT*>(global_weights->template Data<MLFloat16>()), n,
    //     reinterpret_cast<const CudaT*>(input->template Data<MLFloat16>()), k,
    //     &one, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));

    // TODO: bias need pre-processing, i.e., / *scale_qkvgemm
    float global_alpha = (*scale_input * *scale_global_weight) / *scale_global_qkvgemm;
    ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                        batch_size, sequence_length, n, k,
                                        &global_alpha, input->Data<int8_t>(), global_weights->Data<int8_t>(),
                                        global_bias->Data<float>(), global_gemm_buffer.get() + qkv_size,
                                        (cublasLtOrder_t)order_global_weight_));
    QOrderDequantizeCol32ToRow(stream, device_prop, global_gemm_buffer.get() + qkv_size, (CudaT*)global_gemm_buffer.get(),
                               *scale_global_qkvgemm, batch_size, sequence_length, n);

    // ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
    //                                     batch_size, sequence_length, n, k,
    //                                     &global_alpha, input->Data<int8_t>(), global_weights->Data<int8_t>(),
    //                                     global_bias->Data<float>(), global_gemm_buffer.get(),
    //                                     (cublasLtOrder_t)order_global_weight_));
    // ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, device_prop, batch_size, sequence_length, n, CUDA_R_8I,
    //                             global_gemm_buffer.get(), (cublasLtOrder_t)order_input_,
    //                             global_gemm_buffer.get() + qkv_size, CUBLASLT_ORDER_ROW));

    // const CudaT* half_global_scale = (const CudaT*)(gemm_buffer.get() + qkv_3 + 2*element_size);
    // ORT_RETURN_IF_ERROR(CudaDequantizeLinear(stream, global_gemm_buffer.get() + qkv_size,
    //                                         (CudaT*)global_gemm_buffer.get(), half_global_scale, (const int8_t*)nullptr, qkv_count));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, max_num_global, window_, use_fast_kernel);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize + output_elements * (element_size + sizeof(int8_t)));
  MLFloat16* out_fp16 = (MLFloat16*)(((int8_t*)workspace_buffer.get()) + workSpaceSize);
  if (!LaunchLongformerAttentionKernel(
          device_prop,
          cublas,
          stream,
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          reinterpret_cast<const CudaT*>(mask->template Data<MLFloat16>()),
          reinterpret_cast<const CudaT*>(global_gemm_buffer.get()),
          global_attention->template Data<int>(),
          global_index_buffer.get(),
          batch_global_num_buffer.get(),
          pinned_buffer.get(),
          workspace_buffer.get(),
          out_fp16,
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

  QOrderQuantizeRowToCol32(stream, device_prop, (const CudaT*)out_fp16, output->template MutableData<int8_t>(),
                           *scale_output, batch_size, sequence_length, hidden_size);

  // int8_t* out_tmp_s8 = ((int8_t*)out_fp16) + (output_elements * element_size);
  // const CudaT* out_scale = (const CudaT*)(gemm_buffer.get() + qkv_3 + element_size);
  // ORT_RETURN_IF_ERROR(CudaQuantizeLinear(stream, (CudaT*)out_fp16, out_tmp_s8, out_scale, (const int8_t*)nullptr, output_elements));
  // ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, device_prop, batch_size, sequence_length, hidden_size, CUDA_R_8I,
  //                             out_tmp_s8, CUBLASLT_ORDER_ROW, output->template MutableData<int8_t>(), (cublasLtOrder_t)order_output_));

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release());

  DUBUG_PERF_CUDA_SYNC();
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
