// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/longformer_global_impl.h"
#include "contrib_ops/cuda/bert/longformer_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "contrib_ops/cuda/bert/longformer_attention.h"

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_longformer_attention.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_matmul_utils.h"

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

QOrderedLongformerAttention::QOrderedLongformerAttention(const OpKernelInfo& info)
    : CudaKernel(info), LongformerAttentionBase(info) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, false);
  const cublasLtOrder_t InputOrders[2] = {CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL32};
  const cublasLtOrder_t weight_tiles_for_input_col32[2] = {CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4};
  const cublasLtOrder_t weight_tiles_for_input_row[1] = {CUBLASLT_ORDER_COL};
  int num_allowed_weight_orders = 2;
  const cublasLtOrder_t* allowed_weight_orders = weight_tiles_for_input_col32;

  order_input_ = GetCublasLtOrderAttr(
      info, "order_input", 2, InputOrders,
      "QOrderedLongformerAttention: Only ORDER_ROW or ORDER_COL32 is supported for order_input");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_ROW, "Currently only support input with ORDER_ROW");

  if (order_input_ == CUBLASLT_ORDER_ROW) {
    num_allowed_weight_orders = 1;
    allowed_weight_orders = weight_tiles_for_input_row;
  }
  order_weight_ = GetCublasLtOrderAttr(
      info, "order_weight", num_allowed_weight_orders, allowed_weight_orders,
      "QOrderedLongformerAttention: un-supported order for order_weght");
  order_global_weight_ = GetCublasLtOrderAttr(
      info, "order_global_weight", num_allowed_weight_orders, allowed_weight_orders,
      "QOrderedLongformerAttention: un-supported order for order_global_weight");
  order_output_ = GetCublasLtOrderAttr(
      info, "order_output", 1, (const cublasLtOrder_t*)&order_input_,
      "QOrderedLongformerAttention: oder_output must be same as order_input");

#else

  ORT_ENFORCE(false, "Compiling with CUDA_VERSION >= 11.4 is needed!");

#endif
}

Status
QOrderedLongformerAttention::ComputeInternal(OpKernelContext* context) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* attention_mask = context->Input<Tensor>(7);
  const Tensor* global_weights = context->Input<Tensor>(8);
  const Tensor* global_bias = context->Input<Tensor>(10);
  const Tensor* global_attention_mask = context->Input<Tensor>(12);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), attention_mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention_mask->Shape()));
  // Input shapes:
  //   input                 : (batch_size, sequence_length, hidden_size)
  //   weights               : (3, hidden_size, hidden_size)
  //   bias                  : (3 * hidden_size)              -- bias for Q, K, V)
  //   attention_mask        : (batch_size, sequence_length)
  //   global_weights        : (3, hidden_size, hidden_size)
  //   global_bias           : (3 * hidden_size)              -- bias for Global_Q, Global_K, Global_V)
  //   global_attention_mask : (batch_size, sequence_length)
  // Output shapes:
  //   output         : (batch_size, sequence_length, hidden_size)

  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  size_t output_elements = (size_t)shape.Size();
  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = GetCublasHandle(context);
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream(context);
  CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas, stream));

  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;
  constexpr size_t element_size = sizeof(MLFloat16);

  // TODO: only calculate once per model.
  // Build Global Index
  auto global_index_buffer = GetScratchBuffer<int>(static_cast<size_t>(batch_size) * static_cast<size_t>(sequence_length), context->GetComputeStream());
  auto batch_global_num_buffer = GetScratchBuffer<int>(batch_size, context->GetComputeStream());

  size_t global_scratch_bytes = GetGlobalScratchSize(sequence_length);
  auto global_scratch_buffer = GetScratchBuffer<void>(global_scratch_bytes, context->GetComputeStream());

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(BuildGlobalIndex(device_prop,
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
                                       cudaMemcpyDeviceToHost, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& is_copy_done = new_event.Get();

  CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&is_copy_done, cudaEventDisableTiming));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(is_copy_done, stream));

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  size_t qkv_count = (size_t)m * (size_t)n;
  size_t qkv_size = batch_size * sequence_length * 3 * hidden_size * element_size;
  // Buffer for GEMM outputs of q, k, v, global_q, global_k and global_v
  // TODO(tianleiwu): compact global_q only need batch_size * window * hidden_size * element_size buffer size.
  size_t qkv_3 = qkv_size + qkv_size + 2 * qkv_count * sizeof(int8_t);
  auto gemm_buffer = GetScratchBuffer<int8_t>(qkv_3, context->GetComputeStream());

  const float* scale_input = context->Input<Tensor>(1)->Data<float>();
  const float* scale_weight = context->Input<Tensor>(3)->Data<float>();
  const float* scale_qkvgemm = context->Input<Tensor>(6)->Data<float>();
  const float* scale_global_weight = context->Input<Tensor>(9)->Data<float>();
  const float* scale_global_qkvgemm = context->Input<Tensor>(11)->Data<float>();
  const float* scale_output = context->Input<Tensor>(13)->Data<float>();
  float alpha = (*scale_input * *scale_weight) / *scale_qkvgemm;

  // Note: bias is already pre-processed i.e., / *scale_qkvgemm
  int8_t* s8_gemm_buffer = ((int8_t*)gemm_buffer.get()) + 2 * qkv_size;
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size, sequence_length, n, k,
                                      &alpha, input->Data<int8_t>(), weights->Data<int8_t>(),
                                      bias->Data<float>(),
                                      s8_gemm_buffer,
                                      (cublasLtOrder_t)order_weight_));

  ORT_RETURN_IF_ERROR(QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, device_prop,
                                            s8_gemm_buffer, (CudaT*)gemm_buffer.get(),
                                            *scale_qkvgemm, batch_size, sequence_length, n));

  // Wait for async copy of batch_global_num
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(is_copy_done));

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
  bool disable_compact_memory = (max_num_global > window_ || sequence_length == 2 * window_ || !use_compact_memory_);

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run glboal GEMM.
  CudaT* global_gemm_buffer = nullptr;
  if (max_num_global > 0) {
    global_gemm_buffer = reinterpret_cast<CudaT*>(reinterpret_cast<char*>(gemm_buffer.get()) + qkv_size);
    int8_t* global_s8_gemm_buffer = ((int8_t*)gemm_buffer.get()) + (2 * qkv_size + qkv_count * sizeof(int8_t));
    float global_alpha = (*scale_input * *scale_global_weight) / *scale_global_qkvgemm;
    ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                        batch_size, sequence_length, n, k,
                                        &global_alpha, input->Data<int8_t>(), global_weights->Data<int8_t>(),
                                        global_bias->Data<float>(),
                                        global_s8_gemm_buffer,
                                        (cublasLtOrder_t)order_global_weight_));
    ORT_RETURN_IF_ERROR(QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, device_prop,
                                              global_s8_gemm_buffer, global_gemm_buffer,
                                              *scale_global_qkvgemm, batch_size, sequence_length, n));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size,
                                                             batch_size,
                                                             num_heads_,
                                                             head_size,
                                                             sequence_length,
                                                             max_num_global,
                                                             window_,
                                                             disable_compact_memory);

  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize + output_elements * element_size, context->GetComputeStream());
  MLFloat16* out_fp16 = (MLFloat16*)(((int8_t*)workspace_buffer.get()) + workSpaceSize);
  ORT_RETURN_IF_ERROR(LaunchLongformerAttentionKernel(device_prop,
                                                      cublas,
                                                      stream,
                                                      reinterpret_cast<const CudaT*>(gemm_buffer.get()),
                                                      nullptr,
                                                      reinterpret_cast<const CudaT*>(attention_mask->Data<MLFloat16>()),
                                                      reinterpret_cast<const CudaT*>(global_gemm_buffer),
                                                      nullptr,
                                                      global_attention_mask->Data<int>(),
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
                                                      disable_compact_memory,
                                                      true,     // use_merged_qkv_weights
                                                      false));  // use_half4

  ORT_RETURN_IF_ERROR(QOrderQuantizeRowTo((cublasLtOrder_t)order_input_, stream, device_prop,
                                          (const CudaT*)out_fp16, output->template MutableData<int8_t>(),
                                          *scale_output, batch_size, sequence_length, hidden_size));

  // Defer release of pinned memory since cudaStreamSynchronize is not used here and kernel need access the buffer.
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release(), context->GetComputeStream());

#else

  ORT_UNUSED_PARAMETER(context);
  ORT_ENFORCE(false, "Compiling with CUDA_VERSION >= 11.4 is needed!");

#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
