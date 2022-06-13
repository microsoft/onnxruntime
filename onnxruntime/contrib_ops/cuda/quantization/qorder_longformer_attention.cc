// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wunused-variable"

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/transpose.h"
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
  use_qdq_and_fp16_compute_ = ParseEnvironmentVariableWithDefault<bool>("USE_QDQ_AND_FP16_COMPUTE", false);
  const cublasLtOrder_t InputOrders[2] = {CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL32};
  const cublasLtOrder_t weight_tiles_for_input_col32[2] = {CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4};
  const cublasLtOrder_t weight_tiles_for_input_row[1] = {CUBLASLT_ORDER_COL};
  int num_allowed_weight_orders = 2;
  const cublasLtOrder_t* allowed_weight_orders = weight_tiles_for_input_col32;

  order_input_ = GetCublasLtOrderAttr(info, "order_input", 2, InputOrders,
                                      "QOrderedLongformerAttention: Only ORDER_ROW or ORDER_COL32 is supported for order_input");
  if (order_input_ == CUBLASLT_ORDER_ROW) {
    num_allowed_weight_orders = 1;
    allowed_weight_orders = weight_tiles_for_input_row;
  }
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight", num_allowed_weight_orders, allowed_weight_orders,
                                       "QOrderedLongformerAttention: un-supported order for order_weght");
  order_global_weight_ = GetCublasLtOrderAttr(info, "order_global_weight", num_allowed_weight_orders, allowed_weight_orders,
                                              "QOrderedLongformerAttention: un-supported order for order_global_weight");
  order_output_ = GetCublasLtOrderAttr(info, "order_output", 1, (const cublasLtOrder_t*)&order_input_,
                                       "QOrderedLongformerAttention: oder_output must be same as order_input");
}

static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape[left_num_dims - 2] / left_shape[0];
  }
  int64_t left_k = transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis] : right_shape[right_num_dims - 1];
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}


Status
QOrderedLongformerAttention::ComputeInternal(OpKernelContext* context) const {
  // For Debugging...
  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

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
  auto gemm_buffer = GetScratchBuffer<int8_t>(qkv_3);
  auto gemm_buffer_2 = GetScratchBuffer<int8_t>(qkv_size);

  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;

  const float* scale_input = context->Input<Tensor>(1)->Data<float>();
  const float* scale_weight = context->Input<Tensor>(3)->Data<float>();
  const float* scale_qkvgemm = context->Input<Tensor>(6)->Data<float>();
  const float* scale_global_weight = context->Input<Tensor>(9)->Data<float>();
  const float* scale_global_qkvgemm = context->Input<Tensor>(11)->Data<float>();
  const float* scale_output = context->Input<Tensor>(13)->Data<float>();
  float alpha = (*scale_input * *scale_weight) / *scale_qkvgemm;

  auto& device_prop = GetDeviceProp();

  // Approach-1 - Do the quantized MatMul
   //if (!use_qdq_and_fp16_compute_) {
      // TODO: bias need pre-processing, i.e., / *scale_qkvgemm
      ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                          batch_size, sequence_length, n, k,
                                          &alpha, input->Data<int8_t>(), weights->Data<int8_t>(),
                                          bias->Data<float>(), gemm_buffer.get() + qkv_size,
                                          (cublasLtOrder_t)order_weight_));


      QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, device_prop, gemm_buffer.get() + qkv_size, (CudaT*)gemm_buffer.get(), *scale_qkvgemm, batch_size, sequence_length, n);
  //}

  // Approach-2 - de-quantize and then do cuBlas fp16 Gemm
  // else {
      auto buffer_A = GetScratchBuffer<MLFloat16>(m * k);
      auto buffer_B = GetScratchBuffer<MLFloat16>(k * n);


      QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, device_prop, input->Data<int8_t>(), (CudaT*)buffer_A.get(), *scale_input, batch_size, sequence_length, n);
  
      AllocatorPtr alloc;
      auto status = context->GetTempSpaceAllocator(&alloc);

      auto t = Tensor::Create(DataTypeImpl::GetType<int8_t>(), 
                               TensorShape({n, k}), (void*)weights->Data<int8_t>(), alloc->Info(), 0);
      auto weights_transposed = GetScratchBuffer<int8_t>(k * n);
      auto t_transposed = Tensor::Create(DataTypeImpl::GetType<int8_t>(), 
                                          TensorShape({k, n}), weights_transposed.get(), alloc->Info(), 0);
 
      std::vector<size_t> permutation = {1, 0};

      ORT_RETURN_IF_ERROR(onnxruntime::cuda::Transpose::DoTranspose(device_prop,
                                Stream(),
                                CublasHandle(),
                                permutation,
                                *t, *t_transposed, 
                                nullptr));

      QOrderDequantizeToRow(CUBLASLT_ORDER_ROW, stream, device_prop, t_transposed->Data<int8_t>(), (CudaT*)buffer_B.get(), *scale_weight, 1, k, n);

 

      const CudaT alpha_half = ToCudaType<CudaT>::FromFloat(1.0f);
      const CudaT zero_half = ToCudaType<CudaT>::FromFloat(0.0f);

      int64_t stride_A, stride_B, stride_C, batch_count;

      const int lda = k;
      const int ldb = n;
      const int ldc = n;

      if (CanUseStridedBatchedGemm(input->Shape(), t_transposed->Shape(),
                                          false, false, false, false, stride_A, stride_B, stride_C, batch_count)) {
        CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(CublasHandle(),
                                                              CUBLAS_OP_N,
                                                              CUBLAS_OP_N,
                                                              static_cast<int>(n),
                                                              static_cast<int>(m / batch_size),
                                                              static_cast<int>(k),
                                                              &alpha_half,
                                                              reinterpret_cast<const CudaT*>(buffer_B.get()),
                                                              ldb,
                                                              stride_B,
                                                              reinterpret_cast<const CudaT*>(buffer_A.get()),
                                                              lda,
                                                              stride_A,
                                                              &zero_half,
                                                              reinterpret_cast<CudaT*>(gemm_buffer_2.get()),
                                                              ldc,
                                                              stride_C,
                                                              static_cast<int>(batch_count),
                                                              device_prop));

      }
      else {
          ORT_THROW("No can go");
      }

     // De-quantize and re-quantize the gemm_buffer_2
     auto gemm_buffer_2_quantized = GetScratchBuffer<int8_t>(m * n);

     QOrderQuantizeRowTo(CUBLASLT_ORDER_ROW, stream, device_prop, (CudaT*)gemm_buffer_2.get(), gemm_buffer_2_quantized.get(), *scale_qkvgemm, batch_size, m, n);
     QOrderDequantizeToRow(CUBLASLT_ORDER_ROW, stream, device_prop, gemm_buffer_2_quantized.get(), (CudaT*)gemm_buffer_2.get(), *scale_qkvgemm, batch_size, m, n);
  //}


  // Check the results of the MatMul
  {
      cudaStreamSynchronize(Stream());

      CudaT* matmul_1 = reinterpret_cast<CudaT*>(gemm_buffer.get());
      CudaT* matmul_2 = reinterpret_cast<CudaT*>(gemm_buffer_2.get());

      half* matmul_1_host = (half*)malloc(sizeof(half) * m * n);
      half* matmul_2_host = (half*)malloc(sizeof(half) * m * n);

      cudaMemcpy(matmul_1_host, matmul_1, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(matmul_2_host, matmul_2, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

      float max_diff = 0;

      for (int i = 0; i < m * n; ++i) {
          float f1 = __half2float(matmul_1_host[i]);

          float f2 = __half2float(matmul_2_host[i]);
          // quantize this matmul value
      
          //float quantized_matmul_2_res = f2 / *scale_qkvgemm;
          //float quantized_matmul_2_res_clamped = std::min(std::max(quantized_matmul_2_res, -128.f), 127.f);
          //int8_t quantized_matmul_2_res_clamped_quant = static_cast<int8_t>(std::round(quantized_matmul_2_res_clamped));

          //f2 = quantized_matmul_2_res_clamped_quant * *scale_qkvgemm;

          if (std::abs(f1 - f2) > max_diff) {
              max_diff = std::abs(f1 - f2);
          }
      }

      free(matmul_1_host);
      free(matmul_2_host);
  }

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

    // TODO: bias need pre-processing, i.e., / *scale_qkvgemm
    float global_alpha = (*scale_input * *scale_global_weight) / *scale_global_qkvgemm;
    ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                        batch_size, sequence_length, n, k,
                                        &global_alpha, input->Data<int8_t>(), global_weights->Data<int8_t>(),
                                        global_bias->Data<float>(), global_gemm_buffer.get() + qkv_size,
                                        (cublasLtOrder_t)order_global_weight_));
    QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, device_prop,
                          global_gemm_buffer.get() + qkv_size, (CudaT*)global_gemm_buffer.get(),
                          *scale_global_qkvgemm, batch_size, sequence_length, n);
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, max_num_global, window_, false);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize + output_elements * element_size);
  MLFloat16* out_fp16 = (MLFloat16*)(((int8_t*)workspace_buffer.get()) + workSpaceSize);
  if (!LaunchLongformerAttentionKernel(
          device_prop,
          cublas,
          stream,
          !use_qdq_and_fp16_compute_ ? reinterpret_cast<const CudaT*>(gemm_buffer.get()) 
                                    : reinterpret_cast<const CudaT*>(gemm_buffer_2.get()),
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

  QOrderQuantizeRowTo((cublasLtOrder_t)order_input_, stream, device_prop,
                      (const CudaT*)out_fp16, output->template MutableData<int8_t>(),
                      *scale_output, batch_size, sequence_length, hidden_size);

   
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release());

  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

//#pragma GCC diagnostic pop
