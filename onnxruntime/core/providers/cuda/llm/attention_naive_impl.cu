/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

#include "core/providers/cuda/llm/attention_naive_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include <cuda_fp16.h>

using namespace onnxruntime::cuda;
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace cuda {

Status GemmMatMul(
    cudaStream_t stream, bool has_bias, bool has_scales,
    int32_t dtype_A, int32_t dtype_B,
    int32_t dtype_C, int32_t dtype_Y,
    bool trans_A, bool trans_B, const void* p_input_a, const void* p_input_b,
    const void* p_input_c, const void* p_scale_a, const void* p_scale_b,
    const void* p_scale_y, void* p_output_y, int M, int N, int K, int lda,
    int ldb, int ldd, bool row_major_compute, int64_t sm_count, int iepilogue,
    float alpha, float beta) {
  // TODO: Synchronization should be moved outside of this function.
  // TODO: The function should be split in two parts: create descriptors and run cublasLtMatmul.
  cublasLtEpilogue_t epilogue = static_cast<cublasLtEpilogue_t>(iepilogue);
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  cublasLtHandle_t cublasLt;
  CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&cublasLt));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_A);
  cudaDataType_t b_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_B);
  cudaDataType_t d_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_Y);
  cudaDataType_t scale_cuda_type =
      onnxruntime::cuda::ToCudaDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  cudaDataType_t bias_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_C);

  cublasComputeType_t compute_type;
  switch (d_cuda_type) {
    case CUDA_R_16F:
      switch (a_cuda_type) {
#if !defined(DISABLE_FLOAT8_TYPES)
#if CUDA_VERSION < 11080
#error CUDA_R_8F_E4M3 (float 8 types) is defined with CUDA>=11.8. Set flag DISABLE_FLOAT8_TYPES.
#endif
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
          compute_type = CUBLAS_COMPUTE_32F;
          break;
#endif
        default:
          compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
          break;
      }
      break;
    case CUDA_R_16BF:
      compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
      break;
    case CUDA_R_32F:
      compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
      break;
    default:
      ORT_THROW("Unable to determine computeType in operator GemmFloat8.");
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Adesc, a_cuda_type, trans_A ? K : M, trans_A ? M : K, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Bdesc, b_cuda_type, trans_B ? N : K, trans_B ? K : N, ldb));
  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Ddesc, d_cuda_type, M, N, ldd));

  if (row_major_compute) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
  }

  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, compute_type, scale_cuda_type));
  cublasOperation_t ctransa = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t ctransb = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &ctransa, sizeof(ctransa)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &ctransb, sizeof(ctransb)));

#if CUDA_VERSION >= 11060
  // CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET exists from https://docs.nvidia.com/cuda/archive/11.6.0/pdf/CUBLAS_Library.pdf
  if (sm_count != 0) {
    int math_sm_count = static_cast<int>(sm_count);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count,
        sizeof(math_sm_count)));
  }
#endif

  if (has_scales) {
    // gemm float 8
#if CUDA_VERSION >= 11080
    // CUBLASLT_MATMUL_DESC_FAST_ACCUM, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
    // CUBLASLT_MATMUL_DESC_D_SCALE_POINTER exist from https://docs.nvidia.com/cuda/archive/11.8.0/pdf/CUBLAS_Library.pdf
    const int8_t ifast_accumulation_mode = 1;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc,
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &ifast_accumulation_mode, sizeof(ifast_accumulation_mode)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p_scale_a,
        sizeof(p_scale_a)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p_scale_b,
        sizeof(p_scale_b)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &p_scale_y,
        sizeof(p_scale_b)));
#endif

    // float 8
#if !defined(DISABLE_FLOAT8_TYPES)
    if (dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
        dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2) {
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      CUBLAS_RETURN_IF_ERROR(
          cublasLtMatrixLayoutCreate(&Cdesc, bias_cuda_type, M, N, ldd));
      CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_cuda_type,
          sizeof(bias_cuda_type)));
    } else {
      CUBLAS_RETURN_IF_ERROR(
          cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
    }
#else
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
#endif
  } else {
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
  }

  if (row_major_compute) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
  }

  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

  // See
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true
  // with H100).
  size_t workspaceSize = static_cast<size_t>(1 << 25);  // suggested fixed value 32Mb
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference,
                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspaceSize, sizeof(workspaceSize));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(
      cublasLt, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults);
  ORT_ENFORCE(
      returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to find any suitable algorithm due to ",
      onnxruntime::cuda::cublasGetErrorEnum(cuda_status),
      ", returnedResults=", returnedResults,
      ", alpha=", alpha, ", beta=", beta,
      ", A_type=", onnxruntime::cuda::CudaDataTypeToString(a_cuda_type),
      ", B_type=", onnxruntime::cuda::CudaDataTypeToString(b_cuda_type),
      ", C_type=", onnxruntime::cuda::CudaDataTypeToString(bias_cuda_type),
      ", result_type=", onnxruntime::cuda::CudaDataTypeToString(d_cuda_type),
      ", bias_type=", onnxruntime::cuda::CudaDataTypeToString(bias_cuda_type),
      ", scale_type=", onnxruntime::cuda::CudaDataTypeToString(scale_cuda_type),
      ", computeType=", onnxruntime::cuda::CublasComputeTypeToString(compute_type),
      ", epilogue=", epilogue, ", smCount=", sm_count, ", transA=", trans_A,
      ", transB=", trans_B,
      ", fastAccumulationMode=", 1,
      ", M=", M, ", N=", N, ", K=", K,
      ", lda=", lda, ", ldb=", ldb, ", ldd=", ldd,
      ", workspaceSize=", workspaceSize, ", rowMajorCompute=", (row_major_compute ? 1 : 0),
      ". Check NVIDIA documentation to see what combination is valid: ",
      "https://docs.nvidia.com/cuda/cublas/"
      "index.html?highlight=cublasLtMatmulAlgoGetHeuristic#"
      "cublasltmatmulalgogetheuristic. CUDA>=11.8 is required to use float 8 types.");

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    CUDA_RETURN_IF_ERROR(cudaMalloc(reinterpret_cast<void**>(&workspace), workspaceSize));
  }
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  const void* bias = has_bias ? p_input_c : p_output_y;
  cuda_status = cublasLtMatmul(
      cublasLt, operationDesc, static_cast<const void*>(&alpha), /* alpha */
      p_input_a,                                                 /* A */
      Adesc, p_input_b,                                          /* B */
      Bdesc, static_cast<const void*>(&beta),                    /* beta */
      bias,                                                      /* C */
      Cdesc, p_output_y,                                         /* Y */
      Ddesc, &heuristicResult.algo,                              /* algo */
      workspace,                                                 /* workspace */
      workspaceSize, stream);                                    /* stream */
  ORT_ENFORCE(
      cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to run cublasLtMatmul due to ",
      onnxruntime::cuda::cublasGetErrorEnum(cuda_status),
      ", returnedResults=", returnedResults, ", alpha=", alpha,
      ", A_type=", onnxruntime::cuda::CudaDataTypeToString(a_cuda_type),
      ", B_type=", onnxruntime::cuda::CudaDataTypeToString(b_cuda_type),
      ", result_type=", onnxruntime::cuda::CudaDataTypeToString(d_cuda_type),
      ", bias_type=", onnxruntime::cuda::CudaDataTypeToString(bias_cuda_type),
      ", scale_type=", onnxruntime::cuda::CudaDataTypeToString(scale_cuda_type),
      ", computeType=", onnxruntime::cuda::CublasComputeTypeToString(compute_type),
      ", epilogue=", epilogue, ", smCount=", sm_count, ", transA=", trans_A,
      ", transB=", trans_B,
      ", fastAccumulationMode=", 1,
      " M=", M, " N=", N, ", K=", K, ", lda=", lda, ", ldb=",
      ldb, ", ldd=", ldd, ", workspaceSize=", workspaceSize,
      ", rowMajorCompute=", (row_major_compute ? 1 : 0),
      ". CUDA>=11.8 is required to use float 8 types.");

  if (workspaceSize > 0) {
    CUDA_RETURN_IF_ERROR(cudaFree(workspace));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Ddesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtDestroy(cublasLt));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
