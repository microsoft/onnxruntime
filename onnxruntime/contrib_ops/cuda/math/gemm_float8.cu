// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm_float8.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

static const char* cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "<unknown>";
  }
}

static const char* CudaDataTypeToString(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_16F:
      return "CUDA_R_16F";
    case CUDA_R_16BF:
      return "CUDA_R_16BF";
    case CUDA_R_32F:
      return "CUDA_R_32F";
    case CUDA_R_8F_E4M3:
      return "CUDA_R_8F_E4M3";
    case CUDA_R_8F_E5M2:
      return "CUDA_R_8F_E5M2";
    default:
      return "<unknown>";
  }
}

static const char* CublasComputeTypeToString(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_16F:
      return "CUBLAS_COMPUTE_16F";
    case CUBLAS_COMPUTE_32F:
      return "CUBLAS_COMPUTE_32F";
    case CUBLAS_COMPUTE_32F_FAST_16F:
      return "CUBLAS_COMPUTE_32F_FAST_16F";
    case CUBLAS_COMPUTE_32F_FAST_16BF:
      return "CUBLAS_COMPUTE_32F_FAST_16BF";
    case CUBLAS_COMPUTE_32F_FAST_TF32:
      return "CUBLAS_COMPUTE_32F_FAST_TF32";
    case CUBLAS_COMPUTE_64F:
      return "CUBLAS_COMPUTE_64F";
    default:
      return "<unknown>";
  }
}

// It must exist somewhere already.
cudaDataType_t ToCudaDataType(int32_t element_type) {
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return CUDA_R_32F;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return CUDA_R_16F;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return CUDA_R_16BF;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      return CUDA_R_8F_E4M3;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      return CUDA_R_8F_E5M2;
#endif
    default:
      ORT_THROW("Unexpected element_type=", element_type, ".");
  }
}

// It must exist somewhere already.
int32_t TypeSize(int32_t element_type) {
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return 4;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return 2;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      return 1;
#endif
    default:
      ORT_THROW("Unexpected element_type=", element_type, ".");
  }
}

void GemmFloat8::set(const TensorShape& a_shape, const TensorShape& b_shape, int& M, int& N, int& K, int& lda, int& ldb, int& ldd) const {
  bool row_major = storage_order_ == CUBLASLT_ORDER_ROW;
  constexpr int ir = 0;
  constexpr int ic = 1 - ir;
  if (transA_ && !transB_) {  // TN
    M = a_shape[ic];
    N = b_shape[ic];
    K = a_shape[ir];
    lda = a_shape[row_major ? ic : ir];
    ldb = b_shape[row_major ? ic : ir];
    ldd = b_shape[row_major ? ic : ir];
  } else if (!transA_ && !transB_) {  // NN
    M = a_shape[ir];
    N = b_shape[ic];
    K = a_shape[ic];
    lda = a_shape[row_major ? ic : ir];
    ldb = b_shape[row_major ? ic : ir];
    ldd = b_shape[row_major ? ic : ir];
  } else if (!transA_ && transB_) {  // NT
    M = a_shape[ir];
    N = b_shape[ir];
    K = a_shape[ic];
    lda = a_shape[row_major ? ic : ir];
    ldb = b_shape[row_major ? ic : ir];
    ldd = b_shape[row_major ? ir : ic];
  } else {  // TT
    M = a_shape[ic];
    N = b_shape[ir];
    K = a_shape[ir];
    lda = a_shape[row_major ? ir : ic];
    ldb = b_shape[row_major ? ir : ic];
    ldd = b_shape[row_major ? ic : ir];
  }
}

Status GemmFloat8::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* A = nullptr;
  const Tensor* B = nullptr;
  const Tensor* scale_A = nullptr;
  const Tensor* scale_B = nullptr;
  const Tensor* scale_Y = nullptr;
  int n_inputs = ctx->InputCount();
  if (n_inputs == 2) {
    A = ctx->Input<Tensor>(0);
    B = ctx->Input<Tensor>(1);
  } else if (n_inputs == 5) {
    A = ctx->Input<Tensor>(0);
    B = ctx->Input<Tensor>(1);
    scale_A = ctx->Input<Tensor>(2);
    scale_B = ctx->Input<Tensor>(3);
    scale_Y = ctx->Input<Tensor>(4);
    ORT_ENFORCE(scale_A->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    ORT_ENFORCE(scale_B->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    ORT_ENFORCE(scale_Y->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  } else {
    ORT_THROW("Unexpected number of inputs, it expects 2 or 5 inputs.");
  }

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();

  ORT_ENFORCE(a_shape.GetDims().size() == 2);
  ORT_ENFORCE(b_shape.GetDims().size() == 2);

  int Md, Nd, Kd;
  auto check = set_check(a_shape, b_shape, Md, Nd, Kd);
  if (!check.IsOK())
    return check;
  int M, N, K, lda, ldb, ldd;
  set(a_shape, b_shape, M, N, K, lda, ldb, ldd);
  ORT_ENFORCE(M == Md);
  ORT_ENFORCE(N == Nd);
  ORT_ENFORCE(K == Kd);

  auto dtype_A = A->GetElementType();
  auto dtype_B = B->GetElementType();

  auto* Y = ctx->Output(0, {M, N});

  cudaStream_t stream = Stream(ctx);
  cublasLtHandle_t cublasLt = CublasLtHandle();

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_cuda_type = ToCudaDataType(dtype_A);
  cudaDataType_t b_cuda_type = ToCudaDataType(dtype_B);
  cudaDataType_t d_cuda_type = ToCudaDataType(dtype_);
  cudaDataType_t bias_cuda_type = ToCudaDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  cudaDataType_t scale_cuda_type = bias_cuda_type;

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, a_cuda_type, transA_ ? K : M, transA_ ? M : K, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, b_cuda_type, transB_ ? N : K, transB_ ? K : N, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, d_cuda_type, M, N, ldd));

  if (storage_order_ == CUBLASLT_ORDER_ROW) {
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &storage_order_, sizeof(storage_order_)));
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &storage_order_, sizeof(storage_order_)));
  }

  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, compute_type_, scale_cuda_type));
  cublasOperation_t transa = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  if (sm_count_ != 0) {
    int math_sm_count = static_cast<int>(sm_count_);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count, sizeof(math_sm_count)));
  }

  const void* p_scale_a = nullptr;
  const void* p_scale_b = nullptr;
  const void* p_scale_y = nullptr;
  if (n_inputs == 5) {
    // gemm float 8
    const int8_t ifast_accumulation_mode = fast_accumulation_mode_ ? 1 : 0;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc,
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &ifast_accumulation_mode, sizeof(ifast_accumulation_mode)));
    p_scale_a = scale_A->DataRaw();
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p_scale_a,
        sizeof(p_scale_a)));
    p_scale_b = scale_B->DataRaw();
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p_scale_b,
        sizeof(p_scale_b)));
    p_scale_y = scale_Y->DataRaw();
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &p_scale_y,
        sizeof(p_scale_b)));

    // float 8
#if !defined(DISABLE_FLOAT8_TYPES)
    if (dtype_ == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN ||
        dtype_ == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2) {
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
  } else {
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
  }
#else
    // An output is still needed but it is not initialized.
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
#endif

  bool row_major = storage_order_ == CUBLASLT_ORDER_ROW;
  if (row_major) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_RETURN_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
  }

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

  // TODO: use the context to store the buffer
  size_t workspaceSize = 1 << 25;  // suggested fixed value 32Mb
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
      cublasGetErrorEnum(cuda_status), ", returnedResults=", returnedResults,
      ", alpha=", alpha_,
      ", n_inputs=", n_inputs, ", A_type=", CudaDataTypeToString(a_cuda_type),
      ", B_type=", CudaDataTypeToString(b_cuda_type),
      ", result_type=", CudaDataTypeToString(d_cuda_type),
      ", bias_type=", CudaDataTypeToString(bias_cuda_type),
      ", scale_type=", CudaDataTypeToString(scale_cuda_type),
      ", computeType=", CublasComputeTypeToString(compute_type_),
      ", epilogue=", epilogue, ", smCount=", sm_count_, ", transA=", transA_,
      ", transB=", transB_,
      ", fastAccumulationMode=", (fast_accumulation_mode_ ? 1 : 0),
      ", a_shape=", a_shape[0], "x", a_shape[1], ", b_shape=", b_shape[0], "x",
      b_shape[1], ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb,
      ", ldd=", ldd, ", workspaceSize=", workspaceSize,
      ", rowMajor=", (row_major ? 1 : 0),
      ". Check NVIDIA documentation to see what combination is valid: ",
      "https://docs.nvidia.com/cuda/cublas/"
      "index.html?highlight=cublasLtMatmulAlgoGetHeuristic#"
      "cublasltmatmulalgogetheuristic.");

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    CUDA_CALL_THROW(cudaMalloc((void**)&workspace, workspaceSize));
  }

  // TODO: This is only part changing. Everything before only depends on the input dimensions, type, and storage.
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  float beta = 0;
  void* C = Y->MutableDataRaw();
  cuda_status = cublasLtMatmul(
      cublasLt, operationDesc, static_cast<const void*>(&alpha_), /* alpha */
      A->DataRaw(),                                               /* A */
      Adesc, B->DataRaw(),                                        /* B */
      Bdesc, static_cast<const void*>(&beta),                     /* beta */
      C,                                                          /* C */
      Cdesc, Y->MutableDataRaw(),                                 /* Y */
      Ddesc, &heuristicResult.algo,                               /* algo */
      workspace,                                                  /* workspace */
      workspaceSize, stream);                                     /* stream */
  ORT_ENFORCE(
      cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to run cublasLtMatmul due to ",
      cublasGetErrorEnum(cuda_status), ", returnedResults=", returnedResults,
      ", alpha=", alpha_,
      ", n_inputs=", n_inputs, ", A_type=", CudaDataTypeToString(a_cuda_type),
      ", B_type=", CudaDataTypeToString(b_cuda_type),
      ", result_type=", CudaDataTypeToString(d_cuda_type),
      ", bias_type=", CudaDataTypeToString(bias_cuda_type),
      ", scale_type=", CudaDataTypeToString(scale_cuda_type),
      ", computeType=", CublasComputeTypeToString(compute_type_),
      ", epilogue=", epilogue, ", smCount=", sm_count_, ", transA=", transA_,
      ", transB=", transB_,
      ", fastAccumulationMode=", (fast_accumulation_mode_ ? 1 : 0),
      ", a_shape=", a_shape[0], "x", a_shape[1], ", b_shape=", b_shape[0], "x",
      b_shape[1], ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb,
      ", ldd=", ldd, ", workspaceSize=", workspaceSize,
      ", rowMajor=", (row_major ? 1 : 0),
      ".");

  if (workspaceSize > 0) {
    CUDA_CALL_THROW(cudaFree(workspace));
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
}  // namespace contrib
}  // namespace onnxruntime
