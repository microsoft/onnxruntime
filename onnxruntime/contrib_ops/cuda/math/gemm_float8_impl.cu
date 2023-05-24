// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm_float8_impl.cuh"
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

void GemmFloat8_Impl::set(int M, int N, int K, int& lda, int& ldb, int& ldd) const {
  if (trans_A_ && !trans_B_) {  // TN
    lda = K;
    ldb = K;
    ldd = M;
  } else if (!trans_A_ && !trans_B_) {  // NN
    lda = M;
    ldb = K;
    ldd = M;
  } else if (!trans_A_ && trans_B_) {  // NT
    lda = M;
    ldb = N;
    ldd = M;
  } else {  // TT
    ORT_THROW("trans_A_ == true && trans_B_ == true not allowed.");
  }
}

onnxruntime::Status GemmFloat8_Impl::CudaCompute(
    const int32_t* dtypes, cudaStream_t stream, cublasLtHandle_t handle,
    const Tensor* A, const Tensor* B, const Tensor* C, Tensor* D,
    int M, int N, int K) const {
  int lda, ldb, ldd;
  set(M, N, K, lda, ldb, ldd);

  bool has_C = beta_ != 0 && C != nullptr;

  // broadcast bias if needed and is present
  if (has_C) {
    auto& a_shape = A->Shape();
    auto& b_shape = B->Shape();
    auto& c_shape = C->Shape();
    if (c_shape.Size() == 1) {
      // if C is (), (1,) or (1, 1), broadcast the scalar
      ORT_THROW("Broadcasting is not implemented in GemmFloat8.");
    } else if (c_shape.NumDimensions() == 1 || c_shape[0] == 1) {
      // C is (N,) or (1, N), broadcast using Y(N,M) = 1 * C(N,1) x ones(1,M) + 0 * C
      ORT_THROW("Broadcasting is not implemented in GemmFloat8.");
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * C
      ORT_THROW("Broadcasting is not implemented in GemmFloat8.");
    } else {
      // C is (M, N), no broadcast needed.
      /*
      constexpr bool same_type = std::same_type<DType, BiasType>::value;
      if (same_type) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(out_data, b_data, static_cast<size_t>(M) * N * sizeof(T), cudaMemcpyDeviceToDevice, Stream(ctx)));
      }
      */
    }
  }

  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(B) x op(A) + beta * C
  std::cout << "GemmF8-1\n";

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  std::cout << "GemmF8-2\n";

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType atype = ToCudaDataType(dtypes[0]);
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, atype, trans_A_ ? M : K, trans_A_ ? K : M, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, ToCudaDataType(dtypes[1]), trans_B_ ? K : N, trans_B_ ? N : K, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, ToCudaDataType(dtypes[3]), M, N, ldd));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));

  // CUDA_R_32F is the scale type for the time being since it is not used.
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescCreate#cublasltmatmuldesccreate
  cublasLtMatmulDescCreate(&operationDesc, compute_type_, scale_type_);
  cublasOperation_t transa = trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  const int8_t ifast_accumulation_mode = fast_accumulation_mode_ ? 0 : 1;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &ifast_accumulation_mode, sizeof(ifast_accumulation_mode));

  std::cout << "GemmF8-3\n";
  /*
  // TODO add inputs for the scales.
  // No scale for the time being so no need to set.
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER
  */

  if (sm_count_ != 0) {
    int math_sm_count = sm_count_;
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        &math_sm_count, sizeof(math_sm_count));
  }

  std::cout << "GemmF8-4\n";
  if (has_C) {
    std::cout << "GemmF8-5\n";
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, ToCudaDataType(dtypes[2]), M, N, ldd));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  }
  /*
  // No bias for the time being.
  if (relu_bias) {
    std::cout << "GemmF8-6\n";
    cudaDataType bias_type = ToCudaDataType(dtypes[4]);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_type, sizeof(bias_type)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          relu_bias, sizeof(*relu_bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }
  */

  std::cout << "GemmF8-7\n";
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  cublasLtMatmulPreferenceCreate(&preference);

  // See https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true with H100).
  size_t type_size = std::max(std::max(TypeSize(dtypes[0]), TypeSize(dtypes[1])), std::max(std::max(TypeSize(dtypes[2]), TypeSize(dtypes[3])), TypeSize(dtypes[4])));
  size_t workspaceSize = std::max(K * M, K * N) * type_size;  // suggested fixed value 24Mb
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

  std::cout << "GemmF8-8\n";
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                                              Ddesc, preference, 1, &heuristicResult, &returnedResults);
  ORT_ENFORCE(returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
              "Unable to find any suitable algorithm due to ", cublasGetErrorEnum(cuda_status),
              ", preference=", preference, ", returnedResults=", returnedResults,
              ", A_type=", ToCudaDataType(dtypes[0]), ", B_type=", ToCudaDataType(dtypes[1]),
              ", C_type=", ToCudaDataType(dtypes[2]), ", D_type=", ToCudaDataType(dtypes[3]),
              ", bias_type=", ToCudaDataType(dtypes[4]), ", computeType=", compute_type_,
              ", transA=", trans_A_, ", transB=", trans_B_,
              ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb, ", ldd=", ldd,
              ", workspaceSize=", workspaceSize, ". Check NVDIDIA documentation to see what combination is valid: ",
              "https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic.");
  std::cout << "GemmF8-9\n";
  void* workspace = nullptr;
  CUDA_CALL_THROW(cudaMalloc((void**)&workspace, workspaceSize));
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  cublasLtMatmul(handle,
                 operationDesc,
                 static_cast<const void*>(&alpha_), /* alpha */
                 A,                                 /* A */
                 Adesc,
                 B, /* B */
                 Bdesc,
                 static_cast<const void*>(&beta_), /* beta */
                 C,                                /* C */
                 Cdesc,
                 D, /* D */
                 Ddesc,
                 &heuristicResult.algo, /* algo */
                 workspace,             /* workspace */
                 workspaceSize,
                 stream); /* stream */
  std::cout << "GemmF8-10\n";
  cudaFree(workspace);

  std::cout << "GemmF8-11\n";
  cublasLtMatmulPreferenceDestroy(preference);
  if (Cdesc != nullptr && Cdesc != Ddesc)
    cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatmulDescDestroy(operationDesc);
  std::cout << "GemmF8-12\n";
  return onnxruntime::Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
