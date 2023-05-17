// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm_float8.cuh"

namespace onnxruntime {
namespace cuda {

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
void GemmFloat8_Impl<AType, BType, CType, DType, BiasType>::CudaCompute() const {
  // broadcast bias if needed and is present
  if (beta_ != 0 && C != nullptr) {
    auto& c_shape = C->Shape();
    const CudaT* b_data = reinterpret_cast<const CudaT*>(C->Data<CType>());
    if (c_shape.Size() == 1) {
      // if C is (), (1,) or (1, 1), broadcast the scalar
      ORT_RAISE("Broadcasting is not implemented in GemmFloat8.");
    } else if (c_shape.NumDimensions() == 1 || c_shape[0] == 1) {
      // C is (N,) or (1, N), broadcast using Y(N,M) = 1 * C(N,1) x ones(1,M) + 0 * C
      ORT_RAISE("Broadcasting is not implemented in GemmFloat8.");
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * C
      ORT_RAISE("Broadcasting is not implemented in GemmFloat8.");
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

  CudaDType alpha = alpha_cast_;
  CudaDType beta = beta_cast_;
  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(B) x op(A) + beta * C

  constexpr auto A_type = ToCudaDataType<AType>();
  constexpr auto B_type = ToCudaDataType<BType>();
  constexpr auto C_type = ToCudaDataType<CType>();
  constexpr auto D_type = ToCudaDataType<DType>();
  constexpr auto bias_type = ToCudaDataType<BiasType>();

  // It should be true all the time unless we extend the definition of the kernel to other combinations.
  /*
  constexpr cublasComputeType_t gemm_compute_type = (A_type == CUDA_R_8F_E5M2 || B_type == CUDA_R_8F_E5M2)
                                          ? CUBLAS_COMPUTE_32F
                                          : CUBLAS_COMPUTE_32F_FAST_TF32;
  */

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create matrix descriptors. Not setting any extra attributes.
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, A_type, trans_A_ ? M : K, trans_A_ ? K : M, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, B_type, trans_B_ ? K : N, trans_B_ ? N : K, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  // CUDA_R_32F is the scale type for the time being since it is not used.
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operationDesc, compute_type_, CUDA_R_32F));
  cublasOperation_t transa = trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  const int8_t fast_accumulation_mode = fast_accumulation_mode_ ? 0 : 1;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(fast_accumulation_mode)));

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
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        &math_sm_count, sizeof(math_sm_count)));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, C_type, M, N, ldd));
  if (bias) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_type, sizeof(bias_type)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias_ptr, sizeof(bias_ptr)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                        CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue, sizeof(epilogue)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));

  // See https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true with H100).
  constexpr size_t type_size = std::max(sizeof(AType), sizof(BType), sizeof(CType), sizeof(DType), sizeof(BiasType));
  size_t workspaceSize = std::max(K * M, K * N) * type_size;  // suggested fixed value 24Mb
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize, sizeof(workspaceSize)));

  int returnedResults = 0;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                                        Ddesc, preference, 1, &heuristicResult,
                                                        &returnedResults));
  ORT_ENFORCE(returnedResults > 0, "Unable to find any suitable algorithm.");
  void* workspace = cudaMalloc(workspaceSize);
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(handle,
                                        operationDesc,
                                        static_cast<const void*>(&alpha_cast_), /* alpha */
                                        A,                                      /* A */
                                        Adesc,
                                        B, /* B */
                                        Bdesc,
                                        static_cast<const void*>(&beta_cast_), /* beta */
                                        C,                                     /* C */
                                        Cdesc,
                                        D, /* D */
                                        Ddesc,
                                        &heuristicResult.algo, /* algo */
                                        workspace,             /* workspace */
                                        workspaceSize,
                                        stream)); /* stream */
  cudaFree(workspace);

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Ddesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
}

}  // namespace cuda
}  // namespace onnxruntime
