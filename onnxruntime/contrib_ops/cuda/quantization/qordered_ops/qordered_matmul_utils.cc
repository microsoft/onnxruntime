// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_matmul_utils.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

static Status cublasLtMatMulInt8SetupAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo,
                                          int algo_id, int swizzle,
                                          int custom_option, int tile, int splitk_val,
                                          int reduction_scheme, int stages) {
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoInit(cublasLt_handle, CUBLAS_COMPUTE_32I, CUDA_R_32F,
                                                CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algo_id, &algo));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                              CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                                              &(custom_option), sizeof(custom_option)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                              CUBLASLT_ALGO_CONFIG_TILE_ID,
                                                              &(tile), sizeof(tile)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                              &(splitk_val), sizeof(splitk_val)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                                              &(swizzle), sizeof(swizzle)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                              &(reduction_scheme), sizeof(int)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                                              &(stages), sizeof(stages)));

  return Status::OK();
}

static inline std::string AlgoKey(const cudaDeviceProp& /*device_prop*/,
                                  int batch_count, int m, int n, int k,
                                  cublasLtOrder_t weight_order, cublasLtOrder_t input_output_order) {
  std::stringstream ss;
  ss << batch_count << "-" << m << "_" << n << "_" << k << "-"
     << static_cast<int>(weight_order) << "-" << static_cast<int>(input_output_order);
  return ss.str();
}

CublasLtMMAlgoMap& CublasLtMMAlgoMap::Instance() {
  static CublasLtMMAlgoMap instance;
  return instance;
}

void CublasLtMMAlgoMap::GetAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo,
                                const cudaDeviceProp& device_prop,
                                int batch_count, int m, int n, int k,
                                cublasLtOrder_t weight_order,
                                cublasLtOrder_t input_output_order) const {
  ORT_ENFORCE(input_output_order == CUBLASLT_ORDER_ROW, "Input/Output should be ORDER_ROW");
  ORT_ENFORCE(weight_order == CUBLASLT_ORDER_COL, "Weight should be ORDER_COL");

  const std::string& key = AlgoKey(device_prop, batch_count, m, n, k, weight_order, input_output_order);

  // TODO: Pre-load the best_algos_ from a config file which will be generated offline (or)
  // Find the best algo dynamically in the warm-up run and cache it in `best_algos_`
  auto algo_it = best_algos_.find(key);
  if (algo_it != best_algos_.end() && algo_it->second.workspace_size == 0) {
    const auto& algo_info = algo_it->second;
    ORT_THROW_IF_ERROR(cublasLtMatMulInt8SetupAlgo(cublasLt_handle, algo, algo_info.algo_id,
                                                   algo_info.swizzle, algo_info.custom_option,
                                                   algo_info.tile, algo_info.splitk_val,
                                                   algo_info.reduction_scheme, algo_info.stages));
  } else {
    // Default algo
    int algo_id = 21;
    int stages = 0;
    ORT_THROW_IF_ERROR(cublasLtMatMulInt8SetupAlgo(cublasLt_handle, algo, algo_id, 0, 0, 20, 0, 0, stages));
  }
}

static Status CreateLtMatrixLayout(cublasLtMatrixLayout_t& layout_desc,
                                   int const batch_count, int64_t const rows_after_op, int64_t const cols_after_op,
                                   cudaDataType_t const data_type, cublasLtOrder_t const mat_order,
                                   cublasOperation_t const mat_trans) {
  if (mat_trans == CUBLAS_OP_T) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&layout_desc, data_type, cols_after_op, rows_after_op,
                                                      CalcLeadingDimensionLt(cols_after_op, rows_after_op, mat_order)));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&layout_desc, data_type, rows_after_op, cols_after_op,
                                                      CalcLeadingDimensionLt(rows_after_op, cols_after_op, mat_order)));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layout_desc,
                                                          CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                          &mat_order, sizeof(mat_order)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layout_desc,
                                                          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count, sizeof(batch_count)));

  if (batch_count > 1) {
    int64_t batch_stride = rows_after_op * cols_after_op;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                            &batch_stride, sizeof(batch_stride)));
  }

  return Status::OK();
}

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                       const cudaDeviceProp& device_prop,
                       int32_t batch_count, int64_t m, int64_t n, int64_t k,
                       const float* alpha, const int8_t* A, const int8_t* B, int32_t batch_B,
                       const float* bias,
                       const float* beta, const int8_t* C, int32_t batch_C,
                       int8_t* D,
                       cublasLtOrder_t weight_order,
                       cublasLtPointerMode_t pointer_mode) {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11040
  ORT_RETURN_IF(pointer_mode > CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO,
                "Need CUDA 11.4.2 to support CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST")
#endif

  ORT_RETURN_IF(weight_order != CUBLASLT_ORDER_COL, "Weight should be ORDER_COL");

  const cublasOperation_t transpose_op = CUBLAS_OP_T;

  cublasLtMatmulDesc_t matmul_desc = nullptr;
  auto clean_matmul_desc = gsl::finally([&matmul_desc]() {if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc); });

  cublasLtMatrixLayout_t desc_A = nullptr;
  auto clean_desc_A = gsl::finally([&desc_A]() {if (desc_A) cublasLtMatrixLayoutDestroy(desc_A); });

  cublasLtMatrixLayout_t desc_B = nullptr;
  auto clean_desc_B = gsl::finally([&desc_B]() {if (desc_B) cublasLtMatrixLayoutDestroy(desc_B); });

  cublasLtMatrixLayout_t desc_C = nullptr;
  auto clean_desc_C = gsl::finally([&desc_C]() {if (desc_C) cublasLtMatrixLayoutDestroy(desc_C); });

  cublasLtMatrixLayout_t desc_D = nullptr;
  auto clean_desc_D = gsl::finally([&desc_D]() {if (desc_D) cublasLtMatrixLayoutDestroy(desc_D); });

  constexpr float beta_zero = 0.0f;
  beta = (C == nullptr ? &beta_zero : beta);

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32F));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                        CUBLASLT_MATMUL_DESC_TRANSA,
                                                        &transpose_op, sizeof(transpose_op)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                        CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                        &pointer_mode, sizeof(pointer_mode)));

  if (bias != nullptr) {
    cublasLtEpilogue_t epilogue_bias = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                          &epilogue_bias, sizeof(epilogue_bias)));

    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          &bias, sizeof(bias)));
  }

  ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_A, batch_count, k, m, CUDA_R_8I,
                                           CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // for A'

  ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_B, batch_B, k, n, CUDA_R_8I,
                                           CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // For B

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batch_count, sizeof(batch_count)));

  ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_D, batch_count, n, m,
                                           CUDA_R_8I, CUBLASLT_ORDER_COL,
                                           CUBLAS_OP_N));  // For D'

  if (C != nullptr) {
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_C, batch_C, n, m, CUDA_R_8I,
                                             CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // For C'

    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                            &batch_count, sizeof(batch_count)));
  }

  cublasLtMatmulAlgo_t algo;
  CublasLtMMAlgoMap::Instance().GetAlgo(cublasLt_handle, algo, device_prop, batch_count,
                                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                        weight_order, CUBLASLT_ORDER_ROW);

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc, alpha,
                                        B, desc_B,
                                        A, desc_A,
                                        beta,
                                        C == nullptr ? D : C, C == nullptr ? desc_D : desc_C,
                                        D, desc_D,
                                        &algo, nullptr, 0,  // algo, workspace, workspace_size
                                        stream));
  return Status::OK();
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
