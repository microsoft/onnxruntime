// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_stream_handle.h"
#endif

#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"
#include "core/common/common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

using onnxruntime::contrib::rocm::blas::GemmFastGeluParams;

#ifdef USE_HIPBLASLT

// For large K and small M/N, K dim will be split to multiple workgroups and buffers,
// which will require additional workspace. Here we set the max workspace size to 32MB.
constexpr const size_t kHipBlasLtMaxWorkSpaceSizeInBytes = 32 * 1024 * 1024;

enum ActivationType {
  NONE = 0,
  RELU = 1,
  GELU = 2,
};

template <typename T>
constexpr hipblasltDatatype_t HipBlasDataTypeFor();

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<float>() {
  return HIPBLASLT_R_32F;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<half>() {
  return HIPBLASLT_R_16F;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<BFloat16>() {
  return HIPBLASLT_R_16B;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<double>() {
  return HIPBLASLT_R_64F;
}

template <typename Layout>
constexpr hipblasOperation_t MapCKLayoutToHipBlasLt() {
  if constexpr (std::is_same_v<Layout, Row>) {
    return HIPBLAS_OP_N;
  }
  return HIPBLAS_OP_T;
}

template <typename T, typename ParamsT>
int GetBatchCountFromParams(const ParamsT* params) {
  ORT_UNUSED_PARAMETER(params);
  return 1;
}

template <typename T>
int GetBatchCountFromParams(const StridedBatchedGemmParams<T>* params) {
  return params->batch;
}

template <typename T, typename ParamsT>
const T* GetBiasFromParams(const ParamsT* params) {
  ORT_UNUSED_PARAMETER(params);
  return nullptr;
}

template <typename T>
const T* GetBiasFromParams(const GemmFastGeluParams<T>* params) {
  return params->bias;
}

template <typename T, typename ParamsT>
std::string TypeStringFor() {
  if constexpr (std::is_same_v<ParamsT, GemmParams<T>>) {
    return "Gemm";
  } else if constexpr (std::is_same_v<ParamsT, StridedBatchedGemmParams<T>>) {
    return "StridedBatchedGemm";
  } else if constexpr (std::is_same_v<ParamsT, GemmFastGeluParams<T>>) {
    return "GemmFastGelu";
  }
  return "UnknownType";
}

template <typename T, typename ALayout, typename BLayout, typename ParamsT>
auto GetHipBlasLtTypeStringAndOps(ActivationType activation_type = ActivationType::NONE) {
  hipblasLtHandle_t handle;
  HIPBLASLT_CALL_THROW(hipblasLtCreate(&handle));

  hipblasOperation_t trans_a = MapCKLayoutToHipBlasLt<BLayout>();
  hipblasOperation_t trans_b = MapCKLayoutToHipBlasLt<ALayout>();
  hipblasltDatatype_t in_out_datatype = HipBlasDataTypeFor<T>();
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  HIPBLASLT_CALL_THROW(hipblaslt_ext::getAllAlgos(handle,
                                                  hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                  trans_a,
                                                  trans_b,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  HIPBLASLT_COMPUTE_F32,
                                                  heuristic_result));
  HIPBLASLT_CALL_THROW(hipblasLtDestroy(handle));

  // Sort heuristic_result by algo index to make sure the order of returned algos is deterministic.
  std::sort(heuristic_result.begin(),
            heuristic_result.end(),
            [](hipblasLtMatmulHeuristicResult_t& a, hipblasLtMatmulHeuristicResult_t& b) {
              return hipblaslt_ext::getIndexFromAlgo(a.algo) < hipblaslt_ext::getIndexFromAlgo(b.algo);
            });

  int returned_algo_count = heuristic_result.size();
  std::vector<std::pair<std::string, Op<ParamsT>>> ret;
  for (int i = 0; i < returned_algo_count; i++) {
    hipblasLtMatmulAlgo_t algo = heuristic_result[i].algo;
    int algo_index = hipblaslt_ext::getIndexFromAlgo(algo);
    auto hipblaslt_gemm_op = [=](const ParamsT* params) -> Status {
      hipblasLtHandle_t op_handle;
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtCreate(&op_handle));

      // Note: properties of original matrices A and B are swapped.
      int64_t lda = (params->opb == BlasOp::N) ? params->n : params->k;
      int64_t ldb = (params->opa == BlasOp::N) ? params->k : params->m;
      int64_t ldc = params->n;
      int64_t stride_a = (params->opb == BlasOp::N) ? lda * params->k : lda * params->n;
      int64_t stride_b = (params->opa == BlasOp::N) ? ldb * params->m : ldb * params->k;
      int64_t stride_c = ldc * params->m;
      float alpha = static_cast<float>(params->alpha);
      float beta = static_cast<float>(params->beta);
      int row_a, col_a, row_b, col_b, row_c, col_c;
      row_a = lda;
      col_a = (params->opb == BlasOp::N) ? params->k : params->n;
      row_b = ldb;
      col_b = (params->opa == BlasOp::N) ? params->m : params->k;
      row_c = ldc;
      col_c = params->m;

      hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
      hipblasLtMatmulDesc_t matmul;
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_a, in_out_datatype, row_a, col_a, lda));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_b, in_out_datatype, row_b, col_b, ldb));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(&mat_c, in_out_datatype, row_c, col_c, ldc));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F));

      int batch = GetBatchCountFromParams<T>(params);
      if (batch > 1) {
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutSetAttribute(
            mat_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
      }

      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
          matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
          matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

      // Deduce enable_bias from params
      auto d_bias = GetBiasFromParams<T>(params);
      bool enable_bias = d_bias != nullptr;

      hipblasLtEpilogue_t epilogue;
      switch (activation_type) {
        case ActivationType::NONE:
          epilogue = enable_bias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
          break;
        case ActivationType::RELU:
          epilogue = enable_bias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
          break;
        case ActivationType::GELU:
          epilogue = enable_bias ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU;
          break;
        default:
          throw std::runtime_error("Unsupported activation type for HipBlasLtMatMul");
      }
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
          matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

      if (enable_bias) {
        HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)));
      }

      size_t workspace_size = 0;
      hipblasLtMatmulAlgo_t algo_i = algo;
      auto status = hipblaslt_ext::matmulIsAlgoSupported(op_handle,
                                                         matmul,
                                                         &alpha,
                                                         mat_a,
                                                         mat_b,
                                                         &beta,
                                                         mat_c,
                                                         mat_c,
                                                         algo_i,
                                                         workspace_size);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != HIPBLAS_STATUS_SUCCESS,
          "[hipBLASLt] Solution #", i, " failed: algo ", algo_index, " not supported");

      IAllocatorUniquePtr<void> workspace_buffer;
      if (workspace_size > 0) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(workspace_size > kHipBlasLtMaxWorkSpaceSizeInBytes,
                                                  "Workspace size exceeds limit (32M): ", workspace_size);
        workspace_size = kHipBlasLtMaxWorkSpaceSizeInBytes;
        workspace_buffer = params->tuning_ctx->GetScratchBuffer(workspace_size, params->stream);
      }

      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmul(op_handle,
                                                matmul,
                                                &alpha,
                                                params->b,
                                                mat_a,
                                                params->a,
                                                mat_b,
                                                &beta,
                                                params->c,
                                                mat_c,
                                                params->c,
                                                mat_c,
                                                &algo_i,
                                                workspace_buffer.get(),
                                                workspace_size,
                                                params->StreamHandle()));

      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatmulDescDestroy(matmul));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_a));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_b));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtMatrixLayoutDestroy(mat_c));
      HIPBLASLT_RETURN_IF_ERROR(hipblasLtDestroy(op_handle));
      return Status::OK();
    };
    std::string type_string = onnxruntime::MakeString(
        TypeStringFor<T, ParamsT>(), "HipBlasLt_", i, "_algo_", algo_index);
    ret.emplace_back(type_string, std::move(hipblaslt_gemm_op));
  }
  return ret;
}

template <typename T, typename ALayout, typename BLayout>
auto GetHipBlasLtGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, ALayout, BLayout, GemmParams<T>>();
}

template <typename T, typename ALayout, typename BLayout>
auto GetHipBlasLtStridedBatchedGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, ALayout, BLayout, StridedBatchedGemmParams<T>>();
}

template <typename T, typename ALayout, typename BLayout>
auto GetHipBlasLtGemmFastGeluTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, ALayout, BLayout, GemmFastGeluParams<T>>(ActivationType::GELU);
}

#endif  // USE_HIPBLASLT

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
