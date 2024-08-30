// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

// RAII style guard to set stream and restore original stream for rocblas_handle
class RocblasHandleStreamGuard {
 public:
  RocblasHandleStreamGuard(rocblas_handle handle, hipStream_t stream) : handle_{handle} {
    ROCBLAS_CALL_THROW(rocblas_get_stream(handle_, &original_stream_));
    ROCBLAS_CALL_THROW(rocblas_set_stream(handle_, stream));
  }

  ~RocblasHandleStreamGuard() {
    ROCBLAS_CALL_THROW(rocblas_set_stream(handle_, original_stream_));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(RocblasHandleStreamGuard);

 private:
  rocblas_handle handle_;
  hipStream_t original_stream_;
};

#ifdef USE_ROCBLAS_EXTENSION_API

template <typename T>
constexpr rocblas_datatype RocBlasDataTypeFor();

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<half>() {
  return rocblas_datatype_f16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<BFloat16>() {
  return rocblas_datatype_bf16_r;
}

template <typename T>
constexpr rocblas_datatype RocBlasComputeTypeFor();

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<half>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // FP16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<BFloat16>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // BF16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <typename T>
auto DoCastForHalfOrBfloat16(const T fp) {
  return fp;
}

template <>
inline auto DoCastForHalfOrBfloat16<half>(const half fp) {
  // alpha and beta should be the same as compute_type, in half case it is float.
  float h = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(&fp));
  return h;
}

template <>
inline auto DoCastForHalfOrBfloat16<BFloat16>(const BFloat16 fp) {
  // alpha and beta should be the same as compute_type, in bfloat16 case it is float.
  float h = fp.ToFloat();
  return h;
}

template <typename T>
auto GetRocBlasGemmTypeStringAndOps() {
  rocblas_handle handle;
  ROCBLAS_CALL_THROW(rocblas_create_handle(&handle));

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           nullptr,
                                                           &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           solutions.data(),
                                                           &solution_size));

  ROCBLAS_CALL_THROW(rocblas_destroy_handle(handle));

  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Op<GemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const GemmParams<T>* params) -> Status {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_ex(
          params->handle,
          params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->n, params->m, params->k,
          &h_a,
          params->b, input_output_type, params->ldb,
          params->a, input_output_type, params->lda,
          &h_b,
          params->c, input_output_type, params->ldc,
          params->c, input_output_type, params->ldc,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != rocblas_status_success,
          "[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status));

      return Status::OK();
    };
    ret.emplace_back(std::make_pair(
        onnxruntime::MakeString("RocBlasGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

template <typename T>
auto GetRocBlasBatchedGemmTypeStringAndOps() {
  rocblas_handle handle;
  ROCBLAS_CALL_THROW(rocblas_create_handle(&handle));

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                   input_output_type,
                                                                   input_output_type,
                                                                   compute_type,
                                                                   rocblas_gemm_flags_none,
                                                                   nullptr,
                                                                   &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                   input_output_type,
                                                                   input_output_type,
                                                                   compute_type,
                                                                   rocblas_gemm_flags_none,
                                                                   solutions.data(),
                                                                   &solution_size));

  ROCBLAS_CALL_THROW(rocblas_destroy_handle(handle));

  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Op<BatchedGemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const BatchedGemmParams<T>* params) -> Status {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_batched_ex(
          params->handle,
          params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->n, params->m, params->k,
          &h_a,
          params->bs, input_output_type, params->ldb,
          params->as, input_output_type, params->lda,
          &h_b,
          params->cs, input_output_type, params->ldc,
          params->cs, input_output_type, params->ldc,
          params->batch,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != rocblas_status_success,
          "[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status));

      return Status::OK();
    };
    ret.emplace_back(std::make_pair(
        onnxruntime::MakeString("RocBlasBatchedGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

template <typename T>
auto GetRocBlasStridedBatchedGemmTypeStringAndOps() {
  rocblas_handle handle;
  ROCBLAS_CALL_THROW(rocblas_create_handle(&handle));

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           nullptr,
                                                           &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           solutions.data(),
                                                           &solution_size));

  ROCBLAS_CALL_THROW(rocblas_destroy_handle(handle));

  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Op<StridedBatchedGemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const StridedBatchedGemmParams<T>* params) -> Status {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_strided_batched_ex(
          params->handle,
          params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->n, params->m, params->k,
          &h_a,
          params->b, input_output_type, params->ldb, params->stride_b,
          params->a, input_output_type, params->lda, params->stride_a,
          &h_b,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->batch,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != rocblas_status_success,
          "[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status));

      return Status::OK();
    };
    ret.emplace_back(std::make_pair(
        onnxruntime::MakeString("RocBlasStridedBatchedGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

#endif  // USE_ROCBLAS_EXTENSION_API

template <typename T>
Status RocBlasGemmOp(const GemmParams<T>* params) {
  RocblasHandleStreamGuard guard(params->handle, params->StreamHandle());
  // NOTE: rocblas assumes the storage is column-majored, swapping A and B makes it have the same interface
  // as those with row-majored convention. That is, if you treat the storage as row-majored but view the matrices as
  // transposed, then by using the property Transpose(A*B) = Tranpose(B)*Transpose(A), the correctness is obvious.
  return ROCBLAS_CALL(rocblasGemmHelper(
      params->handle,
      params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->n, params->m, params->k,
      &(params->alpha),
      params->b, params->ldb,
      params->a, params->lda,
      &(params->beta),
      params->c, params->ldc));
}

template <typename T>
Status RocBlasBatchedGemmOp(const BatchedGemmParams<T>* params) {
  RocblasHandleStreamGuard guard(params->handle, params->StreamHandle());
  return ROCBLAS_CALL(rocblasGemmBatchedHelper(
      params->handle,
      params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->n, params->m, params->k,
      &(params->alpha),
      params->bs, params->ldb,
      params->as, params->lda,
      &(params->beta),
      params->cs, params->ldc,
      params->batch));
}

template <typename T>
Status RocBlasStridedBatchedGemmOp(const StridedBatchedGemmParams<T>* params) {
  RocblasHandleStreamGuard guard(params->handle, params->StreamHandle());
  return ROCBLAS_CALL(rocblasGemmStridedBatchedHelper(
      params->handle,
      params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->n, params->m, params->k,
      &(params->alpha),
      params->b, params->ldb, params->stride_b,
      params->a, params->lda, params->stride_a,
      &(params->beta),
      params->c, params->ldc, params->stride_c,
      params->batch));
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
