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
constexpr rocblas_datatype RocBlasDataTypeFor(const T*) {
  static_assert(sizeof(T) == -1, "Unsupported type for rocBLAS operation.");
  // The code below should be unreachable due to the static_assert above.
  // But the compiler doesn't like not having a return statement, so we
  // return something sensible.
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<float>(const float*) {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<half>(const half*) {
  return rocblas_datatype_f16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<double>(const double*) {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<BFloat16>(const BFloat16*) {
  return rocblas_datatype_bf16_r;
}

template <typename T>
constexpr rocblas_datatype RocBlasComputeTypeFor(const T*) {
  static_assert(sizeof(T) == -1, "Unsupported type for rocBLAS operation.");
  // The code below should be unreachable due to the static_assert above.
  // But the compiler doesn't like not having a return statement, so we
  // return something sensible.
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<float>(const float*) {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<half>(const half*) {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // FP16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<double>(const double*) {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<BFloat16>(const BFloat16*) {
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
auto DoCastForHalfOrBfloat16<half>(const half fp) {
  // alpha and beta should be the same as compute_type, in half case it is float.
  float h = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(&fp));
  return h;
}

template <>
auto DoCastForHalfOrBfloat16<BFloat16>(const BFloat16 fp) {
  // alpha and beta should be the same as compute_type, in bfloat16 case it is float.
  float h = fp.ToFloat();
  return h;
}

template <typename T>
class IndexedRocBlasGemmOp {
 public:
  IndexedRocBlasGemmOp()
      : index_(0) {}
  IndexedRocBlasGemmOp(int index)
      : index_(index) {}

  Status operator()(const GemmParams<T>* params) {
    RocblasHandleStreamGuard guard(params->handle, params->stream);
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    return ROCBLAS_CALL(
        rocblas_gemm_ex(
            params->handle,
            params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->n, params->m, params->k,
            &h_a,
            params->b, RocBlasDataTypeFor(params->b), params->ldb,
            params->a, RocBlasDataTypeFor(params->a), params->lda,
            &h_b,
            params->c, RocBlasDataTypeFor(params->c), params->ldc,
            params->c, RocBlasDataTypeFor(params->c), params->ldc,
            RocBlasComputeTypeFor(params->a),
            rocblas_gemm_algo_solution_index,
            index_,
            rocblas_gemm_flags_none));
  }

  Status IsSupported(const GemmParams<T>*) {
    return Status::OK();
  }

 private:
  int index_;
};

template <typename T>
class RocBlasGemmTunableOp : public TunableOp<GemmParams<T>> {
 public:
  RocBlasGemmTunableOp() {
    // Ensure that the default implementation is always present
    this->RegisterOp(IndexedRocBlasGemmOp<T>{0});
  }

  Status IsSupported(const GemmParams<T>* params) {
    ORT_UNUSED_PARAMETER(params);
    return Status::OK();
  }

 protected:
  virtual int FindFastest(const GemmParams<T>* params) override {
    auto solution_indices = this->GetSolutions(params);
    std::vector<Op<GemmParams<T>>> candidates;
    for (int solution_idx : solution_indices) {
      candidates.emplace_back(IndexedRocBlasGemmOp<T>{solution_idx});
    }

    auto id = this->FindFastestImpl(params, candidates);
    // memoize the result
    this->RegisterOp(std::move(candidates[id]));
    return this->NumberOfOps() - 1;
  }

 private:
  std::vector<int> GetSolutions(const GemmParams<T>* params) {
    int num_solutions = 0;
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    // Get the number of candidate solutions
    ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->b, RocBlasDataTypeFor(params->b), params->ldb,
        params->a, RocBlasDataTypeFor(params->a), params->lda,
        &h_b,
        params->c, RocBlasDataTypeFor(params->c), params->ldc,
        params->c, RocBlasDataTypeFor(params->c), params->ldc,
        RocBlasComputeTypeFor(params->a),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        NULL,
        &num_solutions));

    // Get the actual candidate solutions
    std::vector<int> solutions(num_solutions);
    ROCBLAS_CALL_THROW(rocblas_gemm_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->b, RocBlasDataTypeFor(params->b), params->ldb,
        params->a, RocBlasDataTypeFor(params->a), params->lda,
        &h_b,
        params->c, RocBlasDataTypeFor(params->c), params->ldc,
        params->c, RocBlasDataTypeFor(params->c), params->ldc,
        RocBlasComputeTypeFor(params->a),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        solutions.data(),
        &num_solutions));

    return solutions;
  }
};

template <typename T>
class IndexedRocBlasBatchedGemmOp {
 public:
  IndexedRocBlasBatchedGemmOp()
      : index_(0) {}
  IndexedRocBlasBatchedGemmOp(int index)
      : index_(index) {}

  Status operator()(const BatchedGemmParams<T>* params) {
    RocblasHandleStreamGuard guard(params->handle, params->stream);
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    return ROCBLAS_CALL(
        rocblas_gemm_batched_ex(
            params->handle,
            params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->n, params->m, params->k,
            &h_a,
            params->bs, RocBlasDataTypeFor(*(params->bs)), params->ldb,
            params->as, RocBlasDataTypeFor(*(params->as)), params->lda,
            &h_b,
            params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
            params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
            params->batch,
            RocBlasComputeTypeFor(*(params->as)),
            rocblas_gemm_algo_solution_index,
            index_,
            rocblas_gemm_flags_none));
  }

  Status IsSupported(const BatchedGemmParams<T>*) {
    return Status::OK();
  }

 private:
  int index_;
};

template <typename T>
class RocBlasBatchedGemmTunableOp : public TunableOp<BatchedGemmParams<T>> {
 public:
  RocBlasBatchedGemmTunableOp() {
    // Ensure that the default implementation is always present
    this->RegisterOp(IndexedRocBlasBatchedGemmOp<T>{0});
  }

  Status IsSupported(const BatchedGemmParams<T>* params) {
    ORT_UNUSED_PARAMETER(params);
    return Status::OK();
  }

 protected:
  virtual int FindFastest(const BatchedGemmParams<T>* params) override {
    auto solution_indices = this->GetSolutions(params);
    std::vector<Op<BatchedGemmParams<T>>> candidates;
    for (int solution_idx : solution_indices) {
      candidates.emplace_back(IndexedRocBlasBatchedGemmOp<T>{solution_idx});
    }

    auto id = this->FindFastestImpl(params, candidates);
    // memoize the result
    this->RegisterOp(std::move(candidates[id]));
    return this->NumberOfOps() - 1;
  }

 private:
  std::vector<int> GetSolutions(const BatchedGemmParams<T>* params) {
    int num_solutions = 0;
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    // Get the number of candidate solutions
    ROCBLAS_CALL_THROW(rocblas_gemm_batched_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->bs, RocBlasDataTypeFor(*(params->bs)), params->ldb,
        params->as, RocBlasDataTypeFor(*(params->as)), params->lda,
        &h_b,
        params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
        params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
        params->batch,
        RocBlasComputeTypeFor(*(params->as)),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        NULL,
        &num_solutions));

    // Get the actual candidate solutions
    std::vector<int> solutions(num_solutions);
    ROCBLAS_CALL_THROW(rocblas_gemm_batched_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->bs, RocBlasDataTypeFor(*(params->bs)), params->ldb,
        params->as, RocBlasDataTypeFor(*(params->as)), params->lda,
        &h_b,
        params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
        params->cs, RocBlasDataTypeFor(*(params->cs)), params->ldc,
        params->batch,
        RocBlasComputeTypeFor(*(params->as)),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        solutions.data(),
        &num_solutions));

    return solutions;
  }
};

template <typename T>
class IndexedRocBlasStridedBatchedGemmOp {
 public:
  IndexedRocBlasStridedBatchedGemmOp()
      : index_(0) {}
  IndexedRocBlasStridedBatchedGemmOp(int index)
      : index_(index) {}

  Status operator()(const StridedBatchedGemmParams<T>* params) {
    RocblasHandleStreamGuard guard(params->handle, params->stream);
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    return ROCBLAS_CALL(
        rocblas_gemm_strided_batched_ex(
            params->handle,
            params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
            params->n, params->m, params->k,
            &h_a,
            params->b, RocBlasDataTypeFor(params->b), params->ldb, params->stride_b,
            params->a, RocBlasDataTypeFor(params->a), params->lda, params->stride_a,
            &h_b,
            params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
            params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
            params->batch,
            RocBlasComputeTypeFor(params->a),
            rocblas_gemm_algo_solution_index,
            index_,
            rocblas_gemm_flags_none));
  }

  Status IsSupported(const StridedBatchedGemmParams<T>*) {
    return Status::OK();
  }

 private:
  int index_;
};

template <typename T>
class RocBlasStridedBatchedGemmTunableOp : public TunableOp<StridedBatchedGemmParams<T>> {
 public:
  RocBlasStridedBatchedGemmTunableOp() {
    // Ensure that the default implementation is always present
    this->RegisterOp(IndexedRocBlasStridedBatchedGemmOp<T>{0});
  }

  Status IsSupported(const StridedBatchedGemmParams<T>* params) {
    ORT_UNUSED_PARAMETER(params);
    return Status::OK();
  }

 protected:
  virtual int FindFastest(const StridedBatchedGemmParams<T>* params) override {
    auto solution_indices = this->GetSolutions(params);
    std::vector<Op<StridedBatchedGemmParams<T>>> candidates;
    for (int solution_idx : solution_indices) {
      candidates.emplace_back(IndexedRocBlasStridedBatchedGemmOp<T>{solution_idx});
    }

    auto id = this->FindFastestImpl(params, candidates);
    // memoize the result
    this->RegisterOp(std::move(candidates[id]));
    return this->NumberOfOps() - 1;
  }

 private:
  std::vector<int> GetSolutions(const StridedBatchedGemmParams<T>* params) {
    int num_solutions = 0;
    auto h_a = DoCastForHalfOrBfloat16(params->alpha);
    auto h_b = DoCastForHalfOrBfloat16(params->beta);
    // Get the number of candidate solutions
    ROCBLAS_CALL_THROW(rocblas_gemm_strided_batched_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->b, RocBlasDataTypeFor(params->b), params->ldb, params->stride_b,
        params->a, RocBlasDataTypeFor(params->a), params->lda, params->stride_a,
        &h_b,
        params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
        params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
        params->batch,
        RocBlasComputeTypeFor(params->a),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        NULL,
        &num_solutions));

    // Get the actual candidate solutions
    std::vector<int> solutions(num_solutions);
    ROCBLAS_CALL_THROW(rocblas_gemm_strided_batched_ex_get_solutions(
        params->handle,
        params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
        params->n, params->m, params->k,
        &h_a,
        params->b, RocBlasDataTypeFor(params->b), params->ldb, params->stride_b,
        params->a, RocBlasDataTypeFor(params->a), params->lda, params->stride_a,
        &h_b,
        params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
        params->c, RocBlasDataTypeFor(params->c), params->ldc, params->stride_c,
        params->batch,
        RocBlasComputeTypeFor(params->a),
        rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none,
        solutions.data(),
        &num_solutions));

    return solutions;
  }
};

#endif /* #ifdef USE_ROCBLAS_EXTENSION_API */

template <typename T>
Status RocBlasGemmOp(const GemmParams<T>* params) {
  RocblasHandleStreamGuard guard(params->handle, params->stream);
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
  RocblasHandleStreamGuard guard(params->handle, params->stream);
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
  RocblasHandleStreamGuard guard(params->handle, params->stream);
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
