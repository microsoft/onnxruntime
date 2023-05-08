// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/device/gemm_grouped.h"
// #include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
// #include "cutlass/gemm/kernel/default_gemm_grouped.h"
// #include "cutlass/gemm/kernel/gemm_grouped.h"
namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

// struct MatrixSize {
//   int64_t m;
//   int64_t n;
//   int64_t k;
// };

template <bool UseGroupGemm>
void GenerateLdaLdbLdcLdd(const std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                          gsl::span<int64_t> lda_span,
                          gsl::span<int64_t> ldb_span,
                          gsl::span<int64_t> ldc_span,
                          gsl::span<int64_t> ldd_span);

template <typename T, bool UseGroupGemm, bool DeviceOnlyMode>
Status GroupGemm_Impl(
    const CudaKernel* kernel,
    Stream* stream,
    std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
    cutlass::gemm::GemmCoord* problem_sizes_gpu_ptr,
    int64_t problem_count,
    int64_t* lda_gpu_ptr,
    int64_t* ldb_gpu_ptr,
    int64_t* ldc_gpu_ptr,
    int64_t* ldd_gpu_ptr,
    T** data_ptr_a_gpu_ptr,
    T** data_ptr_b_gpu_ptr,
    T** data_ptr_c_gpu_ptr,
    T** data_ptr_d_gpu_ptr);

// GroupGemm_Impl<CudaT, true, true>(onnxruntime::Stream*, std::vector<cutlass::gemm::GemmCoord>&,
//  cutlass::gemm::GemmCoord*, size_t&, long int*, long int*, long int*, long int*, const __half**,
//  const __half**, __half**, __half**, const CudaT*, const CudaT*, CudaT*)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
