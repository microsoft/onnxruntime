// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/device/gemm_grouped.h"
// #include "cutlass/gemm/device/gemm_universal.h"
// #include "cutlass/gemm/gemm.h"
// #include "cutlass/gemm/kernel/default_gemm_grouped.h"
// #include "cutlass/gemm/kernel/gemm_grouped.h"
namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T, bool UseGroupGemm, bool DeviceOnlyMode>
Status GroupGemm_Impl(
    const CudaKernel* kernel,
    Stream* stream,
    std::vector<std::tuple<int64_t, int64_t, int64_t>>& problem_sizes,
    int64_t problem_count,
    std::vector<const T*> data_ptr_a_vec,
    std::vector<const T*> data_ptr_b_vec,
    std::vector<T*> data_ptr_c_vec,
    std::vector<T*> data_ptr_d_vec);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
