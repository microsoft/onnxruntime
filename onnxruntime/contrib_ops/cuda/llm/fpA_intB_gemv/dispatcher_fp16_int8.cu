// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"


namespace ort_llm {
namespace kernels {
namespace fpA_intB_gemv {

INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true, 64);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace ort_llm
