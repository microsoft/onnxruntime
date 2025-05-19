// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"


namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true, 64);

// KTile=128 for Ada w4a8
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true, 128);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
