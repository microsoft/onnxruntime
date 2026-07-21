// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Slim forward-declaring header for the Level-1 (partition-time) MatMulNBits workspace estimate
// (Phase-A memory roadmap, issue microsoft/onnxruntime#29775). It deliberately pulls in NO CUTLASS
// or kernel headers so that CUDAExecutionProvider::GetCapability() can call the estimate without
// dragging the heavy fpA_intB template headers into cuda_execution_provider.cc. The full definition
// lives in matmul_nbits.cc.
//
// The whole header body is conditionally compiled: core CUDA EP code is compiled in configurations
// where DISABLE_CONTRIB_OPS is set, so this must not declare anything there.
#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM)

#include <cstddef>
#include <optional>

#include <cuda_runtime_api.h>

namespace onnxruntime {
class Node;
namespace contrib {
namespace cuda {

std::optional<size_t> EstimateMatMulNBitsWorkspace(const Node& node, const cudaDeviceProp& device_prop);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM)
