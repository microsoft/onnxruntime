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
#if !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM

#include <cstddef>
#include <optional>

#include <cuda_runtime_api.h>

namespace onnxruntime {
// NOTE: we deliberately do NOT forward-declare Node here. This header has exactly two includers,
// and they live in genuinely different Node "worlds":
//   - core/providers/cuda/cuda_execution_provider.cc includes provider_api.h first, where the
//     shared-provider bridge declares `struct Node;` (provider_wrappedtypes.h defines `struct Node`).
//   - test/providers/cuda/test_cases/matmul_nbits_e2e_workspace_test.cc includes core/graph/graph.h
//     first, where the in-tree `Node` is a `class` (core/graph/graph.h: `class Node { ... }`).
// Forward-declaring Node ourselves would force us to pick a single tag (class or struct); either
// choice mismatches one of the two includers and triggers MSVC C4099 / GCC-Clang -Wmismatched-tags
// there. Both real includers already bring in a correct Node declaration (via their own core/bridge
// headers) BEFORE including this header, so the `const Node&` parameter below is already visible and
// no declaration of our own is needed. Keep this header included AFTER a Node-declaring header.
namespace contrib {
namespace cuda {

std::optional<size_t> EstimateMatMulNBitsWorkspace(const Node& node, const cudaDeviceProp& device_prop);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM
