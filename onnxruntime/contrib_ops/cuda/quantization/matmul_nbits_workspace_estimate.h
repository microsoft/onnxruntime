// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Slim forward-declaring header for the Level-1 (partition-time) MatMulNBits memory estimate
// (Phase-A memory roadmap, issue microsoft/onnxruntime#29775). It deliberately pulls in NO CUTLASS
// or kernel headers so that CUDAExecutionProvider::GetCapability() can call the estimate without
// dragging the heavy fpA_intB template headers into cuda_execution_provider.cc. The full definition
// lives in matmul_nbits.cc.
//
// The whole header body is conditionally compiled: core CUDA EP code is compiled in configurations
// where DISABLE_CONTRIB_OPS is set, so this must not declare anything there.
#if !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include <cuda_runtime_api.h>
#include <gsl/gsl>

#include "core/framework/level1_memory_estimate.h"

namespace onnxruntime {
// NOTE: we deliberately do NOT forward-declare Node here. This header has includers in genuinely
// different Node "worlds":
//   - core/providers/cuda/cuda_execution_provider.cc includes provider_api.h first, where the
//     shared-provider bridge declares `struct Node;` (provider_wrappedtypes.h defines `struct Node`).
//   - test/providers/cuda/test_cases/matmul_nbits_e2e_workspace_test.cc includes core/graph/graph.h
//     first, where the in-tree `Node` is a `class` (core/graph/graph.h: `class Node { ... }`).
//   - matmul_nbits.h includes cuda_kernel.h first, which supplies the appropriate Node declaration
//     for its build world.
// Forward-declaring Node ourselves would force us to pick a single tag (class or struct); either
// choice mismatches an includer and triggers MSVC C4099 / GCC-Clang -Wmismatched-tags
// there. All includers bring in a correct Node declaration (via their own core/bridge
// headers) BEFORE including this header, so the `const Node&` parameter below is already visible and
// no declaration of our own is needed. Keep this header included AFTER a Node-declaring header.
namespace contrib {
namespace cuda {

struct MatMulNBitsMemoryEstimateOptions {
  std::optional<std::string_view> fpa_intb_gemm;
  std::optional<std::string_view> profile_m;
  // Applies only to the explicit-shape overload and only when the node's canonical input shape is dynamic.
  bool input_shape_is_upper_bound = false;
};

// Product of all input-A dimensions except the final K dimension. A known zero takes precedence
// over unknown/negative dimensions and potential overflow elsewhere in the leading dimensions.
// This graph-type-free helper is shared by the Level-1 shape wrappers and Level-2 TensorShape path.
std::optional<int64_t> ComputeMatMulNBitsLeadingDimProduct(gsl::span<const int64_t> input_a_shape);

std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemory(
    const Node& node, const cudaDeviceProp& device_prop,
    MatMulNBitsMemoryEstimateOptions options = {});

// Compatibility wrappers for callers that need only the runtime workspace component.
std::optional<size_t> EstimateMatMulNBitsWorkspace(const Node& node, const cudaDeviceProp& device_prop);

// Uses an estimation-only input A shape. Set input_shape_is_upper_bound when it was propagated
// from maximum graph inputs; a canonical static input shape remains exact.
std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemory(
    const Node& node, gsl::span<const int64_t> input_a_shape, const cudaDeviceProp& device_prop,
    MatMulNBitsMemoryEstimateOptions options = {});
std::optional<size_t> EstimateMatMulNBitsWorkspace(
    const Node& node, gsl::span<const int64_t> input_a_shape, const cudaDeviceProp& device_prop);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM
