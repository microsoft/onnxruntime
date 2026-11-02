// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "onnxruntime_c_api.h"

namespace onnxruntime::test {

// Private, same-build host/module interface; not a versioned plugin API. Neither
// side may unload its module while the other holds its objects or callbacks.
// All pointers are required unless documented otherwise. OrtStatus results are
// null on success; callers own failures and release them through the host OrtApi.
struct WebGpuInternalTestHostApi {
  // Borrow ep until release_kernel_info. Return borrowed info and an owned opaque
  // holder containing it; release that holder after all info/context users die.
  const OrtKernelInfo* (*create_kernel_info)(OrtEp*, void**);
  void (*release_kernel_info)(void*);
  // Return the holder's borrowed empty context (no inputs or outputs).
  OrtKernelContext* (*kernel_context)(void*);
};

// Snapshot of the selected EP's device limits/features and module-wide dispatch
// counter. matmul_unavailable is indexed by webgpu::MatMulAlgorithm (0..5);
// null means available, otherwise a borrowed static reason valid until unload.
struct WebGpuTestCapabilities {
  uint32_t max_workgroup_size_x{};
  uint32_t max_workgroup_invocations{};
  bool supports_dp4a{};
  bool supports_fp8_matrix{};
  uint64_t fp8_matrix_dispatches{};
  const char* matmul_unavailable[6]{};
};

// Initialize the module API/logger, install borrowed host callbacks and run GTest.
// api_base, logger and host_api must remain valid until return. argc/argv follow
// InitGoogleTest's mutable argument contract. Call once per loaded module, without
// live factories/EPs/contexts; return the GTest exit code (1 on setup exception).
// Tears down module state before returning; do not use that state afterwards.
using RunWebGpuInternalTestsFn = int (*)(const OrtApiBase*, const OrtLogger*, const WebGpuInternalTestHostApi*, int, char**);

// The remaining calls require CreateEpFactories to have initialized the module.
// Begin borrows session options for the call and returns an owned context token
// and its DP4A capability. Binding size 0 keeps the adapter limit; nonzero forces
// the exact test limit. On failure outputs are unspecified and no token is owned.
// Keep the token alive while creating EPs that share its context; release only
// through End, before ReleaseEpFactory. These setup calls must be serialized.
using BeginWebGpuTestContextFn = OrtStatus* (*)(const OrtSessionOptions*, uint64_t, void**, bool*) noexcept;
// Consume a token returned by Begin; null is allowed. No further token access.
using EndWebGpuTestContextFn = void (*)(void*) noexcept;
// Borrow a WebGPU OrtEp from this test module, with no live session using it.
// algorithm is a MatMulAlgorithm value (0..5); -1 restores automatic selection.
using ConfigureWebGpuTestEpFn = OrtStatus* (*)(OrtEp*, int) noexcept;
// Borrow a WebGPU OrtEp from this module (or the same-build production module)
// and fill the caller-owned snapshot. This reads cached metadata, not GPU work.
using InspectWebGpuTestEpFn = OrtStatus* (*)(const OrtEp*, WebGpuTestCapabilities*) noexcept;
}  // namespace onnxruntime::test
