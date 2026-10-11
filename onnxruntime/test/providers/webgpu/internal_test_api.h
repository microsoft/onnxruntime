// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "onnxruntime_c_api.h"

namespace onnxruntime::test {
struct WebGpuInternalTestHostApi {
  const OrtKernelInfo* (*create_kernel_info)(OrtEp*, void**);
  void (*release_kernel_info)(void*);
  OrtKernelContext* (*kernel_context)(void*);
};
struct WebGpuTestCapabilities {
  uint32_t max_workgroup_size_x{};
  uint32_t max_workgroup_invocations{};
  bool supports_dp4a{};
  bool supports_fp8_matrix{};
  uint64_t fp8_matrix_dispatches{};
  const char* matmul_unavailable[6]{};
};
using RunWebGpuInternalTestsFn = int (*)(const OrtApiBase*, const OrtLogger*, const WebGpuInternalTestHostApi*, int, char**);
using BeginWebGpuTestContextFn = OrtStatus* (*)(const OrtSessionOptions*, uint64_t, void**, bool*) noexcept;
using EndWebGpuTestContextFn = void (*)(void*) noexcept;
using ConfigureWebGpuTestEpFn = OrtStatus* (*)(OrtEp*, int) noexcept;
using InspectWebGpuTestEpFn = OrtStatus* (*)(const OrtEp*, WebGpuTestCapabilities*) noexcept;
}  // namespace onnxruntime::test
