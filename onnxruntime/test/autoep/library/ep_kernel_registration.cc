// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "ep_kernel_registration.h"
#include "kernels/utils.h"

// Include kernels:
#include "kernels/memcpy.h"

// Forward declarations of kernel classes used as template args for BuildKernelCreateInfo
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost);

// Table of BuildKernelCreateInfo functions for each operator
static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    BuildKernelCreateInfo<void>,  // Dummy to avoid table becoming empty.
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost)>,
};

constexpr size_t num_kernel_create_info_funcs = sizeof(build_kernel_create_info_funcs) /
                                                sizeof(build_kernel_create_info_funcs[0]);

size_t GetNumKernels() {
  static_assert(num_kernel_create_info_funcs >= 1);
  return num_kernel_create_info_funcs - 1;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, OrtKernelRegistry** kernel_registry) {
  *kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  const OrtEpApi& ep_api = Ort::GetEpApi();
  RETURN_IF_ERROR(ep_api.CreateKernelRegistry(kernel_registry));

  OrtStatus* status = nullptr;

  // Add kernel creation info to registry
  for (auto& build_func : build_kernel_create_info_funcs) {
    KernelCreateInfo kernel_create_info = {};

    status = build_func(ep_name, &kernel_create_info);
    DeferOrtRelease<OrtKernelDef> release_kernel_def(&kernel_create_info.kernel_def, ep_api.ReleaseKernelDef);

    if (status != nullptr) {
      break;
    }

    if (kernel_create_info.kernel_def != nullptr) {
      status = ep_api.KernelRegistry_AddKernel(*kernel_registry,
                                               kernel_create_info.kernel_def,  // copied
                                               kernel_create_info.kernel_create_func,
                                               kernel_create_info.kernel_create_func_state);
      if (status != nullptr) {
        break;
      }
    }
  }

  if (status != nullptr) {
    ep_api.ReleaseKernelRegistry(*kernel_registry);
    *kernel_registry = nullptr;
  }

  return status;
}
