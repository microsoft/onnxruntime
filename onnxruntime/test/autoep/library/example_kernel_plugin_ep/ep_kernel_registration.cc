// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "ep_kernel_registration.h"
#include "kernels/utils.h"

// Include kernels:
#include "kernels/mul.h"

// Forward declarations of kernel classes used as template args for BuildKernelCreateInfo
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 7, Mul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 13, Squeeze);

// Table of BuildKernelCreateInfo functions for each operator
static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 7, Mul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 13, Squeeze)>,
};

constexpr size_t num_kernel_create_info_funcs = sizeof(build_kernel_create_info_funcs) /
                                                sizeof(build_kernel_create_info_funcs[0]);

size_t GetNumKernels() {
  return num_kernel_create_info_funcs;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state, OrtKernelRegistry** kernel_registry) {
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
    status = build_func(ep_name, create_kernel_state, &kernel_create_info);

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
