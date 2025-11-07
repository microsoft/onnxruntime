// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "ep_kernel_registration.h"
#include "kernels/utils.h"

// Include kernels:
#include "kernels/mul.h"
#include "kernels/squeeze.h"

// Table of BuildKernelCreateInfo functions for each operator
static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOnnxDomain, 7, 24, Mul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOnnxDomain, 13, 24, Squeeze)>,
};

constexpr size_t num_kernel_create_info_funcs = sizeof(build_kernel_create_info_funcs) /
                                                sizeof(build_kernel_create_info_funcs[0]);

size_t GetNumKernels() {
  return num_kernel_create_info_funcs;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  Ort::KernelRegistry kernel_registry;
  Ort::Status status = Ort::KernelRegistry::Create(kernel_registry);
  if (!status.IsOK()) {
    return status.release();
  }

  // Add kernel creation info to registry
  for (auto& build_func : build_kernel_create_info_funcs) {
    KernelCreateInfo kernel_create_info = {};
    status = Ort::Status{build_func(ep_name, create_kernel_state, &kernel_create_info)};

    if (!status.IsOK()) {
      break;
    }

    if (kernel_create_info.kernel_def != nullptr) {
      status = kernel_registry.AddKernel(kernel_create_info.kernel_def,
                                         kernel_create_info.kernel_create_func,
                                         kernel_create_info.kernel_create_func_state);

      if (!status.IsOK()) {
        break;
      }
    }
  }

  *out_kernel_registry = status.IsOK() ? kernel_registry.release() : nullptr;
  return status.release();
}
