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

static Ort::Status RegisterKernels(Ort::KernelRegistry& kernel_registry, const char* ep_name,
                                   void* create_kernel_state) {
  for (auto& build_func : build_kernel_create_info_funcs) {
    KernelCreateInfo kernel_create_info = {};
    RETURN_IF_ERROR_CXX(build_func(ep_name, create_kernel_state, &kernel_create_info));

    if (kernel_create_info.kernel_def != nullptr) {
      RETURN_IF_ERROR_CXX(kernel_registry.AddKernel(kernel_create_info.kernel_def,
                                                    kernel_create_info.kernel_create_func,
                                                    kernel_create_info.kernel_create_func_state));
    }
  }

  return Ort::Status{nullptr};
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  try {
    Ort::KernelRegistry kernel_registry;
    Ort::Status status = RegisterKernels(kernel_registry, ep_name, create_kernel_state);

    *out_kernel_registry = status.IsOK() ? kernel_registry.release() : nullptr;
    return status.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
