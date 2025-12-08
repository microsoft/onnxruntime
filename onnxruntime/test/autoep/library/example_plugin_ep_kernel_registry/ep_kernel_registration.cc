// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "ep_kernel_registration.h"
#include "kernels/utils.h"

// Table of BuildKernelCreateInfo functions for each operator
static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    // Mul version 14
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 14, Mul)>,

    // Relu version 14
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 14, Relu)>,

    // Support Squeeze 21, 23, and 24.
    // Note: end versions are inclusive.
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOnnxDomain, 21, 22, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 23, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 24, Squeeze)>,
};

size_t GetNumKernels() {
  return std::size(build_kernel_create_info_funcs);
}

static OrtStatus* RegisterKernels(Ort::KernelRegistry& kernel_registry, const char* ep_name,
                                  void* create_kernel_state) {
  for (auto& build_func : build_kernel_create_info_funcs) {
    KernelCreateInfo kernel_create_info = {};
    RETURN_IF_ERROR(build_func(ep_name, create_kernel_state, &kernel_create_info));

    if (kernel_create_info.kernel_def != nullptr) {
      RETURN_IF_ERROR(kernel_registry.AddKernel(kernel_create_info.kernel_def,
                                                kernel_create_info.kernel_create_func,
                                                kernel_create_info.kernel_create_func_state));
    }
  }

  return nullptr;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  try {
    Ort::KernelRegistry kernel_registry;
    Ort::Status status{RegisterKernels(kernel_registry, ep_name, create_kernel_state)};

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
