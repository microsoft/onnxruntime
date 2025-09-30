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
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost)>,
};

constexpr size_t num_kernels = sizeof(build_kernel_create_info_funcs) /
                               sizeof(build_kernel_create_info_funcs[0]);

size_t GetNumKernels() { return num_kernels; }

OrtStatus* CreateKernelCreateInfos(const char* ep_name, std::vector<OrtKernelCreateInfo*>& result) {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  std::vector<OrtKernelCreateInfo*> kernel_create_infos;
  kernel_create_infos.reserve(num_kernels);

  for (auto& build_func : build_kernel_create_info_funcs) {
    OrtKernelCreateInfo* kernel_create_info = nullptr;

    if (OrtStatus* status = build_func(ep_name, &kernel_create_info); status != nullptr) {
      // Error occurred: clean up OrtKernelCreateInfo instances and return error.
      for (OrtKernelCreateInfo* create_info_to_release : kernel_create_infos) {
        ep_api.ReleaseKernelCreateInfo(create_info_to_release);
      }
      return status;
    }

    kernel_create_infos.push_back(kernel_create_info);
  }

  result = std::move(kernel_create_infos);
  return nullptr;
}
