// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../example_plugin_ep_utils.h"
#include "data_types.h"

using BuildKernelCreateInfoFn = OrtStatus* (*)(const char*, OrtKernelCreateInfo**);

template <typename T>
OrtStatus* BuildKernelCreateInfo(const char* ep_name, /*out*/ OrtKernelCreateInfo** result);

static constexpr const char* kOnnxDomain = "";

// Naming convention for operator kernel classes
#define ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, ver, name) \
  example_ep_##name##_##domain##_ver##ver

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, builder, kernel_class)                                   \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, ver, name);                                                 \
  template <>                                                                                               \
  OrtStatus*                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, ver, name)>(const char* ep_name,            \
                                                                            OrtKernelCreateInfo** result) { \
    try {                                                                                                   \
      const OrtEpApi& ep_api = Ort::GetEpApi();                                                             \
      *result = nullptr;                                                                                    \
                                                                                                            \
      OrtKernelDef* kernel_def = builder.SetOperatorType(#name)                                             \
                                     .SetDomain(domain)                                                     \
                                     .SetSinceVersion(ver)                                                  \
                                     .SetExecutionProvider(ep_name)                                         \
                                     .Build();                                                              \
                                                                                                            \
      DeferOrtRelease<OrtKernelDef> release_kernel_def(&kernel_def, ep_api.ReleaseKernelDef);               \
                                                                                                            \
      auto kernel_create_func = [](OrtKernelCreateContext* /*ctx*/, void* state, const OrtKernelInfo* info, \
                                   OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {                     \
        (void)state;                                                                                        \
        *kernel_out = nullptr;                                                                              \
                                                                                                            \
        std::unique_ptr<kernel_class> kernel;                                                               \
        RETURN_IF_ERROR(kernel_class::Create(info, kernel));                                                \
        *kernel_out = kernel.release();                                                                     \
        return nullptr;                                                                                     \
      };                                                                                                    \
                                                                                                            \
      RETURN_IF_ERROR(ep_api.CreateKernelCreationInfo(kernel_def, kernel_create_func, nullptr, result));    \
                                                                                                            \
    } catch (const Ort::Exception& ex) {                                                                    \
      Ort::Status status(ex);                                                                               \
      return status.release();                                                                              \
    } catch (const std::exception& ex) {                                                                    \
      Ort::Status status(ex.what(), ORT_EP_FAIL);                                                           \
      return status.release();                                                                              \
    }                                                                                                       \
    return nullptr;                                                                                         \
  }
