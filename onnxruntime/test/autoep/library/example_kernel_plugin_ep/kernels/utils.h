// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"
#include "data_types.h"

/// <summary>
/// Contains information to create a kernel: kernel definition, creation function + state.
/// </summary>
struct KernelCreateInfo {
  KernelCreateInfo() = default;
  KernelCreateInfo(Ort::KernelDef def, OrtKernelCreateFunc func, void* state)
      : kernel_def{std::move(def)}, kernel_create_func{func}, kernel_create_func_state{state} {}

  Ort::KernelDef kernel_def{nullptr};
  OrtKernelCreateFunc kernel_create_func = nullptr;
  void* kernel_create_func_state = nullptr;
};

using BuildKernelCreateInfoFn = OrtStatus* (*)(const char*, void*, KernelCreateInfo*);

template <typename T>
OrtStatus* BuildKernelCreateInfo(const char* ep_name, void* create_func_state, /*out*/ KernelCreateInfo* result);

template <>
inline OrtStatus* BuildKernelCreateInfo<void>(const char* /*ep_name*/, void* /*create_func_state*/,
                                              /*out*/ KernelCreateInfo* result) {
  result->kernel_def = Ort::KernelDef{nullptr};
  result->kernel_create_func = nullptr;
  result->kernel_create_func_state = nullptr;
  return nullptr;
}

static constexpr const char* kOnnxDomain = "";

// Naming convention for operator kernel classes
#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name) \
  example_ep_##name##_##domain##_ver##startver##_##endver

#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, builder, kernel_class)            \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name);                          \
  template <>                                                                                               \
  OrtStatus*                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name)>(         \
      const char* ep_name,                                                                                  \
      void* create_kernel_state,                                                                            \
      KernelCreateInfo* result) {                                                                           \
    try {                                                                                                   \
      Ort::KernelDef kernel_def = builder.SetOperatorType(#name)                                            \
                                      .SetDomain(domain)                                                    \
                                      .SetSinceVersion(startver, endver)                                    \
                                      .SetExecutionProvider(ep_name)                                        \
                                      .Build();                                                             \
                                                                                                            \
      auto kernel_create_func = [](OrtKernelCreateContext* /*ctx*/, void* state, const OrtKernelInfo* info, \
                                   OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {                     \
        *kernel_out = nullptr;                                                                              \
                                                                                                            \
        std::unique_ptr<kernel_class> kernel;                                                               \
        RETURN_IF_ERROR(kernel_class::Create(info, state, kernel));                                         \
        *kernel_out = kernel.release();                                                                     \
        return nullptr;                                                                                     \
      };                                                                                                    \
                                                                                                            \
      *result = KernelCreateInfo(std::move(kernel_def), kernel_create_func, create_kernel_state);           \
    } catch (const Ort::Exception& ex) {                                                                    \
      Ort::Status status(ex);                                                                               \
      return status.release();                                                                              \
    } catch (const std::exception& ex) {                                                                    \
      Ort::Status status(ex.what(), ORT_EP_FAIL);                                                           \
      return status.release();                                                                              \
    }                                                                                                       \
    return nullptr;                                                                                         \
  }
