// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/get_execution_providers.h"

#include "core/graph/constants.h"
#include "core/common/common.h"

#include <algorithm>
#include <string_view>

#if !defined(ORT_MINIMAL_BUILD)
#include "core/platform/env.h"
#include <filesystem>
#endif

namespace onnxruntime {

namespace {
struct ProviderInfo {
  std::string_view name;
  bool available;
  // Shared library base name (without prefix/extension), or nullptr for
  // statically linked providers. This is the single source of truth for
  // mapping provider names to their library filenames.
  const char* library_base_name;
};

// all providers ordered by default priority from highest to lowest
// kCpuExecutionProvider should always be last
//
// library_base_name: the base name of the shared library for this provider,
// e.g. "onnxruntime_providers_cuda" -> onnxruntime_providers_cuda.dll (Win)
//      or libonnxruntime_providers_cuda.so (Linux).
// nullptr means the provider is statically linked (no shared library to check).
constexpr ProviderInfo kProvidersInPriorityOrder[] =
    {
        {
            kNvTensorRTRTXExecutionProvider,
#ifdef USE_NV
            true,
#else
            false,
#endif
            "onnxruntime_providers_nv_tensorrt_rtx",
        },
        {
            kTensorrtExecutionProvider,
#ifdef USE_TENSORRT
            true,
#else
            false,
#endif
            "onnxruntime_providers_tensorrt",
        },
        {
            kCudaExecutionProvider,
#ifdef USE_CUDA
            true,
#else
            false,
#endif
            "onnxruntime_providers_cuda",
        },
        {
            kMIGraphXExecutionProvider,
#ifdef USE_MIGRAPHX
            true,
#else
            false,
#endif
            "onnxruntime_providers_migraphx",
        },
        {
            kOpenVINOExecutionProvider,
#ifdef USE_OPENVINO
            true,
#else
            false,
#endif
            "onnxruntime_providers_openvino",
        },
        {
            kDnnlExecutionProvider,
#ifdef USE_DNNL
            true,
#else
            false,
#endif
            "onnxruntime_providers_dnnl",
        },
        {
            kVitisAIExecutionProvider,
#ifdef USE_VITISAI
            true,
#else
            false,
#endif
            "onnxruntime_providers_vitisai",
        },
        {
            kQnnExecutionProvider,
#ifdef USE_QNN
            true,
#else
            false,
#endif
#if defined(BUILD_QNN_EP_STATIC_LIB) && BUILD_QNN_EP_STATIC_LIB
            nullptr,  // QNN is statically linked in this build
#else
            "onnxruntime_providers_qnn",
#endif
        },
        {
            kNnapiExecutionProvider,
#ifdef USE_NNAPI
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kVSINPUExecutionProvider,
#ifdef USE_VSINPU
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kJsExecutionProvider,
#ifdef USE_JSEP
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kCoreMLExecutionProvider,
#ifdef USE_COREML
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kAclExecutionProvider,
#ifdef USE_ACL
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kDmlExecutionProvider,
#ifdef USE_DML
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kRknpuExecutionProvider,
#ifdef USE_RKNPU
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kWebNNExecutionProvider,
#ifdef USE_WEBNN
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kWebGpuExecutionProvider,
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
            true,
#else
            false,
#endif
            // When USE_WEBGPU is defined with ORT_USE_EP_API_ADAPTERS, WebGPU is
            // loaded via the plugin EP system, not via the provider bridge. We mark
            // it as nullptr (statically linked) here; the plugin EP adapter handles
            // its own loading. If WebGPU is built without adapters, it's static.
            nullptr,
        },
        {
            kXnnpackExecutionProvider,
#ifdef USE_XNNPACK
            true,
#else
            false,
#endif
            nullptr,
        },
        {
            kCannExecutionProvider,
#ifdef USE_CANN
            true,
#else
            false,
#endif
            "onnxruntime_providers_cann",
        },
        {
            kAzureExecutionProvider,
#ifdef USE_AZURE
            true,
#else
            false,
#endif
            nullptr,
        },
        {kCpuExecutionProvider, true, nullptr},  // CPU is always last, always static
};

constexpr size_t kAllExecutionProvidersCount = sizeof(kProvidersInPriorityOrder) / sizeof(ProviderInfo);

#if !defined(ORT_MINIMAL_BUILD)
// Check whether the shared library file for a provider exists on disk.
// This does NOT load the library — it only checks file existence using
// std::filesystem, so there are no side effects (no memory footprint
// increase, no hardware initialization, no error logs).
bool DoesProviderLibraryExist(const char* library_base_name) {
  if (library_base_name == nullptr) {
    // Statically linked — always present
    return true;
  }

  // Build the expected library filename using the same convention as
  // ProviderLibrary in provider_bridge_ort.cc:
  //   GetRuntimePath() + LIBRARY_PREFIX + base_name + LIBRARY_EXTENSION
#ifdef _WIN32
  std::string lib_filename = std::string(library_base_name) + ".dll";
#elif defined(__APPLE__)
  std::string lib_filename = std::string("lib") + library_base_name + ".dylib";
#else
  std::string lib_filename = std::string("lib") + library_base_name + ".so";
#endif

  std::filesystem::path full_path(Env::Default().GetRuntimePath());
  full_path /= lib_filename;

  std::error_code ec;
  return std::filesystem::exists(full_path, ec) && !ec;
}
#endif  // !ORT_MINIMAL_BUILD

// Find the ProviderInfo entry for a given provider name
const ProviderInfo* FindProviderInfo(const std::string& provider_name) {
  for (const auto& provider : kProvidersInPriorityOrder) {
    if (provider.name == provider_name) {
      return &provider;
    }
  }
  return nullptr;
}

}  // namespace

const std::vector<std::string>& GetAllExecutionProviderNames() {
  static const auto all_execution_providers = []() {
    std::vector<std::string> result{};
    result.reserve(kAllExecutionProvidersCount);
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      result.push_back(std::string(provider.name));
    }
    return result;
  }();

  return all_execution_providers;
}

const std::vector<std::string>& GetAvailableExecutionProviderNames() {
  static const auto available_execution_providers = []() {
    std::vector<std::string> result{};
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      if (provider.available) {
        result.push_back(std::string(provider.name));
      }
    }
    return result;
  }();

  return available_execution_providers;
}

bool IsExecutionProviderUsable(const std::string& provider_name) {
  const auto* info = FindProviderInfo(provider_name);
  if (info == nullptr || !info->available) {
    return false;
  }

#if !defined(ORT_MINIMAL_BUILD)
  return DoesProviderLibraryExist(info->library_base_name);
#else
  // In minimal builds, shared-library providers are not used.
  // If compiled in, it is usable.
  return true;
#endif
}

std::vector<std::string> GetUsableExecutionProviderNames() {
  std::vector<std::string> usable{};
  for (const auto& provider : kProvidersInPriorityOrder) {
    if (!provider.available) {
      continue;
    }
#if !defined(ORT_MINIMAL_BUILD)
    if (!DoesProviderLibraryExist(provider.library_base_name)) {
      continue;
    }
#endif
    usable.push_back(std::string(provider.name));
  }
  return usable;
}

}  // namespace onnxruntime
