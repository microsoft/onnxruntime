// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// See the comment on ORT_SKIP_API_MANUAL_INIT in include/onnxruntime/ep/api.h for why manual init of the C++ API is
// only used when this EP is built as its own shared library. Key this off the same macro so that every translation
// unit in the binary agrees on ORT_API_MANUAL_INIT.
#if !defined(ORT_SKIP_API_MANUAL_INIT)
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#else
#include "onnxruntime_cxx_api.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "core/platform/env_var.h"
#include "core/providers/webgpu/ep/factory.h"
#include "core/session/onnxruntime_env_config_keys.h"

// When this EP is statically linked into the host binary instead of being built as a separate plugin library, the
// entry points below are given a `WebGpu_` prefix so that multiple statically linked plugin EPs can coexist in one
// binary. ORT core declares the prefixed names in onnxruntime/core/session/plugin_ep/ep_static_plugins.cc, which
// spells them out literally, so the prefix is hardcoded on both sides rather than configured by the build.
//
// The shared library build must leave the entry points unprefixed, because they are resolved by exact name.
#if defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#define ORT_PLUGIN_EP_ENTRY_POINT_CONCAT_IMPL(prefix, name) prefix##name
#define ORT_PLUGIN_EP_ENTRY_POINT_CONCAT(prefix, name) ORT_PLUGIN_EP_ENTRY_POINT_CONCAT_IMPL(prefix, name)
#define ORT_PLUGIN_EP_ENTRY_POINT(name) ORT_PLUGIN_EP_ENTRY_POINT_CONCAT(WebGpu_, name)
#else
#define ORT_PLUGIN_EP_ENTRY_POINT(name) name
#endif

// To make symbols visible on macOS/iOS. Only needed when this EP is built as a separate shared library.
#if defined(__APPLE__) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

namespace onnxruntime {
namespace webgpu {
void CleanupWebGpuContexts();
void CleanupKernelRegistries();
}  // namespace webgpu
}  // namespace onnxruntime

#if !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace google {
namespace protobuf {
void ShutdownProtobufLibrary();
}  // namespace protobuf
}  // namespace google
#endif

namespace {
// Set to "1" to allow WebGPU to be advertised against a CPU hardware device when no GPU hardware device is available.
constexpr const char* kAllowSoftwareAdapterEnvironmentVariable = "ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER";
}  // namespace

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* ORT_PLUGIN_EP_ENTRY_POINT(CreateEpFactories)(
    const char* /*registration_name*/, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories, size_t max_factories, size_t* num_factories) noexcept {
  {
    // Note: We can't use the EXCEPTION_TO_RETURNED_STATUS_BEGIN/EXCEPTION_TO_RETURNED_STATUS_END macros around the
    // call to `onnxruntime::ep::ApiInit()` because they depend on the API to create `OrtStatus`. We need to create an
    // `OrtStatus` more conservatively.

    // Creates an `OrtStatus` for the error or falls back to printing the error message and aborting.
    auto report_error = [](const OrtApiBase* ort_api_base, const char* message) -> OrtStatus* {
      if (ort_api_base != nullptr) {
        // Note: `OrtApi::CreateStatus` has been around since the v1 API, so we'll try to obtain it with the v1 API.
        // The `static_assert` ensures that `CreateStatus` has the same offset in the `OrtApi` struct in v1 and the
        // current version. `OrtApiBase::GetApi()` could theoretically return `OrtApi` structs with different layouts
        // for different versions, but `CreateStatus` has maintained the same offset across all versions so far.
        constexpr size_t kCreateStatusOffsetInV1Api = 0;
        static_assert(offsetof(OrtApi, CreateStatus) / sizeof(void*) == kCreateStatusOffsetInV1Api,
                      "OrtApi::CreateStatus is not at the same offset as it was in the v1 OrtApi.");
        if (const OrtApi* ort_api_v1 = ort_api_base->GetApi(1); ort_api_v1 != nullptr) {
          return ort_api_v1->CreateStatus(OrtErrorCode::ORT_FAIL, message);
        }
      }

      fprintf(stderr, "Error: %s\nUnable to use OrtApi::CreateStatus() to create an OrtStatus. Aborting.\n", message);
      std::abort();
    };

    try {
      // Manual init for the C++ API
      onnxruntime::ep::ApiInit(ort_api_base, ORT_PLUGIN_EP_MIN_ORT_VERSION);
    } catch (const std::exception& e) {
      return report_error(ort_api_base, e.what());
    } catch (...) {
      return report_error(ort_api_base, "Unknown exception");
    }
  }

  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  if (max_factories < 1) {
    return onnxruntime::ep::Api().ort.CreateStatus(ORT_INVALID_ARGUMENT,
                                                   "Not enough space to return EP factory. Need at least one.");
  }

  Ort::KeyValuePairs env_config = Ort::GetEnvConfigEntries();
  onnxruntime::webgpu::ep::Factory::Config factory_config;
  const char* allow_virtual_devices_value = env_config.GetValue(kOrtEnvAllowVirtualDevices);
  factory_config.allow_virtual_devices =
      allow_virtual_devices_value != nullptr && std::strcmp(allow_virtual_devices_value, "1") == 0;

  factory_config.allow_software_adapter =
      onnxruntime::detail::GetEnvironmentVar(kAllowSoftwareAdapterEnvironmentVariable) == "1";

  // Initialize the global default logger
  ::onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(default_logger);

  if (factory_config.allow_software_adapter) {
    LOGS_DEFAULT(WARNING) << kAllowSoftwareAdapterEnvironmentVariable
                          << "=1: WebGPU EP may use a CPU hardware device to make the WebGPU EP selectable as an ORT "
                             "EP device when no GPU hardware device is available.";
  }

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory =
      std::make_unique<onnxruntime::webgpu::ep::Factory>(factory_config);

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

EXPORT_SYMBOL OrtStatus* ORT_PLUGIN_EP_ENTRY_POINT(ReleaseEpFactory)(OrtEpFactory* factory) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // STEP.1 - Release the factory
  delete static_cast<onnxruntime::webgpu::ep::Factory*>(factory);

  // STEP.2 - Clean up cached kernel registries
  onnxruntime::webgpu::CleanupKernelRegistries();

  // STEP.3 - Clean up WebGPU contexts
  onnxruntime::webgpu::CleanupWebGpuContexts();

  // STEP.4 - Destroy the global default logger wrapper
  ::onnxruntime::ep::adapter::LoggingManager::DestroyDefaultLogger();

#if !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  // STEP.5 - Shutdown protobuf library.
  // Only do this when this EP owns the process-global state, i.e. when it is built as a separate library.
  // When statically linked into the host binary, protobuf is shared with the host, which owns its lifetime.
  google::protobuf::ShutdownProtobufLibrary();
#endif

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // extern "C"
