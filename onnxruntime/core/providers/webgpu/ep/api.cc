// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "core/providers/webgpu/ep/factory.h"

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
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

namespace google {
namespace protobuf {
void ShutdownProtobufLibrary();
}  // namespace protobuf
}  // namespace google

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
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
      onnxruntime::ep::ApiInit(ort_api_base);
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

  // Initialize the global default logger
  ::onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(default_logger);

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<onnxruntime::webgpu::ep::Factory>();

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // STEP.1 - Release the factory
  delete static_cast<onnxruntime::webgpu::ep::Factory*>(factory);

  // STEP.2 - Clean up cached kernel registries
  onnxruntime::webgpu::CleanupKernelRegistries();

  // STEP.3 - Clean up WebGPU contexts
  onnxruntime::webgpu::CleanupWebGpuContexts();

  // STEP.4 - Destroy the global default logger wrapper
  ::onnxruntime::ep::adapter::LoggingManager::DestroyDefaultLogger();

  // STEP.5 - Shutdown protobuf library
  google::protobuf::ShutdownProtobufLibrary();

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // extern "C"
