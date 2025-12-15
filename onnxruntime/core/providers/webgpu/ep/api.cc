// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

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
}  // namespace webgpu
}  // namespace onnxruntime

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  // Manual init for the C++ API
  onnxruntime::ep::ApiInit(ort_api_base);

  if (max_factories < 1) {
    return onnxruntime::ep::Api().ort.CreateStatus(ORT_INVALID_ARGUMENT,
                                                   "Not enough space to return EP factory. Need at least one.");
  }

  // Initialize the global default logger
  ::onnxruntime::ep::detail::Logger::CreateDefaultLogger(default_logger);

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<onnxruntime::webgpu::ep::Factory>();

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<onnxruntime::webgpu::ep::Factory*>(factory);
  onnxruntime::webgpu::CleanupWebGpuContexts();
  return nullptr;
}

}  // extern "C"
