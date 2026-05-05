// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DLL entry points for the CUDA Plugin Execution Provider.
// Exports CreateEpFactories() and ReleaseEpFactory() as the
// public interface for ORT to load and use the CUDA EP as a plugin.

#include "onnxruntime_cxx_api.h"

#include "cuda_ep_factory.h"

#ifndef _WIN32
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {

/// Create the CUDA EP factory instances.
/// Called by ORT when loading the CUDA plugin EP DLL.
EXPORT_SYMBOL OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // Initialize the C++ API FIRST before any C++ wrapper usage
  Ort::InitApi(ort_api);

  if (default_logger == nullptr) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA Plugin EP: default_logger must not be null.");
  }

  // Log initialization (default_logger is guaranteed non-null after the check above).
  {
    std::string msg = "CreateEpFactories: Initializing CUDA Plugin EP with registration name: ";
    msg += (registration_name ? registration_name : "NULL");
    auto* status = ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_INFO,
                                              msg.c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__);
    if (status) ort_api->ReleaseStatus(status);
  }

  if (max_factories < 1) {
    auto* log_status = ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_ERROR,
                                                  "CreateEpFactories: max_factories < 1",
                                                  ORT_FILE, __LINE__, __FUNCTION__);
    if (log_status) ort_api->ReleaseStatus(log_status);
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA Plugin EP: Not enough space to return EP factory. Need at least one.");
  }

  try {
    auto factory = std::make_unique<onnxruntime::cuda_plugin::CudaEpFactory>(
        *ort_api, *ep_api, *default_logger);

    factories[0] = factory.release();
    *num_factories = 1;

    auto* log_status = ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_INFO,
                                                  "CreateEpFactories: Successfully created CUDA EP factory",
                                                  ORT_FILE, __LINE__, __FUNCTION__);
    if (log_status) ort_api->ReleaseStatus(log_status);
  } catch (const std::exception& ex) {
    auto* log_status = ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_ERROR,
                                                  ex.what(), ORT_FILE, __LINE__, __FUNCTION__);
    if (log_status) ort_api->ReleaseStatus(log_status);
    return ort_api->CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/// Release a CUDA EP factory instance.
/// Called by ORT when unloading the CUDA plugin EP DLL.
EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<onnxruntime::cuda_plugin::CudaEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
