// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DLL entry points for the CUDA Plugin Execution Provider.
// Exports CreateEpFactories() and ReleaseEpFactory() as the
// public interface for ORT to load and use the CUDA EP as a plugin.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "onnxruntime_cxx_api.h"

#include "cuda_ep_factory.h"
#include "ep/api.h"  // onnxruntime::ep::ApiInit(), onnxruntime::ep::Api()

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
  // Creates an OrtStatus for the error, or falls back to printing the message and aborting.
  // We cannot use the C++ API or the EXCEPTION_TO_* helpers here because the ORT API is not
  // initialized until onnxruntime::ep::ApiInit() succeeds, and ApiInit() itself can throw before
  // that point. OrtStatus must therefore be created conservatively via the v1 API.
  auto report_error = [](const OrtApiBase* api_base, const char* message) -> OrtStatus* {
    if (api_base != nullptr) {
      // OrtApi::CreateStatus has existed since the v1 API and has kept the same offset across all
      // versions, so it is safe to obtain it via the v1 API to construct an OrtStatus.
      constexpr size_t kCreateStatusOffsetInV1Api = 0;
      static_assert(offsetof(OrtApi, CreateStatus) / sizeof(void*) == kCreateStatusOffsetInV1Api,
                    "OrtApi::CreateStatus is not at the same offset as it was in the v1 OrtApi.");
      if (const OrtApi* ort_api_v1 = api_base->GetApi(1); ort_api_v1 != nullptr) {
        return ort_api_v1->CreateStatus(ORT_FAIL, message);
      }
    }

    fprintf(stderr,
            "CUDA Plugin EP error: %s\nUnable to use OrtApi::CreateStatus() to create an OrtStatus. Aborting.\n",
            message);
    std::abort();
  };

  if (ort_api_base == nullptr) {
    return report_error(nullptr, "CUDA Plugin EP: CreateEpFactories called with a null OrtApiBase.");
  }

  // Negotiate the ORT API version with the runtime and enforce the minimum supported ORT version
  // baked in at build time from plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION (ORT_PLUGIN_EP_MIN_ORT_VERSION).
  // ApiInit() parses the runtime version string, requests the matching OrtApi for that runtime, and
  // initializes the C++ API (Ort::InitApi). This lets the same plugin binary load on older ORT
  // runtimes (>= the floor) instead of hard-requiring the API version it was compiled against.
  try {
    onnxruntime::ep::ApiInit(ort_api_base, ORT_PLUGIN_EP_MIN_ORT_VERSION);
  } catch (const std::exception& e) {
    return report_error(ort_api_base, e.what());
  } catch (...) {
    return report_error(ort_api_base, "CUDA Plugin EP: unknown error while initializing the ORT API.");
  }

  const OrtApi* ort_api = &onnxruntime::ep::Api().ort;
  const OrtEpApi* ep_api = &onnxruntime::ep::Api().ep;

  if (default_logger == nullptr) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA Plugin EP: default_logger must not be null.");
  }

  // This is a C entry point, so no C++ exception may escape it. Wrap everything that can throw
  // (e.g. std::string operations and factory construction) and report failures back to ORT as an
  // OrtStatus instead.
  try {
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
  } catch (...) {
    constexpr const char* unknown_error = "CUDA Plugin EP: unknown error in CreateEpFactories.";
    auto* log_status = ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_ERROR,
                                                  unknown_error, ORT_FILE, __LINE__, __FUNCTION__);
    if (log_status) ort_api->ReleaseStatus(log_status);
    return ort_api->CreateStatus(ORT_EP_FAIL, unknown_error);
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
