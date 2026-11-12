// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#if defined(ORT_TEST_OPTIONAL_PROVIDER_valid)
#include "core/framework/execution_provider.h"
#include "core/providers/shared_library/provider_host_api.h"
#endif

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#define ORT_TEST_EXPORT __declspec(dllexport)
#else
#define ORT_TEST_EXPORT __attribute__((visibility("default")))
#endif

#if defined(ORT_TEST_OPTIONAL_PROVIDER_shared)
extern "C" ORT_TEST_EXPORT void Provider_SetHost(void*) {}
#elif defined(ORT_TEST_OPTIONAL_PROVIDER_valid)
namespace {
struct OptionalProvider : onnxruntime::Provider {
  int initialization_count = 0;

  void Initialize() override { ++initialization_count; }
  void Shutdown() override { initialization_count = 0; }
  void* GetInfo() override { return &initialization_count; }
} optional_provider;
}  // namespace

extern "C" ORT_TEST_EXPORT onnxruntime::Provider* GetProvider() {
  return &optional_provider;
}
extern "C" ORT_TEST_EXPORT int GetInitializationCount() {
  return optional_provider.initialization_count;
}
#elif defined(ORT_TEST_OPTIONAL_PROVIDER_plugin) || defined(ORT_TEST_OPTIONAL_PROVIDER_plugin_missing_export)
namespace {
// Store events outside the DLL so an unload/reload cannot reset the observed history.
// The fixture must unload synchronously: do not add GNU-unique symbols, TLS destructors, or RTLD_NODELETE.
struct PluginLifetimeTracker {
  PluginLifetimeTracker() { Record('L'); }
  ~PluginLifetimeTracker() { Record('U'); }

  static void Record(char event) {
    constexpr const char* name = "ORT_TEST_PLUGIN_LIFETIME_EVENTS";
#if defined(_WIN32)
    const DWORD required_size = GetEnvironmentVariableA(name, nullptr, 0);
    if (required_size == 0) {
      return;
    }
    std::string events(required_size, '\0');
    SetLastError(ERROR_SUCCESS);
    const DWORD length = GetEnvironmentVariableA(name, events.data(), required_size);
    if (length >= required_size || (length == 0 && GetLastError() != ERROR_SUCCESS)) {
      std::fputs("Failed to read optional-provider load history.\n", stderr);
      std::abort();
    }
    events.resize(length);
    events.push_back(event);
    if (!SetEnvironmentVariableA(name, events.c_str())) {
      std::fputs("Failed to update optional-provider load history.\n", stderr);
      std::abort();
    }
#else
    if (const char* events = std::getenv(name)) {
      const std::string updated = std::string{events} + event;
      setenv(name, updated.c_str(), 1);
    }
#endif
  }
} plugin_lifetime_tracker;

const char* ORT_API_CALL GetPluginName(const OrtEpFactory*) noexcept {
  return "ProbePlugin";
}

OrtStatus* ORT_API_CALL GetSupportedDevices(OrtEpFactory*, const OrtHardwareDevice* const*, size_t,
                                            OrtEpDevice**, size_t, size_t* num_devices) noexcept {
  *num_devices = 0;
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateDataTransfer(OrtEpFactory*, OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;
  return nullptr;
}
}  // namespace

extern "C" ORT_TEST_EXPORT OrtStatus* CreateEpFactories(
    const char* registration_name, const OrtApiBase* api_base, const OrtLogger*,
    OrtEpFactory** factories, size_t, size_t* num_factories) {
  if (std::string{registration_name} == "fail_create_factories") {
    return api_base->GetApi(ORT_API_VERSION)->CreateStatus(ORT_FAIL, "Factory creation failed.");
  }
  auto factory = std::make_unique<OrtEpFactory>();
  factory->ort_version_supported = ORT_API_VERSION;
  factory->GetName = GetPluginName;
  factory->GetVendor = GetPluginName;
  factory->GetSupportedDevices = GetSupportedDevices;
  factory->CreateDataTransfer = CreateDataTransfer;
  factories[0] = factory.release();
  *num_factories = 1;
  return nullptr;
}

#if defined(ORT_TEST_OPTIONAL_PROVIDER_plugin)
extern "C" ORT_TEST_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}
#endif
#else
extern "C" ORT_TEST_EXPORT int GetInitializationCount() { return 0; }
#endif
