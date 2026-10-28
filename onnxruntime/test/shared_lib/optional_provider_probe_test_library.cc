#include "core/framework/execution_provider.h"
#include "core/providers/shared_library/provider_host_api.h"

#include <cstdlib>
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
struct PluginLifetimeTracker {
  PluginLifetimeTracker() { Record('L'); }
  ~PluginLifetimeTracker() { Record('U'); }

  static void Record(char event) {
    constexpr const char* name = "ORT_TEST_PLUGIN_LIFETIME_EVENTS";
#if defined(_WIN32)
    char events[64]{};
    const DWORD length = GetEnvironmentVariableA(name, events, sizeof(events));
    if (length == 0 || length >= sizeof(events) - 1) {
      return;
    }
    events[length] = event;
    SetEnvironmentVariableA(name, events);
#else
    if (const char* events = std::getenv(name)) {
      const std::string updated = std::string{events} + event;
      setenv(name, updated.c_str(), 1);
    }
#endif
  }
} plugin_lifetime_tracker;
}  // namespace

extern "C" ORT_TEST_EXPORT OrtStatus* CreateEpFactories(
    const char* registration_name, const OrtApiBase* api_base, const OrtLogger*,
    OrtEpFactory**, size_t, size_t* num_factories) {
  if (std::string{registration_name} == "fail_create_factories") {
    return api_base->GetApi(ORT_API_VERSION)->CreateStatus(ORT_FAIL, "Factory creation failed.");
  }
  *num_factories = 0;
  return nullptr;
}

#if defined(ORT_TEST_OPTIONAL_PROVIDER_plugin)
extern "C" ORT_TEST_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory*) {
  return nullptr;
}
#endif
#else
extern "C" ORT_TEST_EXPORT int GetInitializationCount() { return 0; }
#endif
