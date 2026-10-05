#include "core/framework/execution_provider.h"
#include "core/providers/shared_library/provider_host_api.h"

#if defined(_WIN32)
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
  void Shutdown() override {}
  void* GetInfo() override { return &initialization_count; }
} optional_provider;
}  // namespace

extern "C" ORT_TEST_EXPORT onnxruntime::Provider* GetProvider() {
  return &optional_provider;
}
extern "C" ORT_TEST_EXPORT int GetInitializationCount() {
  return optional_provider.initialization_count;
}
#else
extern "C" ORT_TEST_EXPORT int GetInitializationCount() { return 0; }
#endif
