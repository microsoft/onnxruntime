#if defined(_WIN32)
#define ORT_TEST_EXPORT __declspec(dllexport)
#else
#define ORT_TEST_EXPORT __attribute__((visibility("default")))
#endif

namespace {
int host_call_count = 0;
}

extern "C" ORT_TEST_EXPORT int GetProviderSetHostCallCount() {
  return host_call_count;
}

#if defined(ORT_TEST_PROVIDER_SET_HOST)
extern "C" ORT_TEST_EXPORT void Provider_SetHost(void* host) {
  if (host != nullptr) {
    ++host_call_count;
  }
}
#endif