// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Prototype plugin EP SIDE MODULE.
//
// This stands in for onnxruntime_providers_webgpu built as a plugin EP shared library.
// It mirrors the structure of onnxruntime/core/providers/webgpu/ep/api.cc:
//   - manual OrtApi init from the OrtApiBase handed in by the host
//   - CreateEpFactories / ReleaseEpFactory exported entry points
//   - an OrtEpFactory vtable implementation
//
// It additionally probes the things that are specific to (and risky under) an Emscripten
// dynamic-linking build:
//   - C++ exceptions thrown and caught inside the side module
//   - RTTI / typeid across the module boundary
//   - EM_ASM, i.e. can a side module reach JS glue (this is how emdawnwebgpu works)

#include <emscripten.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "onnxruntime_c_api.h"

namespace {

const OrtApi* g_ort = nullptr;

// Mirrors onnxruntime::ep::ApiInit()
void ApiInit(const OrtApiBase* base) {
  if (base == nullptr) {
    throw std::runtime_error("null OrtApiBase");
  }
  g_ort = base->GetApi(ORT_API_VERSION);
  if (g_ort == nullptr) {
    throw std::runtime_error("host ORT is too old for this plugin EP");
  }
  printf("[plugin] ApiInit OK, host version string = \"%s\"\n", base->GetVersionString());
}

struct PrototypeFactory : OrtEpFactory {
  PrototypeFactory() {
    std::memset(static_cast<OrtEpFactory*>(this), 0, sizeof(OrtEpFactory));
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory*) noexcept { return "WebGpuPrototype"; }
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory*) noexcept { return "Microsoft"; }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* /*this_ptr*/,
                                                         const OrtHardwareDevice* const* /*devices*/,
                                                         size_t /*num_devices*/,
                                                         OrtEpDevice** /*ep_devices*/,
                                                         size_t max_ep_devices,
                                                         size_t* num_ep_devices) noexcept {
    // Exercise a C++ exception thrown and caught entirely inside the side module,
    // then converted to an OrtStatus allocated by the MAIN module.
    try {
      if (max_ep_devices == 0) {
        throw std::runtime_error("no room for OrtEpDevice (thrown inside the side module)");
      }
      *num_ep_devices = 0;
      return nullptr;
    } catch (const std::exception& e) {
      printf("[plugin] caught %s inside side module: %s\n", typeid(e).name(), e.what());
      return g_ort->CreateStatus(ORT_EP_FAIL, e.what());
    }
  }
};

}  // namespace

// Optional ballast so the side module exceeds Chromium's 4KB synchronous-compile limit for
// `new WebAssembly.Module()` on the main thread. A real WebGPU plugin EP is megabytes, so the
// padded variant is the realistic one.
#ifdef PROTOTYPE_PAD_SIDE_MODULE
extern "C" __attribute__((visibility("default"))) int PrototypePadding(int x) {
#define PAD_STEP(i) x = (x * 31 + (i)) ^ (x >> 3);
#define PAD_10(b) PAD_STEP(b + 0) PAD_STEP(b + 1) PAD_STEP(b + 2) PAD_STEP(b + 3) PAD_STEP(b + 4) \
                  PAD_STEP(b + 5) PAD_STEP(b + 6) PAD_STEP(b + 7) PAD_STEP(b + 8) PAD_STEP(b + 9)
#define PAD_100(b) PAD_10(b + 0) PAD_10(b + 10) PAD_10(b + 20) PAD_10(b + 30) PAD_10(b + 40) \
                   PAD_10(b + 50) PAD_10(b + 60) PAD_10(b + 70) PAD_10(b + 80) PAD_10(b + 90)
#define PAD_1000(b) PAD_100(b + 0) PAD_100(b + 100) PAD_100(b + 200) PAD_100(b + 300) PAD_100(b + 400) \
                    PAD_100(b + 500) PAD_100(b + 600) PAD_100(b + 700) PAD_100(b + 800) PAD_100(b + 900)
  PAD_1000(0) PAD_1000(1000) PAD_1000(2000) PAD_1000(3000) PAD_1000(4000)
  return x;
}
#endif

extern "C" {

__attribute__((visibility("default")))
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* /*default_logger*/,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories) noexcept {
  printf("[plugin] CreateEpFactories(\"%s\") entered in the SIDE MODULE\n", registration_name);

  try {
    ApiInit(ort_api_base);
  } catch (const std::exception& e) {
    if (ort_api_base != nullptr) {
      if (const OrtApi* v1 = ort_api_base->GetApi(1); v1 != nullptr) {
        return v1->CreateStatus(ORT_FAIL, e.what());
      }
    }
    fprintf(stderr, "[plugin] fatal: %s\n", e.what());
    return nullptr;
  }

  // Probe: can a side module reach JS glue? emdawnwebgpu is implemented as JS library code,
  // so this determines whether the WebGPU EP's Dawn layer could live in the side module.
  int has_webgpu = EM_ASM_INT({
    if (typeof navigator !== 'undefined' && navigator.gpu) return 1;
    return 0;
  });
  printf("[plugin] EM_ASM from side module works; navigator.gpu present = %d\n", has_webgpu);

  if (max_factories < 1) {
    return g_ort->CreateStatus(ORT_INVALID_ARGUMENT, "need at least one factory slot");
  }

  // Allocated in the SIDE module and destroyed in the SIDE module (in ReleaseEpFactory), but
  // held by the MAIN module in between. This is the ownership pattern the EP ABI uses
  // throughout -- no heap ownership crosses the module boundary.
  auto factory = std::make_unique<PrototypeFactory>();
  factories[0] = factory.release();
  *num_factories = 1;

  printf("[plugin] CreateEpFactories OK\n");
  return nullptr;
}

__attribute__((visibility("default")))
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) noexcept {
  printf("[plugin] ReleaseEpFactory entered in the SIDE MODULE\n");
  delete static_cast<PrototypeFactory*>(factory);
  return nullptr;
}

}  // extern "C"
