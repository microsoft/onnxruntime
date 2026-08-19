// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Prototype "mini onnxruntime-web" MAIN MODULE.
//
// This stands in for onnxruntime.wasm. It implements the small slice of ORT that a plugin EP
// library actually touches at load time:
//
//   OrtApiBase / OrtApi                     -> handed to the EP library
//   RegisterExecutionProviderLibrary(...)   -> dlopen + dlsym("CreateEpFactories")
//   UnregisterExecutionProviderLibrary(...) -> ReleaseEpFactory + dlclose
//
// It deliberately uses the *real* ONNX Runtime public headers so the ABI exercised here is the
// same ABI a real plugin EP uses.

#include <dlfcn.h>
#include <emscripten.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace {

// ---------------------------------------------------------------------------
// Minimal OrtStatus + OrtApi implementation living in the MAIN module.
// The side module will call these through function pointers, which is what
// makes this a real cross-module (main <- side) call test.
// ---------------------------------------------------------------------------

struct MiniStatus {
  OrtErrorCode code;
  std::string msg;
};

OrtStatus* ORT_API_CALL MiniCreateStatus(OrtErrorCode code, const char* msg) noexcept {
  // Allocated in the MAIN module, freed in the MAIN module, but the pointer travels
  // through the side module. Exercises a shared heap.
  auto* s = new MiniStatus{code, msg ? msg : ""};
  return reinterpret_cast<OrtStatus*>(s);
}

OrtErrorCode ORT_API_CALL MiniGetErrorCode(const OrtStatus* status) noexcept {
  return reinterpret_cast<const MiniStatus*>(status)->code;
}

const char* ORT_API_CALL MiniGetErrorMessage(const OrtStatus* status) noexcept {
  return reinterpret_cast<const MiniStatus*>(status)->msg.c_str();
}

void ORT_API_CALL MiniReleaseStatus(OrtStatus* status) noexcept {
  delete reinterpret_cast<MiniStatus*>(status);
}

OrtApi g_mini_api{};

const OrtApi* ORT_API_CALL MiniGetApi(uint32_t version) noexcept {
  printf("[host] side module asked for OrtApi version %u (host ORT_API_VERSION=%d)\n",
         version, ORT_API_VERSION);
  if (version > ORT_API_VERSION) {
    return nullptr;
  }
  return &g_mini_api;
}

const char* ORT_API_CALL MiniGetVersionString() noexcept {
  return "1.24.0-wasm-plugin-ep-prototype";
}

OrtApiBase g_mini_api_base{MiniGetApi, MiniGetVersionString};

void InitMiniApi() {
  g_mini_api.CreateStatus = MiniCreateStatus;
  g_mini_api.GetErrorCode = MiniGetErrorCode;
  g_mini_api.GetErrorMessage = MiniGetErrorMessage;
  g_mini_api.ReleaseStatus = MiniReleaseStatus;
}

// ---------------------------------------------------------------------------
// The registered EP library, mirroring onnxruntime::EpLibraryPlugin.
// ---------------------------------------------------------------------------

struct RegisteredEpLibrary {
  std::string registration_name;
  void* handle = nullptr;
  CreateEpApiFactoriesFn create_fn = nullptr;
  ReleaseEpApiFactoryFn release_fn = nullptr;
  std::vector<OrtEpFactory*> factories;
};

RegisteredEpLibrary g_lib;

}  // namespace

extern "C" {

// Mirrors OrtApi::RegisterExecutionProviderLibrary.
EMSCRIPTEN_KEEPALIVE int OrtRegisterExecutionProviderLibrary(const char* registration_name,
                                                             const char* library_path) {
  InitMiniApi();

  printf("[host] dlopen(\"%s\")\n", library_path);
  void* handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    printf("[host] ERROR dlopen failed: %s\n", dlerror());
    return 1;
  }
  printf("[host] dlopen OK, handle=%p\n", handle);

  auto create_fn = reinterpret_cast<CreateEpApiFactoriesFn>(dlsym(handle, "CreateEpFactories"));
  auto release_fn = reinterpret_cast<ReleaseEpApiFactoryFn>(dlsym(handle, "ReleaseEpFactory"));
  if (create_fn == nullptr || release_fn == nullptr) {
    printf("[host] ERROR dlsym failed: %s\n", dlerror());
    dlclose(handle);
    return 2;
  }
  printf("[host] dlsym CreateEpFactories=%p ReleaseEpFactory=%p\n",
         reinterpret_cast<void*>(create_fn), reinterpret_cast<void*>(release_fn));

  OrtEpFactory* factories[4] = {nullptr, nullptr, nullptr, nullptr};
  size_t num_factories = 0;

  // NOTE: passing nullptr for the OrtLogger; the prototype EP does not use it.
  OrtStatus* status = create_fn(registration_name, &g_mini_api_base, nullptr,
                                factories, 4, &num_factories);
  if (status != nullptr) {
    printf("[host] ERROR CreateEpFactories: %s\n", MiniGetErrorMessage(status));
    MiniReleaseStatus(status);
    dlclose(handle);
    return 3;
  }

  printf("[host] CreateEpFactories returned %zu factor%s\n", num_factories,
         num_factories == 1 ? "y" : "ies");

  g_lib.registration_name = registration_name;
  g_lib.handle = handle;
  g_lib.create_fn = create_fn;
  g_lib.release_fn = release_fn;
  g_lib.factories.assign(factories, factories + num_factories);

  for (size_t i = 0; i < num_factories; ++i) {
    OrtEpFactory* f = g_lib.factories[i];
    // main -> side calls through the OrtEpFactory vtable.
    printf("[host]   factory[%zu]: name=\"%s\" vendor=\"%s\" ort_version_supported=%u\n",
           i, f->GetName(f), f->GetVendor(f), f->ort_version_supported);
    if (f->ort_version_supported > static_cast<uint32_t>(ORT_API_VERSION)) {
      printf("[host]   ERROR EP compiled against newer ORT than host\n");
      return 4;
    }
  }

  // Shared-heap probe: the SIDE module allocates, the MAIN module reads and frees.
  auto alloc_fn = reinterpret_cast<void* (*)(size_t)>(dlsym(handle, "PrototypeAllocForHost"));
  if (alloc_fn == nullptr) {
    printf("[host] ERROR dlsym PrototypeAllocForHost failed: %s\n", dlerror());
    return 5;
  }
  constexpr size_t kProbeSize = 64;
  auto* probe = static_cast<unsigned char*>(alloc_fn(kProbeSize));
  if (probe == nullptr) {
    printf("[host] ERROR side module allocation returned null\n");
    return 6;
  }
  for (size_t i = 0; i < kProbeSize; ++i) {
    if (probe[i] != 0xAB) {
      printf("[host] ERROR side-module buffer corrupt at byte %zu: 0x%02X\n", i, probe[i]);
      return 7;
    }
  }
  printf("[host] read %zu bytes allocated by the side module at %p; freeing in main module\n",
         kProbeSize, static_cast<void*>(probe));
  std::free(probe);  // side module malloc'd it, main module frees it -> one shared allocator
  printf("[host] free() of side-module allocation OK\n");

  return 0;
}

// Ask the plugin to deliberately fail so we can verify that an OrtStatus created in the
// MAIN module by side-module code (and a C++ exception thrown/caught inside the side
// module) survives the module boundary.
EMSCRIPTEN_KEEPALIVE int OrtTestErrorPath() {
  if (g_lib.factories.empty()) {
    return -1;
  }
  OrtEpFactory* f = g_lib.factories[0];
  size_t n = 0;
  // max_ep_devices == 0 makes the prototype EP return an error status.
  OrtStatus* status = f->GetSupportedDevices(f, nullptr, 0, nullptr, 0, &n);
  if (status == nullptr) {
    printf("[host] ERROR expected a failure status\n");
    return -2;
  }
  printf("[host] got OrtStatus from side module: code=%d msg=\"%s\"\n",
         MiniGetErrorCode(status), MiniGetErrorMessage(status));
  MiniReleaseStatus(status);  // allocated by main, routed through side, freed by main
  return 0;
}

EMSCRIPTEN_KEEPALIVE int OrtUnregisterExecutionProviderLibrary() {
  if (g_lib.handle == nullptr) {
    return -1;
  }
  for (OrtEpFactory* f : g_lib.factories) {
    OrtStatus* status = g_lib.release_fn(f);
    if (status != nullptr) {
      printf("[host] ERROR ReleaseEpFactory: %s\n", MiniGetErrorMessage(status));
      MiniReleaseStatus(status);
      return -2;
    }
  }
  g_lib.factories.clear();
  printf("[host] ReleaseEpFactory OK, dlclose\n");
  dlclose(g_lib.handle);
  g_lib.handle = nullptr;
  return 0;
}

}  // extern "C"

int main() {
  printf("[host] main module ready (ORT_API_VERSION=%d)\n", ORT_API_VERSION);

  // The real onnxruntime.wasm uses both C++ exceptions and EM_ASM. With -sMAIN_MODULE=2 the
  // main module is dead-code-eliminated, so the Emscripten EH globals (__THREW__, invoke_*
  // trampolines) and the EM_ASM support function only exist if the MAIN module itself pulls
  // them in. A side module that throws or uses EM_ASM cannot add them after the fact.
  try {
    if (getenv("__never_set__") != nullptr) {
      throw std::runtime_error("unreachable");
    }
  } catch (const std::exception&) {
  }
  int probe = EM_ASM_INT({ return 1; });
  (void)probe;
  return 0;
}
