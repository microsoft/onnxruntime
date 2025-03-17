// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/rpcmem_library.h"

#include "core/common/logging/logging.h"
#include "core/platform/env.h"

namespace onnxruntime::qnn {

namespace {

const PathChar* GetRpcMemSharedLibraryPath() {
#if defined(_WIN32)
  return ORT_TSTR("libcdsprpc.dll");
#else
  return ORT_TSTR("libcdsprpc.so");
#endif
}

DynamicLibraryHandle LoadDynamicLibrary(const PathString& path, bool global_symbols) {
  // Custom deleter to unload the shared library. Avoid throwing from it because it may run in dtor.
  const auto unload_library = [](void* library_handle) {
    if (library_handle == nullptr) {
      return;
    }

    const auto& env = Env::Default();
    const auto unload_status = env.UnloadDynamicLibrary(library_handle);

    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to unload shared library. Error: " << unload_status.ErrorMessage();
    }
  };

  const auto& env = Env::Default();
  void* library_handle = nullptr;

  const auto load_status = env.LoadDynamicLibrary(path, global_symbols, &library_handle);
  if (!load_status.IsOK()) {
    ORT_THROW("Failed to load ", ToUTF8String(path), ": ", load_status.ErrorMessage());
  }

  return DynamicLibraryHandle{library_handle, unload_library};
}

RpcMemApi CreateApi(void* library_handle) {
  RpcMemApi api{};

  const auto& env = Env::Default();
  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(library_handle, "rpcmem_alloc", (void**)&api.alloc));

  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(library_handle, "rpcmem_free", (void**)&api.free));

  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(library_handle, "rpcmem_to_fd", (void**)&api.to_fd));

  return api;
}

}  // namespace

RpcMemLibrary::RpcMemLibrary()
    : library_handle_(LoadDynamicLibrary(GetRpcMemSharedLibraryPath(), /* global_symbols */ false)),
      api_{CreateApi(library_handle_.get())} {
}

}  // namespace onnxruntime::qnn
