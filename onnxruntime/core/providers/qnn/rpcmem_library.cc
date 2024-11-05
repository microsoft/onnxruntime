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

SharedLibraryHandle LoadSharedLibrary(const PathString& path, bool global_symbols) {
  // Custom deleter to unload the shared library. Avoid throwing from it because it may run in dtor.
  const auto unload_shared_library = [](void* shared_library_handle) {
    if (shared_library_handle == nullptr) {
      return;
    }

    const auto& env = Env::Default();
    const auto unload_status = env.UnloadDynamicLibrary(shared_library_handle);

    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to unload shared library. Error: " << unload_status.ErrorMessage();
    }
  };

  const auto& env = Env::Default();
  void* shared_library_handle = nullptr;
  ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(path, global_symbols, &shared_library_handle));

  return SharedLibraryHandle{shared_library_handle, unload_shared_library};
}

RpcMemApi CreateApi(void* shared_library_handle) {
  RpcMemApi api{};

  const auto& env = Env::Default();
  void* symbol = nullptr;
  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(shared_library_handle, "rpcmem_alloc", &symbol));
  api.alloc = static_cast<rpcmem::AllocFnPtr>(symbol);

  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(shared_library_handle, "rpcmem_free", &symbol));
  api.free = static_cast<rpcmem::FreeFnPtr>(symbol);

  ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(shared_library_handle, "rpcmem_to_fd", &symbol));
  api.to_fd = static_cast<rpcmem::ToFdFnPtr>(symbol);

  return api;
}

}  // namespace

RpcMemLibrary::RpcMemLibrary()
    : shared_library_(LoadSharedLibrary(GetRpcMemSharedLibraryPath(), /* global_symbols */ false)),
      api_{CreateApi(shared_library_.get())} {
}

}  // namespace onnxruntime::qnn
