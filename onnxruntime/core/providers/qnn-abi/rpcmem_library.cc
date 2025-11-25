// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn-abi/rpcmem_library.h"

#if defined(_WIN32)
#include <filesystem>

#include <Windows.h>
#include <sysinfoapi.h>
#include <winsvc.h>
#endif  // defined(_WIN32)

#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime::qnn {

// Unload the dynamic library referenced by `library_handle`.
// Avoid throwing because this may run from a dtor.
void DynamicLibraryHandleDeleter::operator()(void* library_handle) noexcept {
  if (library_handle == nullptr) {
    return;
  }

  const auto unload_status = OrtUnloadDynamicLibrary(library_handle);

  if (!unload_status.IsOK()) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_WARNING,
                ("Failed to unload dynamic library. Error: " + unload_status.GetErrorMessage()).c_str());
  }
}

namespace {

#if defined(_WIN32)

struct ServiceHandleDeleter {
  void operator()(SC_HANDLE handle) { ::CloseServiceHandle(handle); }
};

using UniqueServiceHandle = std::unique_ptr<std::remove_pointer_t<SC_HANDLE>, ServiceHandleDeleter>;

Ort::Status ReadEnvironmentVariable(const wchar_t* name, std::wstring& value_out) {
  const DWORD value_size = ::GetEnvironmentVariableW(name, nullptr, 0);
  RETURN_IF(value_size == 0,
            ("Failed to get environment variable length. GetEnvironmentVariableW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  std::vector<wchar_t> value(value_size);

  RETURN_IF(::GetEnvironmentVariableW(name, value.data(), value_size) == 0,
            ("Failed to get environment variable value. GetEnvironmentVariableW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  value_out = std::wstring{value.data()};
  return Ort::Status();
}

Ort::Status GetServiceBinaryDirectoryPath(const wchar_t* service_name,
                                          std::filesystem::path& service_binary_directory_path_out) {
  SC_HANDLE scm_handle_raw = ::OpenSCManagerW(nullptr,  // local computer
                                              nullptr,  // SERVICES_ACTIVE_DATABASE
                                              STANDARD_RIGHTS_READ);
  RETURN_IF(scm_handle_raw == nullptr,
            ("Failed to open handle to service control manager. OpenSCManagerW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  auto scm_handle = UniqueServiceHandle{scm_handle_raw};

  SC_HANDLE service_handle_raw = ::OpenServiceW(scm_handle.get(),
                                                service_name,
                                                SERVICE_QUERY_CONFIG);
  RETURN_IF(service_handle_raw == nullptr,
            ("Failed to open service handle. OpenServiceW error: " + std::to_string(::GetLastError())).c_str());

  auto service_handle = UniqueServiceHandle{service_handle_raw};

  // get service config required buffer size
  DWORD service_config_buffer_size{};
  RETURN_IF(!::QueryServiceConfigW(service_handle.get(), nullptr, 0, &service_config_buffer_size) &&
                ::GetLastError() != ERROR_INSUFFICIENT_BUFFER,
            ("Failed to query service configuration buffer size. QueryServiceConfigW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  // get the service config
  std::vector<std::byte> service_config_buffer(service_config_buffer_size);
  QUERY_SERVICE_CONFIGW* service_config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(service_config_buffer.data());
  RETURN_IF(!::QueryServiceConfigW(service_handle.get(), service_config, service_config_buffer_size,
                                   &service_config_buffer_size),
            ("Failed to query service configuration. QueryServiceConfigW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  std::wstring service_binary_path_name = service_config->lpBinaryPathName;

  // replace system root placeholder with the value of the SYSTEMROOT environment variable
  const std::wstring system_root_placeholder = L"\\SystemRoot";

  RETURN_IF(service_binary_path_name.find(system_root_placeholder, 0) != 0,
            ("Service binary path '" + std::filesystem::path(service_binary_path_name).string() +
             "' does not start with expected system root placeholder value '" +
             std::filesystem::path(system_root_placeholder).string() + "'.")
                .c_str());

  std::wstring system_root{};
  RETURN_IF_ERROR(ReadEnvironmentVariable(L"SYSTEMROOT", system_root));
  service_binary_path_name.replace(0, system_root_placeholder.size(), system_root);

  const auto service_binary_path = std::filesystem::path{service_binary_path_name};
  auto service_binary_directory_path = service_binary_path.parent_path();

  RETURN_IF(!std::filesystem::exists(service_binary_directory_path),
            ("Service binary directory path does not exist: " + service_binary_directory_path.string()).c_str());

  service_binary_directory_path_out = std::move(service_binary_directory_path);
  return Ort::Status();
}

#endif  // defined(_WIN32)

Ort::Status GetRpcMemDynamicLibraryPath(std::basic_string<ORTCHAR_T>& path_out) {
#if defined(_WIN32)

  std::filesystem::path qcnspmcdm_dir_path{};
  RETURN_IF_ERROR(GetServiceBinaryDirectoryPath(L"qcnspmcdm", qcnspmcdm_dir_path));
  const auto libcdsprpc_path = qcnspmcdm_dir_path / L"libcdsprpc.dll";
  path_out = libcdsprpc_path.wstring();
  return Ort::Status();

#else  // ^^^ defined(_WIN32) / vvv !defined(_WIN32)

  path_out = ORT_TSTR("libcdsprpc.so");
  return Ort::Status();

#endif  // !defined(_WIN32)
}

Ort::Status LoadDynamicLibrary(const std::basic_string<ORTCHAR_T>& path, bool global_symbols,
                               UniqueDynamicLibraryHandle& library_handle_out) {
  void* library_handle_raw = nullptr;
  RETURN_IF_ERROR(OrtLoadDynamicLibrary(path, global_symbols, &library_handle_raw));

  library_handle_out = UniqueDynamicLibraryHandle{library_handle_raw};
  return Ort::Status();
}

UniqueDynamicLibraryHandle GetRpcMemDynamicLibraryHandle() {
  const std::string error_message_prefix = "Failed to initialize RPCMEM dynamic library handle: ";

  std::basic_string<ORTCHAR_T> rpcmem_library_path{};
  auto status = GetRpcMemDynamicLibraryPath(rpcmem_library_path);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW(error_message_prefix + status.GetErrorMessage(), ORT_RUNTIME_EXCEPTION);
  }

  UniqueDynamicLibraryHandle library_handle{};
  status = LoadDynamicLibrary(rpcmem_library_path, /* global_symbols */ false, library_handle);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW(error_message_prefix + status.GetErrorMessage(), ORT_RUNTIME_EXCEPTION);
  }

  return library_handle;
}

RpcMemApi CreateApi(void* library_handle) {
  RpcMemApi api{};

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_alloc", (void**)&api.alloc).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_alloc.", ORT_RUNTIME_EXCEPTION);
  }

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_free", (void**)&api.free).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_free.", ORT_RUNTIME_EXCEPTION);
  }

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_to_fd", (void**)&api.to_fd).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_to_fd.", ORT_RUNTIME_EXCEPTION);
  }

  return api;
}

}  // namespace

RpcMemLibrary::RpcMemLibrary()
    : library_handle_(GetRpcMemDynamicLibraryHandle()),
      api_{CreateApi(library_handle_.get())} {
}

}  // namespace onnxruntime::qnn
