// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <filesystem>
#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/common/status.h"
#include "test/util/include/test/test_environment.h"
#if defined(_WIN32)
#include <windows.h>
#endif
namespace onnxruntime {
namespace test {
#if defined(_WIN32)

Status ReadEnvironmentVariable(const wchar_t* name, std::wstring& value_out) {
  const DWORD value_size = ::GetEnvironmentVariableW(name, nullptr, 0);
  ORT_RETURN_IF(value_size == 0,
                "Failed to get environment variable length. GetEnvironmentVariableW error: ", ::GetLastError());

  std::vector<wchar_t> value(value_size);

  ORT_RETURN_IF(::GetEnvironmentVariableW(name, value.data(), value_size) == 0,
                "Failed to get environment variable value. GetEnvironmentVariableW error: ", ::GetLastError());

  value_out = std::wstring{value.data()};
  return Status::OK();
}

Status GetServiceBinaryDirectoryPath(const wchar_t* service_name,
                                     std::filesystem::path& service_binary_directory_path_out) {
  struct ServiceHandleDeleter {
    void operator()(SC_HANDLE handle) { ::CloseServiceHandle(handle); }
  };

  using UniqueServiceHandle = std::unique_ptr<std::remove_pointer_t<SC_HANDLE>, ServiceHandleDeleter>;

  SC_HANDLE scm_handle_raw = ::OpenSCManagerW(nullptr,  // local computer
                                              nullptr,  // SERVICES_ACTIVE_DATABASE
                                              STANDARD_RIGHTS_READ);
  ORT_RETURN_IF(scm_handle_raw == nullptr,
                "Failed to open handle to service control manager. OpenSCManagerW error: ", ::GetLastError());

  auto scm_handle = UniqueServiceHandle{scm_handle_raw};

  SC_HANDLE service_handle_raw = ::OpenServiceW(scm_handle.get(),
                                                service_name,
                                                SERVICE_QUERY_CONFIG);
  ORT_RETURN_IF(service_handle_raw == nullptr,
                "Failed to open service handle. OpenServiceW error: ", ::GetLastError());

  auto service_handle = UniqueServiceHandle{service_handle_raw};

  // get service config required buffer size
  DWORD service_config_buffer_size{};
  ORT_RETURN_IF(!::QueryServiceConfigW(service_handle.get(), nullptr, 0, &service_config_buffer_size) &&
                    ::GetLastError() != ERROR_INSUFFICIENT_BUFFER,
                "Failed to query service configuration buffer size. QueryServiceConfigW error: ", ::GetLastError());

  // get the service config
  std::vector<std::byte> service_config_buffer(service_config_buffer_size);
  QUERY_SERVICE_CONFIGW* service_config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(service_config_buffer.data());
  ORT_RETURN_IF(!::QueryServiceConfigW(service_handle.get(), service_config, service_config_buffer_size,
                                       &service_config_buffer_size),
                "Failed to query service configuration. QueryServiceConfigW error: ", ::GetLastError());

  std::wstring service_binary_path_name = service_config->lpBinaryPathName;

  // replace system root placeholder with the value of the SYSTEMROOT environment variable
  const std::wstring system_root_placeholder = L"\\SystemRoot";

  ORT_RETURN_IF(service_binary_path_name.find(system_root_placeholder, 0) != 0,
                "Service binary path '", ToUTF8String(service_binary_path_name),
                "' does not start with expected system root placeholder value '",
                ToUTF8String(system_root_placeholder), "'.");

  std::wstring system_root{};
  ORT_RETURN_IF_ERROR(ReadEnvironmentVariable(L"SYSTEMROOT", system_root));
  service_binary_path_name.replace(0, system_root_placeholder.size(), system_root);

  const auto service_binary_path = std::filesystem::path{service_binary_path_name};
  auto service_binary_directory_path = service_binary_path.parent_path();

  ORT_RETURN_IF(!std::filesystem::exists(service_binary_directory_path),
                "Service binary directory path does not exist: ", service_binary_directory_path.string());

  service_binary_directory_path_out = std::move(service_binary_directory_path);
  return Status::OK();
}

#endif  // defined(_WIN32)

Status GetRpcMemDynamicLibraryPath(PathString& path_out) {
#if defined(_WIN32)

  std::filesystem::path qcnspmcdm_dir_path{};
  ORT_RETURN_IF_ERROR(GetServiceBinaryDirectoryPath(L"qcnspmcdm", qcnspmcdm_dir_path));
  const auto libcdsprpc_path = qcnspmcdm_dir_path / L"libcdsprpc.dll";
  path_out = libcdsprpc_path.wstring();
  return Status::OK();

#else  // ^^^ defined(_WIN32) / vvv !defined(_WIN32)

  path_out = ORT_TSTR("libcdsprpc.so");
  return Status::OK();

#endif  // !defined(_WIN32)
}

void TriggerPDReset() {
#if defined(_WIN32)
  onnxruntime::PathString rpcmem_library_path{};
  Status res = GetRpcMemDynamicLibraryPath(rpcmem_library_path);
  HMODULE lib_handle = LoadLibraryW(rpcmem_library_path.c_str());
  if (!lib_handle) {
    return;  // Failed to load library
  }
  typedef int (*RscFnHandleType_t)(uint32_t, void*, uint32_t);
  FARPROC addr = GetProcAddress(lib_handle, "remote_session_control");
  if (!addr) {
    FreeLibrary(lib_handle);
    return;  // Failed to get procedure address
  }
  RscFnHandleType_t rsc_call = reinterpret_cast<RscFnHandleType_t>(addr);
  typedef struct {
    int domain;
  } remote_rpc_process_clean_params;
  remote_rpc_process_clean_params scdata;
  scdata.domain = 3; /*CDSP_DOMAIN_ID*/
  rsc_call(/*FASTRPC_REMOTE_PROCESS_KILL*/ 6, &scdata, sizeof(remote_rpc_process_clean_params));
  if (lib_handle) {
    FreeLibrary(lib_handle);
  }
  lib_handle = nullptr;
#endif  // !defined(_WIN32)
}
}  // namespace test
}  // namespace onnxruntime