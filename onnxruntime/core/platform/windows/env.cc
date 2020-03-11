/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env.h"

#include <Shlwapi.h>
#include <Windows.h>
#include <shellapi.h>

#include <fstream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/platform/scoped_resource.h"
#include "core/platform/windows/telemetry.h"

namespace onnxruntime {

namespace {

struct FileHandleTraits {
  using Handle = HANDLE;
  static Handle GetInvalidHandleValue() noexcept { return INVALID_HANDLE_VALUE; }
  static void CleanUp(Handle h) noexcept {
    if (!CloseHandle(h)) {
      const int err = GetLastError();
      //It indicates potential data loss
      LOGS_DEFAULT(ERROR) << "Failed to close file handle - error code: " << err;
    }
  }
};

// Note: File handle cleanup may fail but this class doesn't expose a way to check if it failed.
//       If that's important, consider using another cleanup method.
using ScopedFileHandle = ScopedResource<FileHandleTraits>;

class WindowsEnv : public Env {
 public:
  void SleepForMicroseconds(int64_t micros) const override { Sleep(static_cast<DWORD>(micros) / 1000); }

  int GetNumCpuCores() const override {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD returnLength = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
      // try GetSystemInfo
      SYSTEM_INFO sysInfo;
      GetSystemInfo(&sysInfo);
      if (sysInfo.dwNumberOfProcessors <= 0) {
        ORT_THROW("Fatal error: 0 count processors from GetSystemInfo");
      }
      // This is the number of logical processors in the current group
      return sysInfo.dwNumberOfProcessors;
    }
    int processorCoreCount = 0;
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ++processorCoreCount;
      }
    }
    if (!processorCoreCount) ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
    return processorCoreCount;
  }

  static WindowsEnv& Instance() {
    static WindowsEnv default_env;
    return default_env;
  }

  PIDType GetSelfPid() const override {
    return GetCurrentProcessId();
  }

  Status GetFileLength(const ORTCHAR_T* file_path, size_t& length) const override {
    ScopedFileHandle file_handle{CreateFileW(
        file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
    LARGE_INTEGER filesize;
    if (!GetFileSizeEx(file_handle.Get(), &filesize)) {
      const int err = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileSizeEx ", ToMBString(file_path), " fail, errcode = ", err);
    }
    if (static_cast<ULONGLONG>(filesize.QuadPart) > std::numeric_limits<size_t>::max()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileLength: File is too large");
    }
    length = static_cast<size_t>(filesize.QuadPart);
    return Status::OK();
  }

  Status ReadFileIntoBuffer(
      const ORTCHAR_T* const file_path, const FileOffsetType offset, const size_t length,
      const gsl::span<char> buffer) const override {
    ORT_RETURN_IF_NOT(file_path);
    ORT_RETURN_IF_NOT(offset >= 0);
    ORT_RETURN_IF_NOT(length <= buffer.size());

    ScopedFileHandle file_handle{CreateFileW(
        file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
    if (!file_handle.IsValid()) {
      const int err = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToMBString(file_path), " fail, errcode = ", err);
    }

    if (length == 0) return Status::OK();

    if (offset > 0) {
      LARGE_INTEGER current_position;
      current_position.QuadPart = offset;
      if (!SetFilePointerEx(file_handle.Get(), current_position, &current_position, FILE_BEGIN)) {
        const int err = GetLastError();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SetFilePointerEx ", ToMBString(file_path), " fail, errcode = ", err);
      }
    }

    size_t total_bytes_read = 0;
    while (total_bytes_read < length) {
      constexpr DWORD k_max_bytes_to_read = 1 << 30;  // read at most 1GB each time
      const size_t bytes_remaining = length - total_bytes_read;
      const DWORD bytes_to_read = static_cast<DWORD>(std::min<size_t>(bytes_remaining, k_max_bytes_to_read));
      DWORD bytes_read;

      if (!ReadFile(
              file_handle.Get(), buffer.data() + total_bytes_read, bytes_to_read, &bytes_read, nullptr)) {
        const int err = GetLastError();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToMBString(file_path), " fail, errcode = ", err);
      }

      if (bytes_read != bytes_to_read) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToMBString(file_path), " fail: unexpected end");
      }

      total_bytes_read += bytes_read;
    }

    return Status::OK();
  }

  Status MapFileIntoMemory(
      const ORTCHAR_T*, FileOffsetType, size_t,
      MappedMemoryPtr&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "MapFileIntoMemory is not implemented on Windows.");
  }

  bool FolderExists(const std::wstring& path) const override {
    DWORD attributes = GetFileAttributesW(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY);
  }

  bool FolderExists(const std::string& path) const override {
    DWORD attributes = GetFileAttributesA(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY);
  }

  common::Status CreateFolder(const std::wstring& path) const override {
    size_t pos = 0;
    do {
      pos = path.find_first_of(L"\\/", pos + 1);
      std::wstring directory = path.substr(0, pos);
      if (FolderExists(directory)) {
        continue;
      }
      if (CreateDirectoryW(directory.c_str(), NULL) == 0) {
        return common::Status(common::SYSTEM, errno);
      }
    } while (pos != std::string::npos);
    return Status::OK();
  }

  common::Status CreateFolder(const std::string& path) const override {
    size_t pos = 0;
    do {
      pos = path.find_first_of("\\/", pos + 1);
      std::string directory = path.substr(0, pos);
      if (FolderExists(directory)) {
        continue;
      }
      if (CreateDirectoryA(directory.c_str(), NULL) == 0) {
        return common::Status(common::SYSTEM, errno);
      }
    } while (pos != std::string::npos);
    return Status::OK();
  }

  common::Status DeleteFolder(const std::wstring& path) const override {
    // SHFileOperation() will also delete files, so check for directory first
    ORT_RETURN_IF_NOT(FolderExists(path), "Directory does not exist: ", ToMBString(path));

    const std::wstring path_ending_with_double_null = path + L'\0';
    SHFILEOPSTRUCTW sh_file_op{};
    sh_file_op.wFunc = FO_DELETE;
    sh_file_op.pFrom = path_ending_with_double_null.c_str();
    sh_file_op.fFlags = FOF_NO_UI;

    const auto result = ::SHFileOperationW(&sh_file_op);
    ORT_RETURN_IF_NOT(result == 0, "SHFileOperation() failed with error: ", result);

    ORT_RETURN_IF_NOT(
        !sh_file_op.fAnyOperationsAborted,
        "SHFileOperation() indicated that an operation was aborted.");

    return Status::OK();
  }

  common::Status DeleteFolder(const std::string& path) const override {
    // SHFileOperation() will also delete files, so check for directory first
    ORT_RETURN_IF_NOT(FolderExists(path), "Directory does not exist: ", path);

    const std::string path_ending_with_double_null = path + '\0';
    SHFILEOPSTRUCTA sh_file_op{};
    sh_file_op.wFunc = FO_DELETE;
    sh_file_op.pFrom = path_ending_with_double_null.c_str();
    sh_file_op.fFlags = FOF_NO_UI;

    const auto result = ::SHFileOperationA(&sh_file_op);
    ORT_RETURN_IF_NOT(result == 0, "SHFileOperation() failed with error: ", result);

    ORT_RETURN_IF_NOT(
        !sh_file_op.fAnyOperationsAborted,
        "SHFileOperation() indicated that an operation was aborted.");

    return Status::OK();
  }

  common::Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const override {
    _sopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const override {
    _sopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileClose(int fd) const override {
    int ret = _close(fd);
    if (0 != ret) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status GetCanonicalPath(
      const std::basic_string<ORTCHAR_T>& path,
      std::basic_string<ORTCHAR_T>& canonical_path) const override {
    // adapted from MSVC STL std::filesystem::canonical() implementation
    // https://github.com/microsoft/STL/blob/ed3cbf36416a385828e7a5987ca52cb42882d84b/stl/inc/filesystem#L2986

    ScopedFileHandle file_handle{CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        nullptr)};

    ORT_RETURN_IF_NOT(
        file_handle.IsValid(), "CreateFile() failed: ", GetLastError());

    constexpr DWORD initial_buffer_size = MAX_PATH;
    std::vector<ORTCHAR_T> result_buffer{};
    result_buffer.resize(initial_buffer_size);

    while (true) {
      const DWORD result_length = GetFinalPathNameByHandleW(
          file_handle.Get(),
          result_buffer.data(),
          static_cast<DWORD>(result_buffer.size()),
          0);

      ORT_RETURN_IF_NOT(
          result_length > 0, "GetFinalPathNameByHandle() failed: ", GetLastError());

      if (result_length < result_buffer.size()) {  // buffer is large enough
        canonical_path.assign(result_buffer.data(), result_length);
        break;
      }

      // need larger buffer
      result_buffer.resize(result_length);
    }

    // update prefixes
    if (canonical_path.find(ORT_TSTR(R"(\\?\)")) == 0) {
      if (canonical_path.size() > 6 &&
          (ORT_TSTR('A') <= canonical_path[4] && canonical_path[4] <= ORT_TSTR('Z') ||
           ORT_TSTR('a') <= canonical_path[4] && canonical_path[4] <= ORT_TSTR('z')) &&
          canonical_path[5] == ORT_TSTR(':')) {
        // "\\?\<drive>:" -> "<drive>:"
        canonical_path.erase(0, 4);
      } else if (canonical_path.find(ORT_TSTR(R"(UNC\)"), 4) == 4) {
        // "\\?\UNC\" -> "\\"
        canonical_path.erase(2, 6);
      }
    }

    return Status::OK();
  }

  virtual Status LoadDynamicLibrary(const std::string& library_filename, void** handle) const override {
    *handle = ::LoadLibraryA(library_filename.c_str());
    if (!handle)
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Failed to load library");
    return common::Status::OK();
  }

  virtual common::Status UnloadDynamicLibrary(void* handle) const override {
    if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0)
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Failed to unload library");
    return common::Status::OK();
  }

  virtual Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
    if (!*symbol)
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Failed to find symbol in library");
    return common::Status::OK();
  }

  virtual std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override {
    ORT_UNUSED_PARAMETER(name);
    ORT_UNUSED_PARAMETER(version);
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  // \brief returns a provider that will handle telemetry on the current platform
  const Telemetry& GetTelemetryProvider() const override {
    return telemetry_provider_;
  }

  // \brief returns a value for the queried variable name (var_name)
  std::string GetEnvironmentVar(const std::string& var_name) const override {
    // Why getenv() should be avoided on Windows:
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
    // Instead use the Win32 API: GetEnvironmentVariableA()

    // Max limit of an environment variable on Windows including the null-terminating character
    constexpr DWORD kBufferSize = 32767;

    // Create buffer to hold the result
    char buffer[kBufferSize];

    auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer, kBufferSize);

    // Will be > 0 if the API call was successful
    if (char_count) {
      return std::string(buffer, buffer + char_count);
    }

    // TODO: Understand the reason for failure by calling GetLastError().
    // If it is due to the specified environment variable being found in the environment block,
    // GetLastError() returns ERROR_ENVVAR_NOT_FOUND.
    // For now, we assume that the environment variable is not found.

    return std::string();
  }

 private:
  WindowsEnv()
      : GetSystemTimePreciseAsFileTime_(nullptr) {
    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
    if (module != nullptr) {
      auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
      GetSystemTimePreciseAsFileTime_ = func;
    }
  }

  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
  WindowsTelemetry telemetry_provider_;
};
}  // namespace

#if defined(PLATFORM_WINDOWS)
const Env& Env::Default() {
  return WindowsEnv::Instance();
}
#endif

}  // namespace onnxruntime
