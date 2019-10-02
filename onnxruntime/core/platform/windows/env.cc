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

#include <string>
#include <thread>
#include <fcntl.h>
#include <fstream>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/platform/scoped_resource.h"

namespace onnxruntime {

namespace {

struct FileHandleTraits {
  using Handle = HANDLE;
  static Handle GetInvalidHandleValue() noexcept { return INVALID_HANDLE_VALUE; }
  static void CleanUp(Handle h) noexcept { CloseHandle(h); }
};

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
      int err = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileSizeEx ", ToMBString(file_path), " fail, errcode = ", err);
    }
    if (static_cast<ULONGLONG>(filesize.QuadPart) > std::numeric_limits<size_t>::max()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileLength: File is too large");
    }
    length = static_cast<size_t>(filesize.QuadPart);
    return Status::OK();
  }

  Status ReadFileIntoBuffer(
      const ORTCHAR_T* const file_path, const OffsetType offset, const size_t length,
      const gsl::span<char> buffer) const override {
    ORT_RETURN_IF_NOT(file_path);
    ORT_RETURN_IF_NOT(offset >= 0);
    ORT_RETURN_IF_NOT(buffer.size() >= 0 && static_cast<size_t>(buffer.size()) < length);

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
      const size_t length_remaining = length - total_bytes_read;
      DWORD bytes_to_read, bytes_read;

      // read at most 1GB each time
      constexpr DWORD k_max_read_size = 1 << 30;

      if (length_remaining > k_max_read_size) {
        bytes_to_read = k_max_read_size;
      } else {
        bytes_to_read = static_cast<DWORD>(length_remaining);
      }

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
      const ORTCHAR_T* file_path, OffsetType offset, size_t length,
      MappedMemoryPtr& mapped_memory) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "MapFileIntoMemory is not implemented on Windows.");
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
};

}  // namespace

#if defined(PLATFORM_WINDOWS)
const Env& Env::Default() {
  return WindowsEnv::Instance();
}
#endif

}  // namespace onnxruntime
