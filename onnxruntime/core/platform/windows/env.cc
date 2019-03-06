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

#include <Shlwapi.h>
#include <Windows.h>

#include <string>
#include <thread>
#include <fcntl.h>
#include <fstream>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/platform/env.h"

namespace onnxruntime {

namespace {

class StdThread : public Thread {
 public:
  StdThread(std::function<void()> fn)
      : thread_(fn) {}

  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

class WindowsEnv : public Env {
 public:
  void SleepForMicroseconds(int64_t micros) const override { Sleep(static_cast<DWORD>(micros) / 1000); }

  Thread* StartThread(const ThreadOptions&, const std::string&,
                      std::function<void()> fn) const override {
    return new StdThread(fn);
  }

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

  EnvThread* CreateThread(std::function<void()> fn) const override {
    return new StdThread(fn);
  }

  Task CreateTask(std::function<void()> f) const override {
    return Task{std::move(f)};
  }
  void ExecuteTask(const Task& t) const override {
    t.f();
  }
  common::Status ReadFileAsString(const wchar_t* fname, std::string* out) const override {
    if (!fname) return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "file name is nullptr");
    if (!out) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'out' cannot be NULL");
    }
    HANDLE hFile = CreateFileW(fname, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      int err = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToMBString(fname), " fail, errcode =", err);
    }
    std::unique_ptr<void, decltype(&CloseHandle)> handler_holder(hFile, CloseHandle);
    LARGE_INTEGER filesize;
    if (!GetFileSizeEx(hFile, &filesize)) {
      int err = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileSizeEx ", ToMBString(fname), " fail, errcode =", err);
    }
    // check the file file for avoiding allocating a zero length buffer
    if (filesize.QuadPart == 0) {  // empty file
      out->clear();
      return Status::OK();
    }
    out->resize(filesize.QuadPart, '\0');
    char* wptr = const_cast<char*>(out->data());
    auto length_remain = filesize.QuadPart;
    DWORD readed = 0;
    for (; length_remain > 0; wptr += readed, length_remain -= readed) {
      //read at most 1GB each time
      DWORD bytes_to_read;
      if (length_remain > (1 << 30)) {
        bytes_to_read = 1 << 30;
      } else {
        bytes_to_read = static_cast<DWORD>(length_remain);
      }
      if (ReadFile(hFile, wptr, bytes_to_read, &readed, nullptr) != TRUE) {
        int err = GetLastError();
        out->clear();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToMBString(fname), " fail, errcode =", err);
      }
      if (readed != bytes_to_read) {
        out->clear();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToMBString(fname), " fail: unexpected end");
      }
    }
    return common::Status::OK();
  }

  common::Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
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
    _sopen_s(&fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
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
