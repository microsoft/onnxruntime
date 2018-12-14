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
    ORT_UNUSED_PARAMETER(library_filename);
    ORT_UNUSED_PARAMETER(handle);
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual common::Status UnloadDynamicLibrary(void* handle) const override {
    ORT_UNUSED_PARAMETER(handle);
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    ORT_UNUSED_PARAMETER(handle);
    ORT_UNUSED_PARAMETER(symbol_name);
    ORT_UNUSED_PARAMETER(symbol);
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
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
