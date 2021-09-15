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

#include <Windows.h>

#include <fstream>
#include <string>
#include <thread>
#include <process.h>
#include <fcntl.h>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/platform/scoped_resource.h"
#include "core/platform/windows/telemetry.h"
#include "unsupported/Eigen/CXX11/src/ThreadPool/ThreadPoolInterface.h"
#include <wil/Resource.h>

#include "core/platform/path_lib.h"  // for LoopDir()

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

namespace onnxruntime {

namespace {

std::wstring Basename(const std::wstring& path) {
  auto basename_index = path.find_last_of(L"/\\") + 1;  // results in 0 if no separator is found
  return path.substr(basename_index);
}

class WindowsThread : public EnvThread {
 private:
  struct Param {
    const ORTCHAR_T* name_prefix;
    int index;
    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param);
    Eigen::ThreadPoolInterface* param;
    const ThreadOptions& thread_options;
    Param(const ORTCHAR_T* name_prefix1,
          int index1,
          unsigned (*start_address1)(int id, Eigen::ThreadPoolInterface* param),
          Eigen::ThreadPoolInterface* param1,
          const ThreadOptions& thread_options1) : name_prefix(name_prefix1), index(index1), start_address(start_address1), param(param1), thread_options(thread_options1) {}
  };

 public:
  WindowsThread(const ORTCHAR_T* name_prefix, int index,
                unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
                const ThreadOptions& thread_options) {
    custom_create_thread_fn = thread_options.custom_create_thread_fn;
    custom_thread_creation_options = thread_options.custom_thread_creation_options;
    custom_join_thread_fn = thread_options.custom_join_thread_fn;
    std::unique_ptr<Param> local_param = std::make_unique<Param>(name_prefix, index, start_address, param, thread_options);
    if (custom_create_thread_fn) {
      custom_thread_handle = custom_create_thread_fn(custom_thread_creation_options, (OrtThreadWorkerFn)CustomThreadMain, local_param.release());
      if (!custom_thread_handle) {
        ORT_THROW("custom_create_thread_fn returned invalid handle.");
      }
    } else {
      hThread.reset(reinterpret_cast<HANDLE>(_beginthreadex(nullptr, thread_options.stack_size, ThreadMain,
                                                            local_param.release(), 0,
                                                            &threadID)));
    }
  }

  ~WindowsThread() {
    if (custom_thread_handle) {
      custom_join_thread_fn(custom_thread_handle);
      custom_thread_handle = nullptr;
    } else {
      DWORD waitStatus = WaitForSingleObject(hThread.get(), INFINITE);
      FAIL_FAST_LAST_ERROR_IF(waitStatus == WAIT_FAILED);
    }
  }

 private:
  typedef HRESULT(WINAPI* SetThreadDescriptionFunc)(HANDLE hThread, PCWSTR lpThreadDescription);

#pragma warning(push)
#pragma warning(disable : 6387)
  static unsigned __stdcall ThreadMain(void* param) {
    std::unique_ptr<Param> p((Param*)param);
    // TODO: should I try to use SetThreadSelectedCpuSets?
    if (!p->thread_options.affinity.empty())
      SetThreadAffinityMask(GetCurrentThread(), p->thread_options.affinity[p->index]);
#if WINVER >= _WIN32_WINNT_WIN10
    constexpr SetThreadDescriptionFunc pSetThrDesc = SetThreadDescription;
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    HMODULE kernelModule = GetModuleHandle(TEXT("kernel32.dll"));
    // kernel32.dll is always loaded
    assert(kernelModule != nullptr);
    auto pSetThrDesc =
        (SetThreadDescriptionFunc)GetProcAddress(kernelModule, "SetThreadDescription");
#else
    constexpr SetThreadDescriptionFunc pSetThrDesc = nullptr;
#endif
    if (pSetThrDesc != nullptr) {
      const ORTCHAR_T* name_prefix =
          (p->name_prefix == nullptr || wcslen(p->name_prefix) == 0) ? L"onnxruntime" : p->name_prefix;
      std::wostringstream oss;
      oss << name_prefix << "-" << p->index;
      // Ignore the error
      (void)pSetThrDesc(GetCurrentThread(), oss.str().c_str());
    }
    unsigned ret = 0;
    ORT_TRY {
      ret = p->start_address(p->index, p->param);
    }
    ORT_CATCH(const std::exception&) {
      p->param->Cancel();
      ret = 1;
    }
    return ret;
  }
#pragma warning(pop)

  static void __stdcall CustomThreadMain(void* param) {
    std::unique_ptr<Param> p((Param*)param);
    ORT_TRY {
      p->start_address(p->index, p->param);
    }
    ORT_CATCH(const std::exception&) {
      p->param->Cancel();
    }
  }
  unsigned threadID = 0;
  wil::unique_handle hThread;
};

class WindowsEnv : public Env {
 public:
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
  EnvThread* CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) {
    return new WindowsThread(name_prefix, index, start_address, param, thread_options);
  }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  void SleepForMicroseconds(int64_t micros) const override {
    Sleep(static_cast<DWORD>(micros) / 1000);
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
    if (!processorCoreCount)
      ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
    return processorCoreCount;
  }

  std::vector<size_t> GetThreadAffinityMasks() const override {
    auto generate_vector_of_n = [](int n) {
      std::vector<size_t> ret(n);
      std::iota(ret.begin(), ret.end(), 0);
      return ret;
    };
    // Indeed 64 should be enough. However, it's harmless to have a little more.
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD returnLength = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
      return generate_vector_of_n(std::thread::hardware_concurrency());
    }
    std::vector<size_t> ret;
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ret.push_back(buffer[i].ProcessorMask);
      }
    }
    if (ret.empty())
      return generate_vector_of_n(std::thread::hardware_concurrency());
    return ret;
  }

  static WindowsEnv& Instance() {
    static WindowsEnv default_env;
    return default_env;
  }

  PIDType GetSelfPid() const override {
    return GetCurrentProcessId();
  }

  Status GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const override {
#if WINVER >= _WIN32_WINNT_WIN8
    wil::unique_hfile file_handle{
        CreateFile2(file_path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
#else
    wil::unique_hfile file_handle{
        CreateFileW(file_path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
#endif
    if (file_handle.get() == INVALID_HANDLE_VALUE) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToUTF8String(Basename(file_path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
    }
    LARGE_INTEGER filesize;
    if (!GetFileSizeEx(file_handle.get(), &filesize)) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileSizeEx ", ToUTF8String(Basename(file_path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
    }
    if (static_cast<ULONGLONG>(filesize.QuadPart) > std::numeric_limits<size_t>::max()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileLength: File is too large");
    }
    length = static_cast<size_t>(filesize.QuadPart);
    return Status::OK();
  }

  common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const override {
    using namespace common;
    if (fd < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid fd was supplied: ", fd);
    }

    struct _stat buf;
    int rc = _fstat(fd, &buf);
    if (rc < 0) {
      return Status(SYSTEM, errno);
    }

    if (buf.st_size < 0) {
      return ORT_MAKE_STATUS(SYSTEM, FAIL, "Received negative size from stat call");
    }

    if (static_cast<unsigned long long>(buf.st_size) > std::numeric_limits<size_t>::max()) {
      return ORT_MAKE_STATUS(SYSTEM, FAIL, "File is too large.");
    }

    file_size = static_cast<size_t>(buf.st_size);
    return Status::OK();
  }

  Status ReadFileIntoBuffer(_In_z_ const ORTCHAR_T* const file_path, const FileOffsetType offset, const size_t length,
                            const gsl::span<char> buffer) const override {
    ORT_RETURN_IF_NOT(file_path, "file_path == nullptr");
    ORT_RETURN_IF_NOT(offset >= 0, "offset < 0");
    ORT_RETURN_IF_NOT(length <= buffer.size(), "length > buffer.size()");
#if WINVER >= _WIN32_WINNT_WIN8
    wil::unique_hfile file_handle{
        CreateFile2(file_path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
#else
    wil::unique_hfile file_handle{
        CreateFileW(file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
#endif
    if (file_handle.get() == INVALID_HANDLE_VALUE) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToUTF8String(Basename(file_path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
    }

    if (length == 0)
      return Status::OK();

    if (offset > 0) {
      LARGE_INTEGER current_position;
      current_position.QuadPart = offset;
      if (!SetFilePointerEx(file_handle.get(), current_position, &current_position, FILE_BEGIN)) {
        const auto error_code = GetLastError();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SetFilePointerEx ", ToUTF8String(Basename(file_path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
      }
    }

    size_t total_bytes_read = 0;
    while (total_bytes_read < length) {
      constexpr DWORD k_max_bytes_to_read = 1 << 30;  // read at most 1GB each time
      const size_t bytes_remaining = length - total_bytes_read;
      const DWORD bytes_to_read = static_cast<DWORD>(std::min<size_t>(bytes_remaining, k_max_bytes_to_read));
      DWORD bytes_read;

      if (!ReadFile(file_handle.get(), buffer.data() + total_bytes_read, bytes_to_read, &bytes_read, nullptr)) {
        const auto error_code = GetLastError();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToUTF8String(Basename(file_path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
      }

      if (bytes_read != bytes_to_read) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFile ", ToUTF8String(Basename(file_path)), " fail: unexpected end");
      }

      total_bytes_read += bytes_read;
    }

    return Status::OK();
  }

  Status MapFileIntoMemory(_In_z_ const ORTCHAR_T*, FileOffsetType, size_t, MappedMemoryPtr&) const override {
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

  common::Status DeleteFolder(const PathString& path) const override {
    Status final_status = Status::OK();
    LoopDir(
        path,
        [this, &path, &final_status](
            const PathString& child_basename, OrtFileType file_type) {
          // ignore . and ..
          if (child_basename == ORT_TSTR(".") || child_basename == ORT_TSTR("..")) {
            return true;
          }

          const PathString child_path = path + GetPathSep<PathChar>() + child_basename;

          if (file_type == OrtFileType::TYPE_DIR) {
            const auto delete_dir_status = DeleteFolder(child_path);
            if (!delete_dir_status.IsOK()) {
              final_status = delete_dir_status;
            }
          } else {  // not directory
            if (!DeleteFileW(child_path.c_str())) {
              const auto error_code = GetLastError();
              final_status = ORT_MAKE_STATUS(
                  ONNXRUNTIME, FAIL,
                  "DeleteFile() failed - path: ", ToUTF8String(Basename(child_path)),
                  ", error code: ", error_code, " - ", std::system_category().message(error_code));
            }
          }

          return final_status.IsOK();
        });

    ORT_RETURN_IF_ERROR(final_status);

    if (!RemoveDirectoryW(path.c_str())) {
      const auto error_code = GetLastError();
      final_status = ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "RemoveDirectory() failed - path: ", ToUTF8String(Basename(path)),
          ", error code: ", error_code, " - ", std::system_category().message(error_code));
    }

    return final_status;
  }

  common::Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) const override {
    _wsopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR,
              _S_IREAD | _S_IWRITE);
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
    _sopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR,
             _S_IREAD | _S_IWRITE);
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
      const PathString& path,
      PathString& canonical_path) const override {
    // adapted from MSVC STL std::filesystem::canonical() implementation
    // https://github.com/microsoft/STL/blob/ed3cbf36416a385828e7a5987ca52cb42882d84b/stl/inc/filesystem#L2986
#if WINVER >= _WIN32_WINNT_WIN8
    wil::unique_hfile file_handle{CreateFile2(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        OPEN_EXISTING,
        NULL)};
#else
    wil::unique_hfile file_handle{CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        nullptr)};
#endif

    if (file_handle.get() == INVALID_HANDLE_VALUE) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToUTF8String(Basename(path)), " fail, errcode = ", error_code, " - ", std::system_category().message(error_code));
    }

    constexpr DWORD initial_buffer_size = MAX_PATH;
    std::vector<PathChar> result_buffer{};
    result_buffer.resize(initial_buffer_size);

    while (true) {
      const DWORD result_length = GetFinalPathNameByHandleW(
          file_handle.get(),
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

  // Return the path of the executable/shared library for the current running code. This is to make it
  // possible to load other shared libraries installed next to our core runtime code.
  std::string GetRuntimePath() const override {
    char buffer[MAX_PATH];
    if (!GetModuleFileNameA(reinterpret_cast<HINSTANCE>(&__ImageBase), buffer, _countof(buffer)))
      return "";

    // Remove the filename at the end, but keep the trailing slash
    std::string path(buffer);
    auto slash_index = path.find_last_of('\\');
    if (slash_index == std::string::npos)
      return "";

    return path.substr(0, slash_index + 1);
  }

  virtual Status LoadDynamicLibrary(const std::string& library_filename, bool /*global_symbols*/, void** handle) const override {
    const std::wstring& wlibrary_filename = ToWideString(library_filename);
#if WINAPI_FAMILY == WINAPI_FAMILY_PC_APP
    *handle = ::LoadPackagedLibrary(wlibrary_filename.c_str(), 0);
#else
    // TODO: in most cases, the path name is a relative path and the behavior of the following line of code is undefined.
    *handle = ::LoadLibraryExW(wlibrary_filename.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
    if (!*handle) {
      const auto error_code = GetLastError();
      static constexpr DWORD bufferLength = 64 * 1024;
      std::wstring s(bufferLength, '\0');
      FormatMessageW(
              FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL,
          error_code,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          (LPWSTR)s.data(),
          0, NULL);
      std::wostringstream oss;
      oss << L"LoadLibrary failed with error " << error_code << L" \"" << s.c_str() << L"\" when trying to load \"" << wlibrary_filename << L"\"";
      std::wstring errmsg = oss.str();
      // TODO: trim the ending '\r' and/or '\n'
      common::Status status(common::ONNXRUNTIME, common::FAIL, ToUTF8String(errmsg));
      return status;
    }
    return Status::OK();
  }

  virtual Status UnloadDynamicLibrary(void* handle) const override {
    if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "FreeLibrary failed with error ", error_code, " - ", std::system_category().message(error_code));
    }
    return Status::OK();
  }

  virtual Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
    if (!*symbol) {
      const auto error_code = GetLastError();
      static constexpr DWORD bufferLength = 64 * 1024;
      std::wstring s(bufferLength, '\0');
      FormatMessageW(
              FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL,
          error_code,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          (LPWSTR)s.data(),
          0, NULL);
      std::wostringstream oss;
      oss << L"Failed to find symbol " << ToWideString(symbol_name) << L" in library, error code: " << error_code << L" \"" << s.c_str() << L"\"";
      std::wstring errmsg = oss.str();
      // TODO: trim the ending '\r' and/or '\n'
      common::Status status(common::ONNXRUNTIME, common::FAIL, ToUTF8String(errmsg));
      return status;
    }
    return Status::OK();
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
    std::string buffer(kBufferSize, '\0');

    // The last argument is the size of the buffer pointed to by the lpBuffer parameter, including the null-terminating character, in characters.
    // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer, not including the terminating null character.
    // Therefore, If the function succeeds, kBufferSize should be larger than char_count.
    auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);

    if (kBufferSize > char_count) {
      buffer.resize(char_count);
      return buffer;
    }

    // Else either the call was failed, or the buffer wasn't large enough.
    // TODO: Understand the reason for failure by calling GetLastError().
    // If it is due to the specified environment variable being found in the environment block,
    // GetLastError() returns ERROR_ENVVAR_NOT_FOUND.
    // For now, we assume that the environment variable is not found.

    return std::string();
  }

 private:
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  WindowsTelemetry telemetry_provider_;
};
}  // namespace

Env& Env::Default() {
  return WindowsEnv::Instance();
}
}  // namespace onnxruntime
