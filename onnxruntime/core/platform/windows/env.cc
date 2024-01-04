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

#include "core/platform/windows/env.h"

#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <climits>
#include <process.h>
#include <fcntl.h>
#include <io.h>

#include "core/common/gsl.h"
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/platform/env.h"
#include "core/platform/scoped_resource.h"
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <wil/Resource.h>

#include "core/platform/path_lib.h"  // for LoopDir()

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

namespace onnxruntime {

class UnmapFileParam {
 public:
  void* addr;
  size_t len;
};

static void UnmapFile(void* param) noexcept {
  std::unique_ptr<UnmapFileParam> p(reinterpret_cast<UnmapFileParam*>(param));
  bool ret = UnmapViewOfFile(p->addr);
  if (!ret) {
    const auto error_code = GetLastError();
    LOGS_DEFAULT(ERROR) << "unmap view of file failed. error code: " << error_code
                        << " error msg: " << std::system_category().message(error_code);
  }
}

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
    std::optional<LogicalProcessors> affinity;
    Param(const ORTCHAR_T* name_prefix1,
          int index1,
          unsigned (*start_address1)(int id, Eigen::ThreadPoolInterface* param),
          Eigen::ThreadPoolInterface* param1)
        : name_prefix(name_prefix1),
          index(index1),
          start_address(start_address1),
          param(param1) {}
  };

 public:
  WindowsThread(const ORTCHAR_T* name_prefix, int index,
                unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
                const ThreadOptions& thread_options) {
    ORT_ENFORCE(index >= 0, "Negative thread index is not allowed");
    custom_create_thread_fn = thread_options.custom_create_thread_fn;
    custom_thread_creation_options = thread_options.custom_thread_creation_options;
    custom_join_thread_fn = thread_options.custom_join_thread_fn;

    std::unique_ptr<Param> local_param = std::make_unique<Param>(name_prefix, index, start_address, param);
    if (narrow<size_t>(index) < thread_options.affinities.size()) {
      local_param->affinity = thread_options.affinities[index];
    }

    if (custom_create_thread_fn) {
      custom_thread_handle = custom_create_thread_fn(custom_thread_creation_options, CustomThreadMain, local_param.get());
      if (!custom_thread_handle) {
        ORT_THROW("custom_create_thread_fn returned invalid handle.");
      }
      local_param.release();
    } else {
      _set_errno(0);
      _set_doserrno(0);
      auto th_handle = _beginthreadex(nullptr, thread_options.stack_size, ThreadMain,
                                      local_param.get(), 0,
                                      &threadID);
      if (th_handle == 0) {
        auto err = errno;
        auto dos_error = _doserrno;
        char message_buf[256];
        strerror_s(message_buf, sizeof(message_buf), err);
        ORT_THROW("WindowThread:_beginthreadex failed with message: ", message_buf, " doserrno: ", dos_error);
      }
      local_param.release();
      hThread.reset(reinterpret_cast<HANDLE>(th_handle));
      // Do not throw beyond this point so we do not lose thread handle and then not being able to join it.
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
    std::unique_ptr<Param> p(static_cast<Param*>(param));

    // Not all machines have kernel32.dll and/or SetThreadDescription (e.g. Azure App Service sandbox)
    // so we need to ensure it's available before calling.
    HMODULE kernelModule = GetModuleHandle(TEXT("kernel32.dll"));
    if (kernelModule != nullptr) {
      auto setThreadDescriptionFn = (SetThreadDescriptionFunc)GetProcAddress(kernelModule, "SetThreadDescription");
      if (setThreadDescriptionFn != nullptr) {
        const ORTCHAR_T* name_prefix = (p->name_prefix == nullptr || wcslen(p->name_prefix) == 0) ? L"onnxruntime"
                                                                                                  : p->name_prefix;
        std::wostringstream oss;
        oss << name_prefix << "-" << p->index;
        // Ignore any errors
        (void)(setThreadDescriptionFn)(GetCurrentThread(), oss.str().c_str());
      }
    }

    unsigned ret = 0;
    ORT_TRY {
      if (p->affinity.has_value() && !p->affinity->empty()) {
        int group_id = -1;
        KAFFINITY mask = 0;
        constexpr KAFFINITY bit = 1;
        const WindowsEnv& env = WindowsEnv::Instance();
        for (auto global_processor_id : *p->affinity) {
          auto processor_info = env.GetProcessorAffinityMask(global_processor_id);
          if (processor_info.local_processor_id > -1 &&
              processor_info.local_processor_id < sizeof(KAFFINITY) * CHAR_BIT) {
            mask |= bit << processor_info.local_processor_id;
          } else {
            // Logical processor id starts from 0 internally, but in ort API, it starts from 1,
            // that's why id need to increase by 1 when logging.
            LOGS_DEFAULT(ERROR) << "Cannot set affinity for thread " << GetCurrentThreadId()
                                << ", processor " << global_processor_id + 1 << " does not exist";
            group_id = -1;
            mask = 0;
            break;
          }
          if (group_id == -1) {
            group_id = processor_info.group_id;
          } else if (group_id != processor_info.group_id) {
            LOGS_DEFAULT(ERROR) << "Cannot set cross-group affinity for thread "
                                << GetCurrentThreadId() << ", first on group "
                                << group_id << ", then on " << processor_info.group_id;
            group_id = -1;
            mask = 0;
            break;
          }
        }  // for
        if (group_id > -1 && mask) {
          GROUP_AFFINITY thread_affinity = {};
          thread_affinity.Group = static_cast<WORD>(group_id);
          thread_affinity.Mask = mask;
          if (SetThreadGroupAffinity(GetCurrentThread(), &thread_affinity, nullptr)) {
            LOGS_DEFAULT(VERBOSE) << "SetThreadAffinityMask done for thread: " << GetCurrentThreadId()
                                  << ", group_id: " << thread_affinity.Group
                                  << ", mask: " << thread_affinity.Mask;
          } else {
            const auto error_code = GetLastError();
            LOGS_DEFAULT(ERROR) << "SetThreadAffinityMask failed for thread: " << GetCurrentThreadId()
                                << ", index: " << p->index
                                << ", mask: " << *p->affinity
                                << ", error code: " << error_code
                                << ", error msg: " << std::system_category().message(error_code)
                                << ". Specify the number of threads explicitly so the affinity is not set.";
          }
        }
      }

      ret = p->start_address(p->index, p->param);
    }
    ORT_CATCH(...) {
      p->param->Cancel();
      ret = 1;
    }
    return ret;
  }
#pragma warning(pop)

  static void CustomThreadMain(void* param) {
    std::unique_ptr<Param> p(static_cast<Param*>(param));
    ORT_TRY {
      p->start_address(p->index, p->param);
    }
    ORT_CATCH(...) {
      p->param->Cancel();
    }
  }
  unsigned threadID = 0;
  wil::unique_handle hThread;
};

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
EnvThread* WindowsEnv::CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                                    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                                    Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) {
  return new WindowsThread(name_prefix, index, start_address, param, thread_options);
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

Env& Env::Default() {
  return WindowsEnv::Instance();
}

void WindowsEnv::SleepForMicroseconds(int64_t micros) const {
  Sleep(static_cast<DWORD>(micros) / 1000);
}

int WindowsEnv::DefaultNumCores() {
  return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
}

int WindowsEnv::GetNumPhysicalCpuCores() const {
  return cores_.empty() ? DefaultNumCores() : static_cast<int>(cores_.size());
}

std::vector<LogicalProcessors> WindowsEnv::GetDefaultThreadAffinities() const {
  return cores_.empty() ? std::vector<LogicalProcessors>(DefaultNumCores(), LogicalProcessors{}) : cores_;
}

WindowsEnv& WindowsEnv::Instance() {
  static WindowsEnv default_env;
  return default_env;
}

PIDType WindowsEnv::GetSelfPid() const {
  return GetCurrentProcessId();
}

Status WindowsEnv::GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const {
  wil::unique_hfile file_handle{
      CreateFile2(file_path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
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

common::Status WindowsEnv::GetFileLength(int fd, /*out*/ size_t& file_size) const {
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

Status WindowsEnv::ReadFileIntoBuffer(_In_z_ const ORTCHAR_T* const file_path, const FileOffsetType offset, const size_t length,
                                      const gsl::span<char> buffer) const {
  ORT_RETURN_IF_NOT(file_path, "file_path == nullptr");
  ORT_RETURN_IF_NOT(offset >= 0, "offset < 0");
  ORT_RETURN_IF_NOT(length <= buffer.size(), "length > buffer.size()");
  wil::unique_hfile file_handle{
      CreateFile2(file_path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
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

Status WindowsEnv::MapFileIntoMemory(_In_z_ const ORTCHAR_T* file_path,
                                     FileOffsetType offset,
                                     size_t length,
                                     MappedMemoryPtr& mapped_memory) const {
  ORT_RETURN_IF_NOT(file_path, "file_path == nullptr");
  ORT_RETURN_IF_NOT(offset >= 0, "offset < 0");

  if (length == 0) {
    mapped_memory = MappedMemoryPtr{};
    return Status::OK();
  }

  wil::unique_hfile file_handle{
      CreateFile2(file_path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "open file ", ToUTF8String(Basename(file_path)),
                           " fail, errcode = ", error_code,
                           " - ", std::system_category().message(error_code));
  }

#if NTDDI_VERSION >= NTDDI_WIN10_RS5 && WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP | WINAPI_PARTITION_SYSTEM)
  wil::unique_hfile file_mapping_handle{
      CreateFileMapping2(file_handle.get(),
                         nullptr,
                         FILE_MAP_READ,
                         PAGE_READONLY,
                         SEC_COMMIT,
                         0,
                         nullptr,
                         nullptr,
                         0)};
#else
  wil::unique_hfile file_mapping_handle{
      CreateFileMappingW(file_handle.get(),
                         nullptr,
                         PAGE_READONLY,
                         0,
                         0,
                         nullptr)};
#endif
  if (file_mapping_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "open file mapping ", ToUTF8String(Basename(file_path)),
                           " fail, errcode = ", error_code,
                           " - ", std::system_category().message(error_code));
  }

  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);

  static const DWORD page_size = sysinfo.dwPageSize;
  static const DWORD allocation_granularity = sysinfo.dwAllocationGranularity;
  const FileOffsetType offset_to_page = offset % static_cast<FileOffsetType>(page_size);
  const size_t mapped_length = length + static_cast<size_t>(offset_to_page);
  const FileOffsetType mapped_offset = offset - offset_to_page;
  if (mapped_offset % allocation_granularity != 0) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "mapped offset must be a multiple of the allocation granularity",
                           " , mapped_offset = ", mapped_offset,
                           " , allocation_granularity = ", allocation_granularity,
                           " , errcode = ", error_code,
                           " - ", std::system_category().message(error_code));
  }

  void* const mapped_base = MapViewOfFile(file_mapping_handle.get(),
                                          FILE_MAP_READ,
                                          0,
                                          static_cast<DWORD>(mapped_offset),
                                          mapped_length);
  GSL_SUPPRESS(r.11)
  mapped_memory =
      MappedMemoryPtr{reinterpret_cast<char*>(mapped_base) + offset_to_page,
                      OrtCallbackInvoker{OrtCallback{UnmapFile, new UnmapFileParam{mapped_base, mapped_length}}}};

  return Status::OK();
}

bool WindowsEnv::FolderExists(const std::wstring& path) const {
  DWORD attributes = GetFileAttributesW(path.c_str());
  return (attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

bool WindowsEnv::FolderExists(const std::string& path) const {
  DWORD attributes = GetFileAttributesA(path.c_str());
  return (attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

common::Status WindowsEnv::CreateFolder(const std::wstring& path) const {
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

common::Status WindowsEnv::CreateFolder(const std::string& path) const {
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

common::Status WindowsEnv::DeleteFolder(const PathString& path) const {
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

common::Status WindowsEnv::FileOpenRd(const std::wstring& path, /*out*/ int& fd) const {
  _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return common::Status(common::SYSTEM, errno);
  }
  return Status::OK();
}

common::Status WindowsEnv::FileOpenWr(const std::wstring& path, /*out*/ int& fd) const {
  _wsopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR,
            _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return common::Status(common::SYSTEM, errno);
  }
  return Status::OK();
}

common::Status WindowsEnv::FileOpenRd(const std::string& path, /*out*/ int& fd) const {
  _sopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return common::Status(common::SYSTEM, errno);
  }
  return Status::OK();
}

common::Status WindowsEnv::FileOpenWr(const std::string& path, /*out*/ int& fd) const {
  _sopen_s(&fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR,
           _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return common::Status(common::SYSTEM, errno);
  }
  return Status::OK();
}

common::Status WindowsEnv::FileClose(int fd) const {
  int ret = _close(fd);
  if (0 != ret) {
    return common::Status(common::SYSTEM, errno);
  }
  return Status::OK();
}

common::Status WindowsEnv::GetCanonicalPath(
    const PathString& path,
    PathString& canonical_path) const {
  // adapted from MSVC STL std::filesystem::canonical() implementation
  // https://github.com/microsoft/STL/blob/ed3cbf36416a385828e7a5987ca52cb42882d84b/stl/inc/filesystem#L2986
  CREATEFILE2_EXTENDED_PARAMETERS param;
  memset(&param, 0, sizeof(param));
  param.dwSize = sizeof(CREATEFILE2_EXTENDED_PARAMETERS);
  param.dwFileFlags = FILE_FLAG_BACKUP_SEMANTICS;
  wil::unique_hfile file_handle{CreateFile2(
      path.c_str(),
      FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      OPEN_EXISTING,
      &param)};

  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToUTF8String(Basename(path)), " fail, errcode = ",
                           error_code, " - ", std::system_category().message(error_code));
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
PathString WindowsEnv::GetRuntimePath() const {
  wchar_t buffer[MAX_PATH];
  if (!GetModuleFileNameW(reinterpret_cast<HINSTANCE>(&__ImageBase), buffer, _countof(buffer))) {
    return PathString();
  }

  // Remove the filename at the end, but keep the trailing slash
  PathString path(buffer);
  auto slash_index = path.find_last_of(ORT_TSTR('\\'));
  if (slash_index == std::string::npos) {
    // Windows supports forward slashes
    slash_index = path.find_last_of(ORT_TSTR('/'));
    if (slash_index == std::string::npos) {
      return PathString();
    }
  }
  return path.substr(0, slash_index + 1);
}

Status WindowsEnv::LoadDynamicLibrary(const PathString& wlibrary_filename, bool /*global_symbols*/, void** handle) const {
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

Status WindowsEnv::UnloadDynamicLibrary(void* handle) const {
  if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "FreeLibrary failed with error ", error_code, " - ", std::system_category().message(error_code));
  }
  return Status::OK();
}

namespace dlfcn_win32 {
// adapted from https://github.com/dlfcn-win32 version 1.3.1.
// Simplified to only support finding symbols in libraries that were linked against.
// If ORT dynamically loads a custom ops library using RegisterCustomOpsLibrary[_V2] the handle from the library load
// is explicitly provided in the call to GetSymbolFromLibrary.
//
/* Load Psapi.dll at runtime, this avoids linking caveat */
bool MyEnumProcessModules(HANDLE hProcess, HMODULE* lphModule, DWORD cb, LPDWORD lpcbNeeded) {
  using EnumProcessModulesFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
  static EnumProcessModulesFn EnumProcessModulesPtr = []() {
    EnumProcessModulesFn fn = nullptr;
    // Windows 7 and newer versions have K32EnumProcessModules in Kernel32.dll which is always pre-loaded
    HMODULE psapi = GetModuleHandleA("Kernel32.dll");
    if (psapi) {
      fn = (EnumProcessModulesFn)(LPVOID)GetProcAddress(psapi, "K32EnumProcessModules");
    }

    return fn;
  }();

  if (EnumProcessModulesPtr == nullptr) {
    return false;
  }

  return EnumProcessModulesPtr(hProcess, lphModule, cb, lpcbNeeded);
}

void* SearchModulesForSymbol(const char* name) {
  HANDLE current_proc = GetCurrentProcess();
  DWORD size = 0;
  void* symbol = nullptr;

  // GetModuleHandle(NULL) only returns the current program file. So if we want to get ALL loaded module including
  // those in linked DLLs, we have to use EnumProcessModules().
  if (MyEnumProcessModules(current_proc, nullptr, 0, &size) != false) {
    size_t num_handles = size / sizeof(HMODULE);
    std::unique_ptr<HMODULE[]> modules = std::make_unique<HMODULE[]>(num_handles);
    HMODULE* modules_ptr = modules.get();
    DWORD cb_needed = 0;
    if (MyEnumProcessModules(current_proc, modules_ptr, size, &cb_needed) != 0 && size == cb_needed) {
      for (size_t i = 0; i < num_handles; i++) {
        symbol = GetProcAddress(modules[i], name);
        if (symbol != nullptr) {
          break;
        }
      }
    }
  }

  return symbol;
}
}  // namespace dlfcn_win32

Status WindowsEnv::GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const {
  Status status = Status::OK();

  // global search to replicate dlsym RTLD_DEFAULT if handle is nullptr
  if (handle == nullptr) {
    *symbol = dlfcn_win32::SearchModulesForSymbol(symbol_name.c_str());
  } else {
    *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
  }

  if (!*symbol) {
    const auto error_code = GetLastError();
    static constexpr DWORD bufferLength = 64 * 1024;
    std::wstring s(bufferLength, '\0');
    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, error_code,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   (LPWSTR)s.data(), 0, NULL);
    std::wostringstream oss;
    oss << L"Failed to find symbol " << ToWideString(symbol_name) << L" in library, error code: "
        << error_code << L" \"" << s.c_str() << L"\"";
    std::wstring errmsg = oss.str();
    // TODO: trim the ending '\r' and/or '\n'
    status = Status(common::ONNXRUNTIME, common::FAIL, ToUTF8String(errmsg));
  }

  return status;
}

std::string WindowsEnv::FormatLibraryFileName(const std::string& name, const std::string& version) const {
  ORT_UNUSED_PARAMETER(name);
  ORT_UNUSED_PARAMETER(version);
  ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
}

// \brief returns a provider that will handle telemetry on the current platform
const Telemetry& WindowsEnv::GetTelemetryProvider() const {
  return telemetry_provider_;
}

// \brief returns a value for the queried variable name (var_name)
std::string WindowsEnv::GetEnvironmentVar(const std::string& var_name) const {
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

/*
Read logical processor info from the map.
{-1,-1} stands for failure.
*/
ProcessorInfo WindowsEnv::GetProcessorAffinityMask(int global_processor_id) const {
  if (global_processor_info_map_.count(global_processor_id)) {
    return global_processor_info_map_.at(global_processor_id);
  } else {
    return {-1, -1};
  }
}

WindowsEnv::WindowsEnv() {
  InitializeCpuInfo();
}

/*
Discover all cores in a windows system.
Note - every "id" here, given it be group id, core id, or logical processor id, starts from 0.
*/
void WindowsEnv::InitializeCpuInfo() {
  DWORD returnLength = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &returnLength);
  auto last_error = GetLastError();
  if (last_error != ERROR_INSUFFICIENT_BUFFER) {
    const auto error_code = GetLastError();
    if (logging::LoggingManager::HasDefaultLogger()) {
      LOGS_DEFAULT(ERROR) << "Failed to calculate byte size for saving cpu info on windows"
                          << ", error code: " << error_code
                          << ", error msg: " << std::system_category().message(error_code);
    }
    return;
  }

  std::unique_ptr<char[]> allocation = std::make_unique<char[]>(returnLength);
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* processorInfos = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(allocation.get());

  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, processorInfos, &returnLength)) {
    const auto error_code = GetLastError();
    if (logging::LoggingManager::HasDefaultLogger()) {
      LOGS_DEFAULT(ERROR) << "Failed to fetch cpu info on windows"
                          << ", error code: " << error_code
                          << ", error msg: " << std::system_category().message(error_code);
    }
    return;
  }

  int core_id = 0;
  int global_processor_id = 0;
  const BYTE* iter = reinterpret_cast<const BYTE*>(processorInfos);
  const BYTE* end = iter + returnLength;
  std::stringstream log_stream;

  while (iter < end) {
    auto processor_info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(iter);
    auto size = processor_info->Size;

    // Discoverred a phyical core and it belongs exclusively to a single group
    if (processor_info->Relationship == RelationProcessorCore &&
        processor_info->Processor.GroupCount == 1) {
      log_stream << std::endl
                 << "core " << core_id + 1 << " consist of logical processors: ";
      LogicalProcessors core_global_proc_ids;
      constexpr KAFFINITY bit = 1;
      constexpr int id_upper_bound = sizeof(KAFFINITY) * CHAR_BIT;
      const auto& group_mask = processor_info->Processor.GroupMask[0];
      for (int logical_proessor_id = 0; logical_proessor_id < id_upper_bound; ++logical_proessor_id) {
        if (group_mask.Mask & (bit << logical_proessor_id)) {
          log_stream << global_processor_id + 1 << " ";
          core_global_proc_ids.push_back(global_processor_id);
          /*
           * Build up a map between global processor id and local processor id.
           * The map helps to bridge between ort API and windows affinity API -
           * we need local processor id to build an affinity mask for a particular group.
           */
          global_processor_info_map_.insert_or_assign(global_processor_id,
                                                      ProcessorInfo{static_cast<int>(group_mask.Group),
                                                                    logical_proessor_id});
          global_processor_id++;
        }
      }
      cores_.push_back(std::move(core_global_proc_ids));
      core_id++;
    }
    iter += size;
  }
  if (logging::LoggingManager::HasDefaultLogger()) {
    LOGS_DEFAULT(VERBOSE) << "Found total " << cores_.size() << " core(s) from windows system:";
    LOGS_DEFAULT(VERBOSE) << log_stream.str();
  }
}
}  // namespace onnxruntime
