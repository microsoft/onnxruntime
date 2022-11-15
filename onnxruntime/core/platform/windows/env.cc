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

#include <Shlwapi.h>
#include <Windows.h>

#include <fstream>
#include <string>
#include <thread>
#include <numeric>
#include <process.h>
#include <fcntl.h>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/common/parse_string.h"
#include "core/platform/env.h"
#include "core/platform/scoped_resource.h"
#include "core/platform/windows/telemetry.h"
#include "unsupported/Eigen/CXX11/src/ThreadPool/ThreadPoolInterface.h"
#include <wil/Resource.h>

#include "core/platform/path_lib.h"  // for LoopDir()

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

namespace onnxruntime {

namespace {

class WindowsThread : public EnvThread {
 private:
  struct Param {
    const ORTCHAR_T* name_prefix;
    int index;
    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param);
    Eigen::ThreadPoolInterface* param;
    const ThreadOptions& thread_options;
  };

 public:
  WindowsThread(const ORTCHAR_T* name_prefix, int index,
                unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
                const ThreadOptions& thread_options)
      : hThread((HANDLE)_beginthreadex(nullptr, thread_options.stack_size, ThreadMain,
                                       new Param{name_prefix, index, start_address, param, thread_options}, 0,
                                       &threadID)) {
  }

  ~WindowsThread() {
    DWORD waitStatus = WaitForSingleObject(hThread.get(), INFINITE);
    FAIL_FAST_LAST_ERROR_IF(waitStatus == WAIT_FAILED);
  }

  // This function is called when the threadpool is cancelled.
  // TODO: Find a way to avoid calling TerminateThread
  void OnCancel() {
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    TerminateThread(hThread.get(), 1);
#endif
  }

 private:
  typedef HRESULT(WINAPI* SetThreadDescriptionFunc)(HANDLE hThread, PCWSTR lpThreadDescription);
  static unsigned __stdcall ThreadMain(void* param) {
    std::unique_ptr<Param> p((Param*)param);
    const auto& affinities = p->thread_options.affinity;
    size_t num_affinity = affinities.size();
    size_t offset = static_cast<size_t>(p->index) * 2;
    if (offset + 1 < num_affinity) {
      GROUP_AFFINITY thread_affinity;
      memset(&thread_affinity, 0x0, sizeof(GROUP_AFFINITY));
      thread_affinity.Group = static_cast<WORD>(affinities[offset]);
      thread_affinity.Mask = static_cast<KAFFINITY>(affinities[offset + 1]);
      if (SetThreadGroupAffinity(GetCurrentThread(), &thread_affinity, nullptr)) {
        LOGS_DEFAULT(WARNING) << "Set group affinity for thread " << p->index << ", "
                              << " group: " << thread_affinity.Group << ", "
                              << " mask: " << thread_affinity.Mask;
      } else {
        LOGS_DEFAULT(WARNING) << "Failed to set group affinity for thread " << p->index << ", "
                              << " group: " << thread_affinity.Group << ", "
                              << " mask: " << thread_affinity.Mask << ","
                              << " error code: " << GetLastError();
      }
    }

#if WINVER >= _WIN32_WINNT_WIN10
    constexpr SetThreadDescriptionFunc pSetThrDesc = SetThreadDescription;
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    // kernel32.dll is always loaded
    auto pSetThrDesc =
        (SetThreadDescriptionFunc)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "SetThreadDescription");
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
  unsigned threadID = 0;
  wil::unique_handle hThread;
};  //WindowsThread

}  // anonymouse namespace

WindowsEnv::WindowsEnv() {
  InitializeCpuInfo();
}

EnvThread* WindowsEnv::CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                                    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                                    Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) {
  return new WindowsThread(name_prefix, index, start_address, param, thread_options);
}

void WindowsEnv::SleepForMicroseconds(int64_t micros) const {
  Sleep(static_cast<DWORD>(micros) / 1000);
}

int WindowsEnv::GetNumCpuCores() const {
  if (cores_.empty()) {
    return static_cast<int>(std::thread::hardware_concurrency()) / 2;
  } else {
    return static_cast<int>(cores_.size());
  }
}

size_t WindowsEnv::GetDefaultThreadpoolSetting(std::vector<uint64_t>& affinities) const {
  ORT_ENFORCE(sizeof(KAFFINITY) <= sizeof(uint64_t), "KAFFINITY is bigger than uint64_t, cannot save it to an uint64_t vector");
  for (const auto& core : cores_) {
    affinities.push_back(core.group_id);
    affinities.push_back(core.processor_bitmask);
  }
  if (affinities.empty()) {
    return std::thread::hardware_concurrency() / 2;
  }
  return affinities.size() / 2;
}

// The function will go over group_vec_ searching for processors between [processor_from, processor_to]
// whenever found a match in a group, two things will happen sequentially:
// 1. Fill the pair of <group_id, processor_mask> for all matched processor in that group;
// 2. Break from the loop to stop searching the next group, this is because windows API will fail if the interval
//    spans across group boundaries.
// Note, pair::second == 0 stands for failure of getting the affinity
std::pair<KAFFINITY, KAFFINITY> WindowsEnv::GetGroupAffinity(int processor_from, int processor_to) const {
  if (processor_from > processor_to) {
    LOGS_DEFAULT(ERROR) << "Processor <from> must be smaller or equal to <to>";
    return {0, 0};
  }
  int processor_id = 1;
  int processor_count = 0;
  uint64_t group_id = 0;
  uint64_t processor_mask = 0;
  for (const auto& group_info : groups_) {
    for (int32_t i = 0; i < group_info.num_processors; ++i, ++processor_id) {
      if (processor_id >= processor_from && processor_id <= processor_to) {
        processor_mask |= BitOne << i;
        processor_count += 1;
      }
    }
    // when mask is set, do not visit next group
    if (processor_mask) {
      break;
    }
    group_id++;
  }
  if (processor_count < processor_to - processor_from + 1) {
    LOGS_DEFAULT(ERROR) << "Processor <from> or <to> cross group boundary";
    return {0, 0};
  }
  return {group_id, processor_mask};
}

// processor_id_strs are simply utf-8 strings
// Note, pair::second == 0 stands for failure of getting the affinity
std::pair<uint64_t, uint64_t> WindowsEnv::GetGroupAffinity(const std::vector<std::string>& processor_id_strs) const {
  if (processor_id_strs.empty()) {
    return {0, 0};
  }
  // use a set, in case of dups
  std::set<int> processor_ids;
  std::for_each(processor_id_strs.begin(),
                processor_id_strs.end(), [&](const std::string& processor_id_str) {
                  // stoi exception will be caught in the caller
                  if (IsDigit(processor_id_str)) {
                    processor_ids.insert(std::stoi(processor_id_str.c_str()));
                  } else {
                    LOGS_DEFAULT(ERROR) << "Found non-digit processor id str: " << processor_id_str;
                  }
                });
  if (processor_ids.size() != processor_id_strs.size()) {
    return {0, 0};
  }
  int processor_id = 1;
  int processor_count = 0;
  uint64_t group_id = 0;
  uint64_t processor_mask = 0;
  for (const auto& group_info : groups_) {
    for (int32_t i = 0; i < group_info.num_processors; ++i, ++processor_id) {
      if (processor_ids.count(processor_id)) {
        processor_mask |= BitOne << i;
        processor_count += 1;
      }
    }
    // when mask is set, do not visit next group
    if (processor_mask) {
      break;
    }
    group_id++;
  }
  if (processor_count < processor_ids.size()) {
    LOGS_DEFAULT(ERROR) << "Processor id(s) cross group boundary";
    return {0, 0};
  }
  return {group_id, processor_mask};
}

size_t WindowsEnv::ReadThreadAffinityConfig(const std::string& affinity_str, std::vector<uint64_t>& affinities) const {
  ORT_ENFORCE(sizeof(KAFFINITY) <= sizeof(uint64_t), "KAFFINITY is bigger than uint64_t, cannot save it to an uint64_t vector");
  if (affinity_str.empty()) {
    return 0;
  }
  try {
    affinities.clear();
    auto all_configs = SplitStr(affinity_str, ';');
    for (const auto& config : all_configs) {
      if (config.empty()) {
        LOGS_DEFAULT(ERROR) << "Found empty affinity string!";
        return 0;
      }
      auto partition_by_hyphen = SplitStr(config, '-');
      if (partition_by_hyphen.size() == 2) {
        if (!IsDigit(partition_by_hyphen[0]) || !IsDigit(partition_by_hyphen[1])) {
          LOGS_DEFAULT(ERROR) << "Found non-digit in affinity str: " << affinity_str;
          return 0;
        }
        int from = stoi(partition_by_hyphen[0]);
        int to = stoi(partition_by_hyphen[1]);
        auto group_affinity = GetGroupAffinity(from, to);
        if (group_affinity.second == 0) {
          return 0;
        }
        affinities.push_back(group_affinity.first);
        affinities.push_back(group_affinity.second);
      } else {
        auto partition_by_comma = SplitStr(config, ',');
        if (partition_by_comma.empty()) {
          LOGS_DEFAULT(ERROR) << "Wrong affinity string format: " << config;
          return 0;
        } else {
          auto group_affinity = GetGroupAffinity(partition_by_comma);
          if (group_affinity.second == 0) {
            return 0;
          }
          affinities.push_back(group_affinity.first);
          affinities.push_back(group_affinity.second);
        }
      }
    }
    if (affinities.empty()) {
      LOGS_DEFAULT(WARNING) << "no thread affinity setting can be read from affinity_str: " << affinity_str;
      return 0;
    }
    return affinities.size() / 2;
  } catch (const std::exception& ex) {
    LOGS_DEFAULT(ERROR) << "Exception caught in WindowsEnv::ReadThreadAffinityConfig: " << ex.what();
  }
  return 0;
}

WindowsEnv& WindowsEnv::Instance() {
  static WindowsEnv default_env;
  return default_env;
}

PIDType WindowsEnv::GetSelfPid() const {
  return GetCurrentProcessId();
}

Status WindowsEnv::GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const {
#if WINVER >= _WIN32_WINNT_WIN8
  wil::unique_hfile file_handle{
      CreateFile2(file_path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
#else
  wil::unique_hfile file_handle{
      CreateFileW(file_path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
#endif
  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const int err = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToMBString(file_path), " fail, errcode = ", err);
  }
  LARGE_INTEGER filesize;
  if (!GetFileSizeEx(file_handle.get(), &filesize)) {
    const int err = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetFileSizeEx ", ToMBString(file_path), " fail, errcode = ", err);
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
#if WINVER >= _WIN32_WINNT_WIN8
  wil::unique_hfile file_handle{
      CreateFile2(file_path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
#else
  wil::unique_hfile file_handle{
      CreateFileW(file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)};
#endif
  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const int err = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToMBString(file_path), " fail, errcode = ", err);
  }

  if (length == 0)
    return Status::OK();

  if (offset > 0) {
    LARGE_INTEGER current_position;
    current_position.QuadPart = offset;
    if (!SetFilePointerEx(file_handle.get(), current_position, &current_position, FILE_BEGIN)) {
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

    if (!ReadFile(file_handle.get(), buffer.data() + total_bytes_read, bytes_to_read, &bytes_read, nullptr)) {
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

Status WindowsEnv::MapFileIntoMemory(_In_z_ const ORTCHAR_T*, FileOffsetType, size_t, MappedMemoryPtr&) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "MapFileIntoMemory is not implemented on Windows.");
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
            const auto err = GetLastError();
            final_status = ORT_MAKE_STATUS(
                ONNXRUNTIME, FAIL,
                "DeleteFile() failed - path: ", ToMBString(child_path),
                ", error code: ", err);
          }
        }

        return final_status.IsOK();
      });

  ORT_RETURN_IF_ERROR(final_status);

  if (!RemoveDirectoryW(path.c_str())) {
    const auto err = GetLastError();
    final_status = ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "RemoveDirectory() failed - path: ", ToMBString(path),
        ", error code: ", err);
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

common::Status WindowsEnv::GetCanonicalPath(const PathString& path, PathString& canonical_path) const {
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
    const int err = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "open file ", ToMBString(path), " fail, errcode = ", err);
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
std::string WindowsEnv::GetRuntimePath() const {
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

Status WindowsEnv::LoadDynamicLibrary(const std::string& library_filename, void** handle) const {
#if WINAPI_FAMILY == WINAPI_FAMILY_PC_APP
  *handle = ::LoadPackagedLibrary(ToWideString(library_filename).c_str(), 0);
#else
  *handle = ::LoadLibraryExA(library_filename.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
  if (!*handle) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load library, error code: ", error_code);
  }
  return Status::OK();
}

Status WindowsEnv::UnloadDynamicLibrary(void* handle) const {
  if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to unload library, error code: ", error_code);
  }
  return Status::OK();
}

Status WindowsEnv::GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const {
  *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
  if (!*symbol) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to find symbol in library, error code: ",
                           error_code);
  }
  return Status::OK();
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

void WindowsEnv::InitializeCpuInfo() {
  if (sizeof(KAFFINITY) > sizeof(uint64_t)) {  // exit if KAFFINITY is bigger than uint64_t, this is unlikely though
    return;
  }
  DWORD returnLength = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &returnLength);
  auto last_error = GetLastError();
  if (last_error != ERROR_INSUFFICIENT_BUFFER) {
    return;
  }

  std::unique_ptr<char[]> allocation = std::make_unique<char[]>(returnLength);
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* processorInfos = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(allocation.get());

  if (!GetLogicalProcessorInformationEx(RelationAll, processorInfos, &returnLength)) {
    return;
  }

  const BYTE* iter = reinterpret_cast<const BYTE*>(processorInfos);
  const BYTE* end = iter + returnLength;
  while (iter < end) {
    auto processor_info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(iter);
    auto size = processor_info->Size;
    if (processor_info->Relationship == RelationGroup) {
      for (int i = 0; i < static_cast<int>(processor_info->Group.ActiveGroupCount); ++i) {
        Group group{static_cast<int32_t>(processor_info->Group.GroupInfo[i].ActiveProcessorCount),
                    static_cast<uint64_t>(processor_info->Group.GroupInfo[i].ActiveProcessorMask)};
        groups_.push_back(std::move(group));
      }
    } else if (processor_info->Relationship == RelationProcessorCore &&
               processor_info->Processor.GroupCount == 1) {
      Core core{static_cast<uint64_t>(processor_info->Processor.GroupMask[0].Group),
                static_cast<uint64_t>(processor_info->Processor.GroupMask[0].Mask)};
      cores_.push_back(std::move(core));
    }
    iter += size;
  }
}

Env& Env::Default() {
  return WindowsEnv::Instance();
}

}  // namespace onnxruntime
