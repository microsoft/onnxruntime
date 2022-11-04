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

#include <fstream>
#include <string>
#include <thread>
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
    const KAFFINITY* affinities = p->thread_options.affinity.data();
    size_t num_affinity = p->thread_options.affinity.size() * sizeof(size_t) / sizeof(KAFFINITY);
    size_t offset = static_cast<size_t>(p->index) << 1;
    if (offset + 1 < num_affinity) {
      GROUP_AFFINITY thread_affinity;
      memset(&thread_affinity, 0x0, sizeof(GROUP_AFFINITY));
      thread_affinity.Group = static_cast<WORD>(affinities[offset]);
      thread_affinity.Mask = affinities[offset + 1];
      if (SetThreadGroupAffinity(GetCurrentThread(), &thread_affinity, nullptr)) {
        LOGS_DEFAULT(WARNING) << "Set group affinity for sub-thread " << p->index << ", "
                              << " group: " << thread_affinity.Group << ", "
                              << " mask: " << thread_affinity.Mask;
      } else {
        LOGS_DEFAULT(WARNING) << "Failed to set group affinity for sub-thread " << p->index << ", "
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
};

class WindowsEnv : public Env {
 public:

  explicit WindowsEnv() {
    group_info_vec_ = GetProcessorGroups();
  }

  EnvThread* CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) {
    return new WindowsThread(name_prefix, index, start_address, param, thread_options);
  }

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

  size_t GetDefaultThreadpoolSetting(std::vector<size_t>& affinities) const override {
    std::vector<KAFFINITY> kAffinities;
    KAFFINITY group_id = 0;
    for (const auto& group_info : this->group_info_vec_) {
      size_t bit_offset = 0;
      constexpr size_t bit_limit = sizeof(KAFFINITY) << 3;
      KAFFINITY active_processor_mask = group_info.ActiveProcessorMask;
      for (int i = 0; i < (group_info.ActiveProcessorCount >> 1); ++i) {
        KAFFINITY processor_mask = 0;
        size_t processor_count = 0;
        while (processor_count < 2 && bit_offset < bit_limit) {
          auto current_bit = (0x1 << bit_offset);
          if (active_processor_mask & current_bit) {
            processor_mask |= current_bit;
            processor_count++;
          }
          bit_offset++;
        }
        kAffinities.push_back(group_id);
        kAffinities.push_back(processor_mask);
      }
      group_id++;
    }
    if (kAffinities.empty()) {
      return std::thread::hardware_concurrency() >> 1;
    }
    size_t num_of_size_t = kAffinities.size() * sizeof(KAFFINITY) / sizeof(size_t);
    affinities.clear();
    affinities.resize(num_of_size_t);
    memcpy(affinities.data(), kAffinities.data(), num_of_size_t * sizeof(size_t));
    return kAffinities.size();
  }

  std::pair<KAFFINITY, KAFFINITY> GetGroupAffinity(int procssor_from, int processor_to) {
    if (procssor_from > processor_to) {
      LOGS_DEFAULT(ERROR) << "Processor <from> must be smaller or equal to <to>";
      return {-1, 0};
    }
    int processor_id = 1;
    int processor_count = 0;
    KAFFINITY group_id = 0;
    KAFFINITY processor_mask = 0;
    for (const auto& group_info : this->group_info_vec_) {
      for (int i = 0; i < group_info.ActiveProcessorCount; ++i, ++processor_id) {
        if (processor_id >= procssor_from && processor_id <= processor_to) {
          processor_mask |= 1ULL << i;
          processor_count += 1;
        }
      }
      if (processor_mask) {
        break;
      }
      group_id++;
    }
    if (processor_count < processor_to - procssor_from + 1) {
      LOGS_DEFAULT(ERROR) << "Processor <from> or <to> crossed group boundary";
      return {-1,0};
    }
    return {group_id, processor_mask};
  }

  std::pair<KAFFINITY, KAFFINITY> GetGroupAffinity(const std::vector<std::string>& processor_id_strs) {
    if (processor_id_strs.empty()) {
      return {-1, 0};
    }
    std::set<int> processor_ids;
    std::for_each(processor_id_strs.begin(),
                  processor_id_strs.end(), [&](const std::string& processor_id_str) {
                    processor_ids.insert( std::stoi(processor_id_str.c_str()));
                  });
    int processor_id = 1;
    int processor_count = 0;
    KAFFINITY group_id = 0;
    KAFFINITY processor_mask = 0;
    for (const auto& group_info : this->group_info_vec_) {
      for (int i = 0; i < group_info.ActiveProcessorCount; ++i, ++processor_id) {
        if (processor_ids.count(processor_id)) {
          processor_mask |= 1ULL << i;
          processor_count += 1;
        }
      }
      if (processor_mask) {
        break;
      }
      group_id++;
    }
    if (processor_count < processor_ids.size()) {
      LOGS_DEFAULT(ERROR) << "Processor id(s) crossed group boundary";
      return {-1, 0};
    }
    return {group_id, processor_mask};
  }

  std::vector<size_t> ReadThreadAffinityConfig(const std::string& affinity_str) {
    try {
      auto config_all_threads = SplitStr(affinity_str, ';');
      std::vector<KAFFINITY> affinities;
      for (const auto& config_per_threads : config_all_threads) {
        if (config_per_threads.empty()) {
          LOGS_DEFAULT(ERROR) << "Found empty affinity string!";
          return {};
        }
        auto partition_by_hyphen = SplitStr(config_per_threads, '-');
        if (partition_by_hyphen.size() == 2) {
          int from = stoi(partition_by_hyphen[0]);
          int to = stoi(partition_by_hyphen[1]);
          auto group_affinity = GetGroupAffinity(from, to);
          if (group_affinity.first == -1) {
            return {};
          }
          affinities.push_back(group_affinity.first);
          affinities.push_back(group_affinity.second);
        } else {
          auto partition_by_comma = SplitStr(config_per_threads, ',');
          if (partition_by_comma.empty()) {
            LOGS_DEFAULT(ERROR) << "Wrong affinity string format: " << config_per_threads;
            return {};
          } else {
            auto group_affinity = GetGroupAffinity(partition_by_comma);
            if (group_affinity.first == -1) {
              return {};
            }
            affinities.push_back(group_affinity.first);
            affinities.push_back(group_affinity.second);
          }
        }
      }
      size_t num_of_size_t = affinities.size() * sizeof(KAFFINITY) / sizeof(size_t);
      std::vector<size_t> ret(num_of_size_t, 0);
      memcpy(ret.data(), affinities.data(), num_of_size_t * sizeof(size_t));
      return ret;
    } catch (const std::exception& ex) {
      LOGS_DEFAULT(ERROR) << "Exception caught in WindowsEnv::ReadThreadAffinityConfig: " << ex.what();
    }
    return {};
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

  virtual Status LoadDynamicLibrary(const std::string& library_filename, void** handle) const override {
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

  virtual Status UnloadDynamicLibrary(void* handle) const override {
    if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to unload library, error code: ", error_code);
    }
    return Status::OK();
  }

  virtual Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
    if (!*symbol) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to find symbol in library, error code: ",
                             error_code);
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

  static std::vector<PROCESSOR_GROUP_INFO> GetProcessorGroups() {
    DWORD returnLength = 0;
    GetLogicalProcessorInformationEx(RelationGroup, nullptr, &returnLength);
    auto last_error = GetLastError();
    if (last_error != ERROR_INSUFFICIENT_BUFFER) {
      return {};
    }

    std::unique_ptr<char[]> allocation = std::make_unique<char[]>(returnLength);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* processorInfos = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(allocation.get());

    if (!GetLogicalProcessorInformationEx(RelationGroup, processorInfos, &returnLength)) {
      last_error = GetLastError();
      return {};
    }

    std::vector<PROCESSOR_GROUP_INFO> group_info_vec;
    auto num_processor_info = returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
    for (int64_t i = 0; i < static_cast<int64_t>(num_processor_info); ++i) {
      if (processorInfos[i].Relationship != RelationGroup) {
        return {};
      }
      for (int64_t j = 0; j < static_cast<int>(processorInfos[i].Group.ActiveGroupCount); ++j) {
        const auto& groupInfo = processorInfos[i].Group.GroupInfo[j];
        group_info_vec.push_back(groupInfo);
      }
    }
    return group_info_vec;
  }

 private:

  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  WindowsTelemetry telemetry_provider_;
  std::vector<PROCESSOR_GROUP_INFO> group_info_vec_;
};
}  // namespace

Env& Env::Default() {
  return WindowsEnv::Instance();
}

}  // namespace onnxruntime
