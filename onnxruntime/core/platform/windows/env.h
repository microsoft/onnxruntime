// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/platform/windows/telemetry.h"
#include <Windows.h>

namespace onnxruntime {

struct Core {
  uint64_t group_id = 0;
  uint64_t processor_bitmask = 0;
};

struct Group {
  int32_t num_processors = 0;
  uint64_t processor_bitmask = 0;
};

class WindowsEnv : public Env {
 public:
  WindowsEnv();
  EnvThread* CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options);
  void SleepForMicroseconds(int64_t micros) const override;
  int GetNumCpuCores() const override;
  size_t GetDefaultThreadpoolSetting(std::vector<uint64_t>& affinities) const override;
  std::pair<KAFFINITY, KAFFINITY> GetGroupAffinity(int processor_from, int processor_to) const;
  std::pair<uint64_t, uint64_t> GetGroupAffinity(const std::vector<std::string>& processor_id_strs) const;
  size_t ReadThreadAffinityConfig(const std::string& affinity_str, std::vector<uint64_t>& affinities) const override;
  static WindowsEnv& Instance();
  PIDType GetSelfPid() const override;
  Status GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const override;
  common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const override;
  Status ReadFileIntoBuffer(_In_z_ const ORTCHAR_T* const file_path, const FileOffsetType offset, const size_t length,
                            const gsl::span<char> buffer) const override;
  Status MapFileIntoMemory(_In_z_ const ORTCHAR_T*, FileOffsetType, size_t, MappedMemoryPtr&) const override;
  bool FolderExists(const std::wstring& path) const override;
  bool FolderExists(const std::string& path) const override;
  common::Status CreateFolder(const std::wstring& path) const override;
  common::Status CreateFolder(const std::string& path) const override;
  common::Status DeleteFolder(const PathString& path) const override;
  common::Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) const override;
  common::Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) const override;
  common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const override;
  common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const override;
  common::Status FileClose(int fd) const override;
  common::Status GetCanonicalPath(const PathString& path, PathString& canonical_path) const override;
  std::string GetRuntimePath() const override;
  Status LoadDynamicLibrary(const std::string& library_filename, void** handle) const override;
  Status UnloadDynamicLibrary(void* handle) const override;
  Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override;
  std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override;
  const Telemetry& GetTelemetryProvider() const override;
  std::string GetEnvironmentVar(const std::string& var_name) const override;

 protected:
  std::vector<Core> cores_;
  std::vector<Group> groups_;
  static constexpr uint64_t BitOne = 1;

 private:
  void InitializeCpuInfo();
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  WindowsTelemetry telemetry_provider_;

};

}  // namespace onnxruntime