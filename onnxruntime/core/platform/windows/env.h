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
#include "core/platform/windows/telemetry.h"
#include "core/common/inlined_containers.h"
#include <Windows.h>

namespace onnxruntime {

/*
Logical processor information:
1. its belonging group;
2. its id within that belonging group.
{-1,-1} stands for an invalid processor info
*/
struct ProcessorInfo {
  int group_id = -1;
  int local_processor_id = -1;
};

/*
GlobalProcessorInfoMap is a map between global processor id
and pair<groupid, local processor id>, the latter is required
to be present during affinity setup.
*/
using GlobalProcessorInfoMap = InlinedHashMap<int, ProcessorInfo>;

class WindowsEnv : public Env {
 public:
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
  EnvThread* CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  void SleepForMicroseconds(int64_t micros) const override;
  static int DefaultNumCores();
  int GetNumPhysicalCpuCores() const override;
  std::vector<LogicalProcessors> GetDefaultThreadAffinities() const override;
  static WindowsEnv& Instance();
  PIDType GetSelfPid() const override;
  Status GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const override;
  common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const override;
  Status ReadFileIntoBuffer(_In_z_ const ORTCHAR_T* const file_path, const FileOffsetType offset, const size_t length,
                            const gsl::span<char> buffer) const override;
  Status MapFileIntoMemory(_In_z_ const ORTCHAR_T* file_path,
                           FileOffsetType offset,
                           size_t length,
                           MappedMemoryPtr& mapped_memory) const override;
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
  PathString GetRuntimePath() const override;
  Status LoadDynamicLibrary(const PathString& library_filename, bool /*global_symbols*/, void** handle) const override;
  Status UnloadDynamicLibrary(void* handle) const override;
  Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override;
  std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override;
  const Telemetry& GetTelemetryProvider() const override;
  std::string GetEnvironmentVar(const std::string& var_name) const override;
  ProcessorInfo GetProcessorAffinityMask(int global_processor_id) const;

 protected:
  /*
   * "cores_" host all physical cores dicoverred in a windows system.
   * Every LogicalProcessors represent a core of logical processors.
   * Specifically, LogicalProcessors is a vector of global processor id.
   * E.g.
   * Assume we have a system of 4 cores, each has 2 logical processors.
   * Then "cores_" will be like:
   * {
   * {0,1} // core 1
   * {2,3} // core 2
   * {4,5} // core 3
   * {6,7} // core 4
   * }
   * Further, assume we have a system of two groups, each has 4 cores like above,
   * then "cores_" will be like:
   * {
   * {0,1}   // core 1, group 1
   * {2,3}   // core 2, group 1
   * {4,5}   // core 3, group 1
   * {6,7}   // core 4, group 1
   * {8,9}   // core 5, group 2
   * {10,11} // core 6, group 2
   * {12,13} // core 7, group 2
   * {14,15} // core 8, group 2
   * }
   */
  std::vector<LogicalProcessors> cores_;
  /*
   * "global_processor_info_map_" is a map of:
   * global_processor_id <--> (group_id, local_processor_id)
   * "global_processor_id" means "index" of a logical processor in the system,
   * "local_processor_id" refers to "index" of a logical processor in its belonging group,
   * E.g.
   * Assume the system have two groups,
   * each group has two cores,
   * and each core has two logical processors.
   * then the global processor ids will be like:
   * 0,1,2,3,4,5,6,7
   * the local processor ids will be like:
   * 0,1,2,3,0,1,2,3
   */
  GlobalProcessorInfoMap global_processor_info_map_;
  WindowsEnv();

 private:
  void InitializeCpuInfo();
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  WindowsTelemetry telemetry_provider_;
};

}  // namespace onnxruntime
