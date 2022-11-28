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

namespace onnxruntime {

class PosixEnv : public Env {
 public:
  static PosixEnv& Instance(); 
  EnvThread* CreateThread(const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) override;
  // we are guessing the number of phys cores based on a popular HT case.
  static int DefaultNumCores();
  // Return the number of physical cores
  int GetNumPhysicalCpuCores() const override;
  std::vector<LogicalProcessors> GetDefaultThreadAffinities() const override;
  void SleepForMicroseconds(int64_t micros) const override;
  PIDType GetSelfPid() const override;
  Status GetFileLength(const PathChar* file_path, size_t& length) const override;
  common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const override;
  Status ReadFileIntoBuffer(const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                            gsl::span<char> buffer) const override;
  Status MapFileIntoMemory(const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                           MappedMemoryPtr& mapped_memory) const override;
  static common::Status ReportSystemError(const char* operation_name, const std::string& path);
  bool FolderExists(const std::string& path) const override;
  common::Status CreateFolder(const std::string& path) const override;
  common::Status DeleteFolder(const PathString& path) const override; 
  common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const override;
  common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const override;
  common::Status FileClose(int fd) const override;
  common::Status GetCanonicalPath(const PathString& path, PathString& canonical_path) const override;
  common::Status LoadDynamicLibrary(const std::string& library_filename, bool global_symbols, void** handle) const override;
  common::Status UnloadDynamicLibrary(void* handle) const override;
  common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override;
  std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override;
  // \brief returns a provider that will handle telemetry on the current platform
  const Telemetry& GetTelemetryProvider() const override;
  // \brief returns a value for the queried variable name (var_name)
  std::string GetEnvironmentVar(const std::string& var_name) const override;

 protected:
  PosixEnv() = default;
 
 private:
  Telemetry telemetry_provider_;
#ifdef CPUINFO_SUPPORTED
  bool cpuinfo_available_{false};
#endif
};

} // namespace onnxruntime

