// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#include <wil/filesystem.h>

#include <utility>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

WindowsFileMapper::WindowsFileMapper(const logging::Logger& logger)
    : logger_(&logger) {
}

WindowsFileMapper::~WindowsFileMapper() {
}

static void UnmapFile(void* addr) noexcept {
  bool successful = UnmapViewOfFile(addr);
  if (!successful) {
    const auto error_code = GetLastError();
    LOGS_DEFAULT(ERROR) << "Failed to unmap view of file with ptr: " << addr
                        << ", Error code: " << error_code << ", \""
                        << std::system_category().message(error_code) << "\"";
  }
}

Status WindowsFileMapper::GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                                       void** mapped_data_ptr) {
  LOGS(*logger_, INFO) << "Creating context bin file mapping for "
                       << bin_filepath;

  ORT_RETURN_IF(bin_filepath.empty(), "Context bin file path is empty");

  std::lock_guard<std::mutex> lock(map_mutex_);
  auto map_it = mapped_memory_ptrs_.find(bin_filepath);
  if (map_it != mapped_memory_ptrs_.end()) {
    *mapped_data_ptr = map_it->second.get();
    LOGS(*logger_, INFO) << "Found existing mapview memory pointer (" << mapped_data_ptr
                         << ") for context bin file: " << bin_filepath;
    return Status::OK();
  }

  std::wstring bin_filepath_wstr(bin_filepath.begin(), bin_filepath.end());
  wil::unique_hfile file_handle{CreateFile2(bin_filepath_wstr.c_str(),
                                            GENERIC_READ,
                                            FILE_SHARE_READ,
                                            OPEN_EXISTING,
                                            NULL)};
  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create file handle for context bin", bin_filepath,
                           ". Error code: ", error_code, ", \"",
                           std::system_category().message(error_code), "\"");
  }

  LOGS(*logger_, VERBOSE) << "Created file handle (" << file_handle.get() << ") for context bin: "
                          << bin_filepath;

  wil::unique_hfile file_mapping_handle{CreateFileMappingW(file_handle.get(),
                                                           nullptr,
                                                           PAGE_READONLY,
                                                           0x00,
                                                           0x00,
                                                           nullptr)};
  if (file_mapping_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create file mapping handle for context bin",
                           bin_filepath, ". Error code: ", error_code, ", \"",
                           std::system_category().message(error_code), "\"");
  }

  LOGS(*logger_, VERBOSE) << "Created file mapping with handle (" << file_mapping_handle.get()
                          << ") for context bin:" << bin_filepath;

  void* const mapped_base_ptr = MapViewOfFile(file_mapping_handle.get(),
                                              FILE_MAP_READ,
                                              0, 0, 0);

  if (mapped_base_ptr == nullptr) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to retrieve mapview pointer for context bin",
                           bin_filepath, ". Error code: ", error_code, ", \"",
                           std::system_category().message(error_code), "\"");
  }

  LOGS(*logger_, INFO) << "Created mapview pointer with address " << mapped_base_ptr
                       << " for context bin " << bin_filepath;

  onnxruntime::Env::MappedMemoryPtr mapped_memory_ptr{reinterpret_cast<char*>(mapped_base_ptr),
                                                      [mapped_base_ptr](void*) {
                                                        UnmapFile(mapped_base_ptr);
                                                      }};

  *mapped_data_ptr = mapped_memory_ptr.get();
  mapped_memory_ptrs_.emplace(bin_filepath, std::move(mapped_memory_ptr));

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
