// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_ENABLED

#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime {
namespace qnn {

WindowsFileMapper::WindowsFileMapper(const logging::Logger& logger) : logger_(&logger) {
}

// Close all handles and registered buffers
// Use LOGS_DEFAULT here as this function will be called during destruction of QnnBackendManager
// At time of destruction. Usage of logger_ will not be available and will result in a seg fault
WindowsFileMapper::~WindowsFileMapper() {
  std::lock_guard<std::mutex> lock(map_mutex_);

  // Ideally, there should be nothing to clean up at this point
  // but free any resources anyway if applicable
  if (!mapping_handle_to_info_map_.empty() || !context_bin_map_view_pointers_.empty()) {
    LOGS_DEFAULT(WARNING) << "File mapping resources still exist. Attempting to free all resources.";
  }

  context_bin_to_mapping_handle_map_.clear();

  for (auto& mapview_ptr : context_bin_map_view_pointers_) {
    CleanUpDataMapping(mapview_ptr, nullptr, 0);
  }

  for (auto& kv : mapping_handle_to_info_map_) {
    HANDLE file_mapping_handle = kv.first;
    auto& mapping_info = kv.second;

    CleanUpDataMappings(mapping_info.mapped_data);
    CloseHandles(mapping_info.file_handle, file_mapping_handle);
  }
  mapping_handle_to_info_map_.clear();
}

Status WindowsFileMapper::MapContextBin(const std::string& bin_filepath,
                                        void** notify_param) {
  LOGS(*logger_, INFO) << "Creating context bin file mapping for "
                       << bin_filepath;

  ORT_RETURN_IF(bin_filepath.empty(), "Context bin file path is empty");

  std::lock_guard<std::mutex> lock(map_mutex_);

  HANDLE file_handle = CreateFileA(bin_filepath.c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   NULL);
  ORT_RETURN_IF(file_handle == INVALID_HANDLE_VALUE,
                "Failed to create file handle for context bin",
                bin_filepath);

  LOGS(*logger_, VERBOSE) << "Created file handle (" << file_handle << ") for context bin: "
                          << bin_filepath;

  HANDLE file_mapping_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0x00, 0x00, NULL);
  ORT_RETURN_IF(file_mapping_handle == INVALID_HANDLE_VALUE,
                "Failed to create file mapping for context bin ", bin_filepath);

  LOGS(*logger_, INFO) << "Created file mapping with handle (" << file_mapping_handle << ") for context bin:"
                       << bin_filepath;

  auto inserted = context_bin_to_mapping_handle_map_[bin_filepath] = file_mapping_handle;
  if (!inserted) {
    CloseHandles(file_handle, file_mapping_handle);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add file handle mapping for context bin: ",
                           bin_filepath);
  }
  mapping_handle_to_info_map_.insert({file_mapping_handle, {file_handle, {}}});

  *notify_param = reinterpret_cast<void*>(file_mapping_handle);
  return Status::OK();
}

Status WindowsFileMapper::ReleaseContextBin(const std::string& bin_filepath) {
  LOGS(*logger_, INFO) << "Removing context bin file mapping for "
                       << bin_filepath;
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto status = Status::OK();

  auto bin_map_it = std::find_if(context_bin_to_mapping_handle_map_.begin(),
                                 context_bin_to_mapping_handle_map_.end(),
                                 [&bin_filepath](const auto& kv) {
                                   return kv.first == bin_filepath;
                                 });

  if (bin_map_it == context_bin_to_mapping_handle_map_.end()) {
    LOGS(*logger_, VERBOSE) << "File handle does not exist for " << bin_filepath;
    return status;
  }

  HANDLE file_mapping_handle = bin_map_it->second;
  auto mapping_it = std::find_if(mapping_handle_to_info_map_.begin(),
                                 mapping_handle_to_info_map_.end(),
                                 [file_mapping_handle](const auto& kv) {
                                   return kv.first == file_mapping_handle;
                                 });

  HANDLE file_handle = nullptr;
  auto it = mapping_handle_to_info_map_.find(file_mapping_handle);
  if (it == mapping_handle_to_info_map_.end()) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "File mapping information does not exist for file mapping handle: ",
                             file_mapping_handle, ", context bin: ", bin_filepath);
  } else {
    MappingInfo_t& mapping_info = it->second;
    file_handle = mapping_info.file_handle;
    auto mapped_data = mapping_info.mapped_data;

    if (!mapped_data.empty()) {
      LOGS(*logger_, WARNING) << "Attemping to remove context bin: " << bin_filepath
                              << ", but data regions still need to be unmapped. "
                              << "Proceeding with unmapping.";
      CleanUpDataMappings(mapped_data);
    }
  }

  // Will ignore handles that are null
  CloseHandles(file_handle, file_mapping_handle);
  ORT_UNUSED_PARAMETER(mapping_handle_to_info_map_.erase(file_mapping_handle));
  ORT_UNUSED_PARAMETER(context_bin_to_mapping_handle_map_.erase(bin_filepath));

  return Status::OK();
}

Status WindowsFileMapper::GetContextBinMappingPointer(const std::string& bin_filepath, void** mapping_ptr) {
  LOGS(*logger_, INFO) << "Creating mapping pointer for " << bin_filepath;

  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = std::find_if(context_bin_to_mapping_handle_map_.begin(),
                         context_bin_to_mapping_handle_map_.end(),
                         [&bin_filepath](const auto& kv) {
                           return kv.first == bin_filepath;
                         });

  ORT_RETURN_IF(it == context_bin_to_mapping_handle_map_.end(),
                "Failed to create mapping pointer: File mapping does not exist for ",
                bin_filepath);

  HANDLE& file_mapping_handle = it->second;

  LPVOID mapview_ptr = MapViewOfFile(file_mapping_handle,
                                     FILE_MAP_READ,
                                     0, 0, 0);

  ORT_RETURN_IF(mapview_ptr == nullptr, "Failed to create mapping pointer for ", bin_filepath);

  if (!context_bin_map_view_pointers_.insert(mapview_ptr).second) {
    LOGS(*logger_, ERROR) << "Unable to insert mapping pointer " << mapview_ptr << " into set";
    if (!UnmapViewOfFile(mapview_ptr)) {
      LOGS(*logger_, ERROR) << "Failed to unmap mapping pointer: " << mapview_ptr;
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create mapping pointer for ", bin_filepath);
  }

  *mapping_ptr = mapview_ptr;
  LOGS(*logger_, INFO) << "Created mapping pointer (" << mapping_ptr << ") for " << bin_filepath;
  return Status::OK();
}

Status WindowsFileMapper::FreeContextBinMappingPointer(LPVOID bin_mapping_pointer) {
  LOGS(*logger_, INFO) << "Releasing mapping pointer " << bin_mapping_pointer;

  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = std::find_if(context_bin_map_view_pointers_.begin(),
                         context_bin_map_view_pointers_.end(),
                         [bin_mapping_pointer](const auto& pointer) {
                           return pointer == bin_mapping_pointer;
                         });

  ORT_RETURN_IF(it == context_bin_map_view_pointers_.end(), "Mapping pointer ",
                bin_mapping_pointer, " cannot be found and is invalid");

  ORT_RETURN_IF(!UnmapViewOfFile(bin_mapping_pointer), "Failed to free mapping pointer ", bin_mapping_pointer);
  ORT_UNUSED_PARAMETER(context_bin_map_view_pointers_.erase(bin_mapping_pointer));
  return Status::OK();
}

Qnn_ErrorHandle_t WindowsFileMapper::MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryDmaDataResponse_t* response,
                                                void* notify_param) {
  if (notify_param == nullptr) {
    LOGS(*logger_, ERROR) << "Attempting to map DMA data for null mapping handle";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> lock(map_mutex_);
  HANDLE file_mapping_handle = reinterpret_cast<HANDLE>(notify_param);
  LOGS(*logger_, INFO) << "Mapping DMA data for request: mapping handle("
                       << file_mapping_handle << "), offset(" << request.offset
                       << "), size(" << request.size << "), isBackendMappingNeeded("
                       << request.isBackendMappingNeeded << ")";

  auto buffer_size = request.size;
  if (buffer_size == 0 || !request.isBackendMappingNeeded) {
    LOGS(*logger_, ERROR) << "Mapping request size must be > 0 with backend mapping required";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto it = mapping_handle_to_info_map_.find(file_mapping_handle);
  if (it == mapping_handle_to_info_map_.end()) {
    LOGS(*logger_, ERROR) << "File mapping info not found for mapping handle: " << file_mapping_handle;
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  MappingInfo_t& mapping_info = it->second;

  // Align to nearest granularity boundary
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  Qnn_ContextBinarySize_t granularity = sys_info.dwAllocationGranularity;
  SIZE_T aligned_offset = request.offset & ~(granularity - 1);
  SIZE_T delta = request.offset - aligned_offset;

  LPVOID aligned_data_ptr = MapViewOfFile(file_mapping_handle,
                                          FILE_MAP_READ,
                                          (aligned_offset >> 32),
                                          (aligned_offset & 0xFFFFFFFF),
                                          (buffer_size + delta));

  if (aligned_data_ptr == nullptr) {
    LOGS(*logger_, ERROR) << "Failed to map DMA data for file mapping handle ";
    return QNN_COMMON_ERROR_SYSTEM;
  }

  LPVOID unaligned_data_ptr = static_cast<char*>(aligned_data_ptr) + delta;
  LOGS(*logger_, INFO) << "Created DMA data mapping with: address(" << aligned_data_ptr
                       << "), aligned offset(" << aligned_offset << "), delta(" << delta
                       << "), unaligned address(" << unaligned_data_ptr << ")";

  rpcmem_lib_.Api().register_buf(unaligned_data_ptr, buffer_size, NULL,
                                 rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_lib_.Api().to_fd(unaligned_data_ptr);
  if (fd == -1) {
    LOGS(*logger_, ERROR) << "Failed to register DMA data mapping to RPCMEM";
    if (!UnmapViewOfFile(aligned_data_ptr)) {
      LOGS(*logger_, ERROR) << "Failed to unmap DMA data with address: " << aligned_data_ptr;
    }
    return QNN_COMMON_ERROR_SYSTEM;
  }

  mapping_info.mapped_data.insert({unaligned_data_ptr, {aligned_data_ptr, buffer_size}});
  response->dmaBuffer.fd = fd;
  response->dmaBuffer.data = reinterpret_cast<void*>(unaligned_data_ptr);
  response->dataStartOffset = 0;
  response->alignedSize = buffer_size;

  return QNN_SUCCESS;
}

// Use LOGS_DEFAULT here as this function will be called during destruction of QnnBackendManager
// At time of destruction. Usage of logger_ will not be available and will result in a seg fault
Qnn_ErrorHandle_t WindowsFileMapper::ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                                    void* notify_param) {
  if (notify_param == nullptr) {
    LOGS_DEFAULT(ERROR) << "Attempting to release DMA data for null mapping handle";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> lock(map_mutex_);
  HANDLE file_mapping_handle = static_cast<HANDLE>(notify_param);
  LOGS_DEFAULT(INFO) << "Releasing DMA data mapping for mapping handle(" << file_mapping_handle
                     << "), address(" << data_mem.dmaBuffer.data << "), size: ("
                     << data_mem.memSize << ")";

  if (data_mem.dmaBuffer.data == nullptr || data_mem.memSize == 0) {
    LOGS_DEFAULT(ERROR) << "Mapping release request address must not be null and size must be > 0";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto mapping_info_it = mapping_handle_to_info_map_.find(file_mapping_handle);
  if (mapping_info_it == mapping_handle_to_info_map_.end()) {
    LOGS_DEFAULT(ERROR) << "File mapping info not found for mapping handle: " << file_mapping_handle;
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  MappingInfo_t& mapping_info = mapping_info_it->second;

  LPVOID unaligned_data_ptr = reinterpret_cast<void*>(data_mem.dmaBuffer.data);
  auto& mapped_data = mapping_info.mapped_data;
  auto mapped_data_it = std::find_if(mapped_data.begin(), mapped_data.end(),
                                     [unaligned_data_ptr](const auto& kv) {
                                       return kv.first == unaligned_data_ptr;
                                     });

  if (mapped_data_it == mapped_data.end()) {
    LOGS_DEFAULT(ERROR) << "Failed to find DMA data mapping for address: " << unaligned_data_ptr;
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  LPVOID aligned_data_ptr = mapped_data_it->second.aligned_data_ptr;

  CleanUpDataMapping(unaligned_data_ptr, aligned_data_ptr, data_mem.memSize);
  if (!mapped_data.erase(unaligned_data_ptr)) {
    LOGS_DEFAULT(WARNING) << "Possible leak: failed to remove unordered_map entry for DMA data address: "
                          << unaligned_data_ptr;
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t WindowsFileMapper::MapRawData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryRawDataResponse_t* response,
                                                void* notify_param) {
  ORT_UNUSED_PARAMETER(request);
  ORT_UNUSED_PARAMETER(response);
  ORT_UNUSED_PARAMETER(notify_param);

  LOGS(*logger_, ERROR) << "File mapping for raw binary data is unsupported on Windows";
  return QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE;
}

// Use LOGS_DEFAULT for all clean up functions below as they will be called during destruction of
// QnnBackendManagerAt time of destruction. Usage of logger_ will not be available and will result
// in a seg fault
Qnn_ErrorHandle_t WindowsFileMapper::ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                                    void* notify_param) {
  ORT_UNUSED_PARAMETER(data_mem);
  ORT_UNUSED_PARAMETER(notify_param);

  LOGS_DEFAULT(ERROR) << "File mapping for raw binary data is unsupported on Windows";
  return QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE;
}

void WindowsFileMapper::CleanUpDataMapping(LPVOID unaligned_data_ptr, LPVOID aligned_data_ptr,
                                           size_t buffer_size) {
  if (unaligned_data_ptr) {
    // Set file descriptor to -1 to signal deregistration
    rpcmem_lib_.Api().register_buf(unaligned_data_ptr, buffer_size, -1,
                                   rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

    auto fd = rpcmem_lib_.Api().to_fd(unaligned_data_ptr);
    if (fd != -1) {
      LOGS_DEFAULT(ERROR) << "Failed to deregister buffer from RPCMEM: " << unaligned_data_ptr;
    }
  }

  if (aligned_data_ptr && !UnmapViewOfFile(aligned_data_ptr)) {
    LOGS_DEFAULT(ERROR) << "Failed to unmap view of pointer: " << aligned_data_ptr;
  }
}

void WindowsFileMapper::CleanUpDataMappings(const std::unordered_map<LPVOID, MappedDataInfo_t>& mapped_data) {
  // Key is unaligned data pointer
  for (const auto& kv : mapped_data) {
    auto mapped_data_info = kv.second;
    // Will handle null ptrs
    CleanUpDataMapping(kv.first, mapped_data_info.aligned_data_ptr,
                       mapped_data_info.buffer_size);
  }
}

void WindowsFileMapper::CloseHandles(HANDLE file_handle, HANDLE file_mapping_handle) {
  if (file_mapping_handle && !CloseHandle(file_mapping_handle)) {
    LOGS_DEFAULT(ERROR) << "Failed to close file mapping handle: " << file_mapping_handle;
  }
  if (file_handle && !CloseHandle(file_handle)) {
    LOGS_DEFAULT(ERROR) << "Failed to close file handle: " << file_handle;
  }
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_ENABLED
