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

WindowsFileMapper::WindowsFileMapper(const logging::Logger& logger,
                                     std::shared_ptr<qnn::RpcMemLibrary> rpcmem_lib)
    : logger_(&logger),
      rpcmem_lib_(rpcmem_lib) {
  ORT_ENFORCE(rpcmem_lib);
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

Qnn_ErrorHandle_t WindowsFileMapper::MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryDmaDataResponse_t* response,
                                                void* mapped_data_ptr) {
  if (mapped_data_ptr == nullptr) {
    LOGS(*logger_, ERROR) << "Attempting to map DMA data for null memory mapped pointer";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  LOGS(*logger_, INFO) << "Mapping DMA data for request: memory mapped pointer("
                       << mapped_data_ptr << "), offset(" << request.offset
                       << "), size(" << request.size << "), isBackendMappingNeeded("
                       << request.isBackendMappingNeeded << ")";

  auto buffer_size = request.size;
  if (buffer_size == 0 || !request.isBackendMappingNeeded) {
    LOGS(*logger_, ERROR) << "Mapping request size must be > 0 with backend mapping required";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // Align to nearest granularity boundary
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  Qnn_ContextBinarySize_t granularity = sys_info.dwAllocationGranularity;
  SIZE_T aligned_offset = request.offset & ~(granularity - 1);
  SIZE_T delta = request.offset - aligned_offset;

  void* aligned_data_ptr = static_cast<char*>(mapped_data_ptr) + aligned_offset;

  void* unaligned_data_ptr = static_cast<char*>(aligned_data_ptr) + delta;
  LOGS(*logger_, INFO) << "Created DMA data mapping with: address(" << aligned_data_ptr
                       << "), aligned offset(" << aligned_offset << "), delta(" << delta
                       << "), unaligned address(" << unaligned_data_ptr << ")";

  rpcmem_lib_->Api().register_buf(unaligned_data_ptr, buffer_size, NULL,
                                  rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_lib_->Api().to_fd(unaligned_data_ptr);
  if (fd == -1) {
    LOGS(*logger_, ERROR) << "Failed to register DMA data mapping to RPCMEM";
    return QNN_COMMON_ERROR_SYSTEM;
  }

  response->dmaBuffer.fd = fd;
  response->dmaBuffer.data = reinterpret_cast<void*>(unaligned_data_ptr);
  response->dataStartOffset = 0;
  response->alignedSize = buffer_size;

  return QNN_SUCCESS;
}

// Use LOGS_DEFAULT here as this function will be called during destruction of QnnBackendManager
// At time of destruction. Usage of logger_ will not be available and will result in a seg fault
Qnn_ErrorHandle_t WindowsFileMapper::ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                                    void* mapped_data_ptr) {
  if (mapped_data_ptr == nullptr) {
    LOGS_DEFAULT(ERROR) << "Attempting to release DMA data for null memory mapped pointer";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  LOGS_DEFAULT(INFO) << "Releasing DMA data mapping for memory mapped pointer("
                     << mapped_data_ptr << "), address(" << data_mem.dmaBuffer.data
                     << "), size: (" << data_mem.memSize << ")";

  if (data_mem.dmaBuffer.data == nullptr || data_mem.memSize == 0) {
    LOGS_DEFAULT(ERROR) << "Mapping release request address must not be null and size must be > 0";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  void* unaligned_data_ptr = data_mem.dmaBuffer.data;
  rpcmem_lib_->Api().register_buf(unaligned_data_ptr, data_mem.memSize, -1,
                                  rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_lib_->Api().to_fd(unaligned_data_ptr);
  if (fd != -1) {
    LOGS_DEFAULT(ERROR) << "Failed to deregister buffer from RPCMEM: " << unaligned_data_ptr;
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t WindowsFileMapper::MapRawData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryRawDataResponse_t* response,
                                                void* mapped_data_ptr) {
  ORT_UNUSED_PARAMETER(request);
  ORT_UNUSED_PARAMETER(response);
  ORT_UNUSED_PARAMETER(mapped_data_ptr);

  LOGS(*logger_, ERROR) << "File mapping for raw binary data is unsupported on Windows";
  return QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE;
}

// Use LOGS_DEFAULT for all clean up functions below as they will be called during destruction of
// QnnBackendManager at time of destruction. Usage of logger_ will not be available and will result
// in a seg fault
Qnn_ErrorHandle_t WindowsFileMapper::ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                                    void* mapped_data_ptr) {
  ORT_UNUSED_PARAMETER(data_mem);
  ORT_UNUSED_PARAMETER(mapped_data_ptr);

  LOGS_DEFAULT(ERROR) << "File mapping for raw binary data is unsupported on Windows";
  return QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE;
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
