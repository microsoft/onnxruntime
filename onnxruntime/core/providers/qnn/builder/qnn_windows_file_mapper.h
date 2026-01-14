// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/builder/qnn_file_mapping_callback_interface.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_ENABLED

#include <string>
#include <unordered_map>

#include <QnnContext.h>

#include "core/providers/qnn/rpcmem_library.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class WindowsFileMapper : public FileMappingCallbackInterface {
 public:
  explicit WindowsFileMapper(const logging::Logger& logger);
  ~WindowsFileMapper() override;
  Status MapContextBin(const std::string& bin_filepath,
                       void** notify_param) override;
  Status ReleaseContextBin(const std::string& model_name) override;

  Status GetContextBinMappingPointer(const std::string& bin_filepath, void** mapping_ptr) override;

  Status FreeContextBinMappingPointer(LPVOID bin_mapping_pointer) override;

  Qnn_ErrorHandle_t MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                               Qnn_ContextBinaryDmaDataResponse_t* response,
                               void* notify_param) override;
  Qnn_ErrorHandle_t ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                   void* notify_param) override;

  Qnn_ErrorHandle_t MapRawData(Qnn_ContextBinaryDataRequest_t request,
                               Qnn_ContextBinaryRawDataResponse_t* response,
                               void* notify_param) override;
  Qnn_ErrorHandle_t ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                   void* notify_param) override;

 private:
  typedef struct MappedDataInfo {
    LPVOID aligned_data_ptr = nullptr;
    size_t buffer_size = 0;
  } MappedDataInfo_t;

  typedef struct MappingInfo {
    HANDLE file_handle;

    // Maps unaligned data pointers to aligned data pointers
    std::unordered_map<LPVOID, MappedDataInfo_t> mapped_data;
  } MappingInfo_t;

  void CleanUpDataMapping(LPVOID unaligned_data_ptr, LPVOID aligned_data_ptr,
                          size_t buffer_size);
  void CleanUpDataMappings(const std::unordered_map<LPVOID, MappedDataInfo_t>& mapped_data);
  void CloseHandles(HANDLE file_handle, HANDLE file_mapping_handle);

  std::mutex map_mutex_;  // Applies to both unordered maps
  std::unordered_map<std::string, HANDLE> context_bin_to_mapping_handle_map_;
  std::unordered_map<HANDLE, MappingInfo_t> mapping_handle_to_info_map_;
  std::unordered_set<LPVOID> context_bin_map_view_pointers_;

  const logging::Logger* logger_;

  RpcMemLibrary rpcmem_lib_;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_ENABLED
