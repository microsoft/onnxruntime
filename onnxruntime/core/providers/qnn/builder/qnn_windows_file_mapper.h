// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/builder/qnn_file_mapping_callback_interface.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#include <string>
#include <unordered_map>

#include <QnnContext.h>

#include "core/platform/env.h"
#include "core/providers/qnn/rpcmem_library.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class WindowsFileMapper : public FileMappingCallbackInterface {
 public:
  explicit WindowsFileMapper(const logging::Logger& logger,
                             std::shared_ptr<qnn::RpcMemLibrary> rpcmem_lib);
  ~WindowsFileMapper() override;

  Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                      void** mapped_data_ptr) override;

  static void UnmapFile(void* addr) noexcept;

  Qnn_ErrorHandle_t MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                               Qnn_ContextBinaryDmaDataResponse_t* response,
                               void* mapped_data_ptr) override;
  Qnn_ErrorHandle_t ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                   void* mapped_data_ptr) override;

  Qnn_ErrorHandle_t MapRawData(Qnn_ContextBinaryDataRequest_t request,
                               Qnn_ContextBinaryRawDataResponse_t* response,
                               void* mapped_data_ptr) override;
  Qnn_ErrorHandle_t ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                   void* mapped_data_ptr) override;

 private:
  // A container of smart pointers of mapview memory pointers to mapped context bins
  std::vector<onnxruntime::Env::MappedMemoryPtr> mapped_memory_ptrs;
  const logging::Logger* logger_;
  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
