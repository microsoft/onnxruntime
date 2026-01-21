// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class FileMappingCallbackInterface {
 public:
  virtual ~FileMappingCallbackInterface() = default;

  virtual Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                              void** mapped_data_ptr) = 0;

  virtual Qnn_ErrorHandle_t MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                       Qnn_ContextBinaryDmaDataResponse_t* response,
                                       void* mapped_data_ptr) = 0;
  virtual Qnn_ErrorHandle_t ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                           void* mapped_data_ptr) = 0;

  virtual Qnn_ErrorHandle_t MapRawData(Qnn_ContextBinaryDataRequest_t request,
                                       Qnn_ContextBinaryRawDataResponse_t* response,
                                       void* mapped_data_ptr) = 0;
  virtual Qnn_ErrorHandle_t ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                           void* mapped_data_ptr) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime