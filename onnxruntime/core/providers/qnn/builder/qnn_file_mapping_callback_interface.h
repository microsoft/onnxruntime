// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/builder/qnn_def.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_ENABLED

#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class FileMappingCallbackInterface {
 public:
  virtual ~FileMappingCallbackInterface() = default;
  virtual Status MapContextBin(const std::string& bin_filepath,
                               void** notify_param) = 0;
  virtual Status ReleaseContextBin(const std::string& model_name) = 0;

  virtual Status GetContextBinMappingPointer(const std::string& bin_filepath, void** mapping_ptr) = 0;

  virtual Status FreeContextBinMappingPointer(LPVOID bin_mapping_pointer) = 0;

  virtual Qnn_ErrorHandle_t MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                       Qnn_ContextBinaryDmaDataResponse_t* _response,
                                       void* notify_param) = 0;
  virtual Qnn_ErrorHandle_t ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                           void* notify_param) = 0;

  virtual Qnn_ErrorHandle_t MapRawData(Qnn_ContextBinaryDataRequest_t request,
                                       Qnn_ContextBinaryRawDataResponse_t* _response,
                                       void* notify_param) = 0;
  virtual Qnn_ErrorHandle_t ReleaseRawData(Qnn_ContextBinaryRawDataMem_t data_mem,
                                           void* notify_param) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_ENABLED