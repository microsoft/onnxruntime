// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class FileMappingInterface {
 public:
  typedef struct MappedWeightInfo {
    size_t aligned_offset = 0;
    size_t delta = 0;
    void* aligned_data_ptr = nullptr;
    void* unaligned_data_ptr = nullptr;
  } MappedWeightInfo_t;

  virtual ~FileMappingInterface() = default;

  virtual Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                              void** mapped_data_ptr) = 0;

  virtual MappedWeightInfo_t GetMappedWeightMemoryPtr(void* mapped_base_ptr,
                                                      const size_t offset) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime