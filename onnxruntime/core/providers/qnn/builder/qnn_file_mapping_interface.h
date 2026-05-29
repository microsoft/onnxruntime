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
  virtual ~FileMappingInterface() = default;

  virtual Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                              void** mapped_data_ptr) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime