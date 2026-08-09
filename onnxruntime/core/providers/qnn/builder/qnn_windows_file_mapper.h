// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/builder/qnn_file_mapping_interface.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class WindowsFileMapper : public FileMappingInterface {
 public:
  explicit WindowsFileMapper(const logging::Logger& logger);
  ~WindowsFileMapper() override;

  // Creates a file mapping of the context binary and returns the
  // mapview pointer of the file mapping
  Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                      void** mapped_data_ptr) override;

 private:
  // A container of smart pointers of mapview memory pointers to mapped context bins
  // key: filepath to context bin, value: smart pointer of mapview memory pointers
  std::mutex map_mutex_;
  std::unordered_map<std::string, onnxruntime::Env::MappedMemoryPtr> mapped_memory_ptrs_;
  const logging::Logger* logger_;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
