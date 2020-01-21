// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <google/protobuf/stubs/status.h>
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"


namespace onnxruntime {
namespace server {

/**
 * A RAII container for MLValue buffers
 */
class MemBufferArray {
 public:
  MemBufferArray() = default;

  uint8_t* AllocNewBuffer(size_t tensor_length) {
    auto* data = new uint8_t[tensor_length];
    memset(data, 0, tensor_length);
    buffers_.push_back(data);
    return data;
  }

  ~MemBufferArray() {
    FreeBuffers();
  }

 private:
  std::vector<uint8_t*> buffers_;

  void FreeBuffers() {
    for (auto* buf : buffers_) {
      delete[] buf;
    }
  }
};

google::protobuf::util::Status GenerateProtobufStatus(const int& onnx_status, const std::string& message);


}  // namespace server
}  // namespace onnxruntime
