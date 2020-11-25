// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>
#include <google/protobuf/stubs/status.h>

#include "util.h"

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

protobufutil::Status GenerateProtobufStatus(const int& onnx_status, const std::string& message) {
  protobufutil::error::Code code = protobufutil::error::Code::UNKNOWN;
  switch (onnx_status) {
    case ORT_OK:
    case ORT_MODEL_LOADED:
      code = protobufutil::error::Code::OK;
      break;
    case ORT_FAIL:
    case ORT_INVALID_ARGUMENT:
    case ORT_INVALID_PROTOBUF:
    case ORT_INVALID_GRAPH:
    case ORT_NO_SUCHFILE:
    case ORT_NO_MODEL:
      code = protobufutil::error::Code::INVALID_ARGUMENT;
      break;
    case ORT_NOT_IMPLEMENTED:
      code = protobufutil::error::Code::UNIMPLEMENTED;
      break;
    case ORT_RUNTIME_EXCEPTION:
    case ORT_EP_FAIL:
      code = protobufutil::error::Code::INTERNAL;
      break;
    default:
      code = protobufutil::error::Code::UNKNOWN;
  }

  std::ostringstream oss;
  oss << "ONNX Runtime Status Code: " << onnx_status << ". " << message;
  return protobufutil::Status(code, oss.str());
}

}  // namespace server
}  // namespace onnxruntime
