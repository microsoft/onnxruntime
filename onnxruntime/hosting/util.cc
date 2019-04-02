// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>
#include "core/common/status.h"
#include "util.h"
#include <google/protobuf/stubs/status.h>

namespace onnxruntime {
namespace hosting {

namespace protobufutil = google::protobuf::util;

//    OK = static_cast<unsigned int>(MLStatus::OK),
//      FAIL = static_cast<unsigned int>(MLStatus::FAIL),
//      INVALID_ARGUMENT = static_cast<unsigned int>(MLStatus::INVALID_ARGUMENT),
//      NO_SUCHFILE = static_cast<unsigned int>(MLStatus::NO_SUCHFILE),
//      NO_MODEL = static_cast<unsigned int>(MLStatus::NO_MODEL),
//      ENGINE_ERROR = static_cast<unsigned int>(MLStatus::ENGINE_ERROR),
//      RUNTIME_EXCEPTION = static_cast<unsigned int>(MLStatus::RUNTIME_EXCEPTION),
//      INVALID_PROTOBUF = static_cast<unsigned int>(MLStatus::INVALID_PROTOBUF),
//      MODEL_LOADED = static_cast<unsigned int>(MLStatus::MODEL_LOADED),
//      NOT_IMPLEMENTED = static_cast<unsigned int>(MLStatus::NOT_IMPLEMENTED),
//      INVALID_GRAPH = static_cast<unsigned int>(MLStatus::INVALID_GRAPH),
//      SHAPE_INFERENCE_NOT_REGISTERED = static_cast<unsigned int>(MLStatus::SHAPE_INFERENCE_NOT_REGISTERED),
//      REQUIREMENT_NOT_REGISTERED = static_cast<unsigned int>(MLStatus::REQUIREMENT_NOT_REGISTERED),

protobufutil::Status GenerateProtoBufStatus(onnxruntime::common::Status onnx_status, const std::string& message) {
  protobufutil::error::Code code;
  switch (onnx_status.Code()) {
    case onnxruntime::common::StatusCode::OK:
    case onnxruntime::common::StatusCode::MODEL_LOADED:
      code = protobufutil::error::Code::OK;
      break;
    case onnxruntime::common::StatusCode::INVALID_ARGUMENT:
    case onnxruntime::common::StatusCode::INVALID_PROTOBUF:
    case onnxruntime::common::StatusCode::INVALID_GRAPH:
    case onnxruntime::common::StatusCode::SHAPE_INFERENCE_NOT_REGISTERED:
    case onnxruntime::common::StatusCode::REQUIREMENT_NOT_REGISTERED:
    case onnxruntime::common::StatusCode::NO_SUCHFILE:
    case onnxruntime::common::StatusCode::NO_MODEL:
      code = protobufutil::error::Code::INVALID_ARGUMENT;
      break;
    case onnxruntime::common::StatusCode::NOT_IMPLEMENTED:
      code = protobufutil::error::Code::UNIMPLEMENTED;
      break;
    case onnxruntime::common::StatusCode::FAIL:
    case onnxruntime::common::StatusCode::RUNTIME_EXCEPTION:
      code = protobufutil::error::Code::INTERNAL;
      break;
    default:
      code = protobufutil::error::Code::UNKNOWN;
  }

  std::ostringstream oss;
  oss << "ONNX Runtime Status Code: " << onnx_status.Code() << ". " << message;
  return protobufutil::Status(code, oss.str());
}

}  // namespace hosting
}  // namespace onnxruntime
