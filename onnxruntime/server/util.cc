// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>
#include <google/protobuf/stubs/status.h>

#include "core/common/status.h"
#include "util.h"

// boost random is using a deprecated header in 1.69
// See: https://github.com/boostorg/random/issues/49
#define BOOST_PENDING_INTEGER_LOG2_HPP
#include <boost/integer/integer_log2.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

protobufutil::Status GenerateProtobufStatus(const onnxruntime::common::Status& onnx_status, const std::string& message) {
  protobufutil::error::Code code = protobufutil::error::Code::UNKNOWN;
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

namespace internal {
std::string InternalRequestId() {
  return boost::uuids::to_string(boost::uuids::random_generator()());
}
}  // namespace internal

}  // namespace server
}  // namespace onnxruntime
