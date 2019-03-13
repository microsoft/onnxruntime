// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/util/json_util.h>
#include "util.h"

namespace protobufutil = google::protobuf::util;

namespace onnx {
namespace hosting {
protobufutil::Status GetRequestFromJson(std::string json_string, /* out */ onnx::hosting::PredictRequest& request) {
  protobufutil::JsonParseOptions options;
  protobufutil::Status result = JsonStringToMessage(json_string, &request, options);

  return result;
}

protobufutil::Status GetRequestFromBinary(std::istream* input_stream, /* out */ onnx::hosting::PredictRequest& request) {
  bool succeeded = request.ParseFromIstream(input_stream);

  if (succeeded) {
    return protobufutil::Status(protobufutil::error::Code::OK, "Parsing istream succeeded.");
  } else {
    std::string error_message = request.InitializationErrorString();  // TODO: log the error
    return protobufutil::Status(protobufutil::error::Code::INVALID_ARGUMENT, error_message.c_str());
  }
}

protobufutil::Status GenerateResponseInJson(onnx::hosting::PredictResponse response, /* out */ std::string json_string) {
  return protobufutil::Status(protobufutil::error::Code::OK, "Parsing istream succeeded.");
}

protobufutil::Status GenerateResponseInStream(onnx::hosting::PredictResponse response, /* out */ std::ostream* output_stream) {
  return protobufutil::Status(protobufutil::error::Code::OK, "Parsing istream succeeded.");
}
}
}