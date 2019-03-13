// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/util/json_util.h>
#include "util.h"

namespace protobufutil = google::protobuf::util;

namespace onnx {
namespace hosting {
protobufutil::Status Util::GetRequestFromJson(std::string json_string, /* out */ onnx::hosting::PredictRequest& request) {
  protobufutil::JsonParseOptions options;
  options.ignore_unknown_fields = true;

  protobufutil::Status result = JsonStringToMessage(json_string, &request, options);
  return result;
}

protobufutil::Status Util::GenerateResponseInJson(onnx::hosting::PredictResponse response, /* out */ std::string& json_string) {
  protobufutil::JsonPrintOptions options;
  options.add_whitespace = false;
  options.always_print_primitive_fields = false;
  options.preserve_proto_field_names = false;

  protobufutil::Status result = MessageToJsonString(response, &json_string, options);
  return result;
}
}
}