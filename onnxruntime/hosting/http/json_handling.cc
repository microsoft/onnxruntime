// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <istream>
#include <string>
#include <boost/beast/core.hpp>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/json_util.h>

#include "predict.pb.h"
#include "json_handling.h"

namespace protobufutil = google::protobuf::util;

namespace onnxruntime {
namespace hosting {

protobufutil::Status GetRequestFromJson(std::string json_string, /* out */ onnxruntime::hosting::PredictRequest& request) {
  protobufutil::JsonParseOptions options;
  options.ignore_unknown_fields = true;

  protobufutil::Status result = JsonStringToMessage(json_string, &request, options);
  return result;
}

protobufutil::Status GenerateResponseInJson(onnxruntime::hosting::PredictResponse response, /* out */ std::string& json_string) {
  protobufutil::JsonPrintOptions options;
  options.add_whitespace = false;
  options.always_print_primitive_fields = false;
  options.always_print_enums_as_ints = false;
  options.preserve_proto_field_names = false;

  protobufutil::Status result = MessageToJsonString(response, &json_string, options);
  return result;
}
}  // namespace hosting
}  // namespace onnxruntime