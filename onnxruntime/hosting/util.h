// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_UTIL_H
#define ONNXRUNTIME_HOSTING_UTIL_H

#include <istream>
#include <string>
#include <google/protobuf/util/json_util.h>

#include "predict.pb.h"

namespace onnx {
namespace hosting {
namespace protobufutil = google::protobuf::util;

class Util {
 public:
  static protobufutil::Status GetRequestFromJson(std::string json_string, /* out */ onnx::hosting::PredictRequest& request);
  static protobufutil::Status GenerateResponseInJson(onnx::hosting::PredictResponse response, /* out */ std::string& json_string);
};
}
}

#endif  // ONNXRUNTIME_HOSTING_UTIL_H