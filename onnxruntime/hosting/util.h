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

protobufutil::Status GetRequestFromJson(std::string json_string, /* out */ onnx::hosting::PredictRequest& request);
protobufutil::Status GetRequestFromBinary(std::istream* input_stream, /* out */ onnx::hosting::PredictRequest& request);
protobufutil::Status GenerateResponseInJson(onnx::hosting::PredictResponse response, /* out */ std::string json_string);
protobufutil::Status GenerateResponseInStream(onnx::hosting::PredictResponse response, /* out */ std::ostream* output_stream);
}
}

#endif  // ONNXRUNTIME_HOSTING_UTIL_H