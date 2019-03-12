// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <istream>
#include <string>

#include "predict.pb.h"

google::protobuf::util::Status GetRequestFromJson(std::string json_string, /* out */ onnx::hosting::PredictRequest& request);
google::protobuf::util::Status GetRequestFromBinary(std::istream* input_stream, /* out */ onnx::hosting::PredictRequest& request);
google::protobuf::util::Status GenerateResponseInJson(onnx::hosting::PredictResponse response, /* out */ std::string json_string);
google::protobuf::util::Status GenerateResponseInStream(onnx::hosting::PredictResponse response, /* out */ std::ostream* output_stream);