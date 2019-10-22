// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <google/protobuf/util/json_util.h>
#include <boost/beast/http.hpp>

#include "predict.pb.h"

namespace onnxruntime {
namespace server {

namespace http = boost::beast::http;

// Deserialize Json input to PredictRequest.
// Unknown fields in the json file will be ignored.
google::protobuf::util::Status GetRequestFromJson(const std::string& json_string, /* out */ onnxruntime::server::PredictRequest& request);

// Serialize PredictResponse to json string
// 1. Proto3 primitive fields with default values will be omitted in JSON output. Eg. int32 field with value 0 will be omitted
// 2. Enums will be printed as string, not int, to improve readability
google::protobuf::util::Status GenerateResponseInJson(const onnxruntime::server::PredictResponse& response, /* out */ std::string& json_string);

// Constructs JSON error message from error code object and error message
std::string CreateJsonError(http::status error_code, const std::string& error_message);

// Escapes a string following the JSON standard
// Mostly taken from here: https://stackoverflow.com/questions/7724448/simple-json-string-escape-for-c/33799784#33799784
std::string escape_string(const std::string& message);

}  // namespace server
}  // namespace onnxruntime
