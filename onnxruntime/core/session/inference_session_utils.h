// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/inference_session.h"
#include "core/framework/session_options.h"
#include "core/common/common.h"
#include "single_include/nlohmann/json.hpp"

using json = nlohmann::json;

namespace onnxruntime {

namespace inference_session_utils {

static const std::string ort_config_key = "ort_config";
static const std::string session_options_key = "session_options";

}  // namespace inference_session_utils

class InferenceSessionUtils {
 public:
  InferenceSessionUtils(const logging::Logger& logger) {
    logger_ = &logger;
  }

  Status ParseOrtConfigJsonInModelProto(const ONNX_NAMESPACE::ModelProto& model_proto);

  Status ParseSessionOptionsFromModelProto(/*out*/ SessionOptions& session_options);

  Status ParseRunOptionsFromModelProto(/*out*/ RunOptions& run_options);

 private:
  const logging::Logger* logger_;
  nlohmann::json parsed_json_;
  bool is_json_parsed_ = false;
};

}  // namespace onnxruntime
