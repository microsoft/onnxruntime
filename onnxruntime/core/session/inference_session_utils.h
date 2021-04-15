// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/flatbuffers/schema/ort.fbs.h"

#if !defined(ORT_MINIMAL_BUILD)
//
// Includes to parse json session config from onnx model file
//
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"
#include "core/common/common.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 28020)
#endif
#include "nlohmann/json.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

using json = nlohmann::json;
#endif

namespace onnxruntime {

namespace inference_session_utils {

// need this value to be accessible in all builds in order to report error for attempted usage in a minimal build
static constexpr const char* kOrtLoadConfigFromModelEnvVar = "ORT_LOAD_CONFIG_FROM_MODEL";

#if !defined(ORT_MINIMAL_BUILD)
//
// Code to parse json session config from onnx model file
//
static constexpr const char* kOrtConfigKey = "ort_config";
static constexpr const char* kSessionOptionsKey = "session_options";

class JsonConfigParser {
 public:
  JsonConfigParser(const logging::Logger& logger) : logger_(logger) {
  }

  Status ParseOrtConfigJsonInModelProto(const ONNX_NAMESPACE::ModelProto& model_proto);

  Status ParseSessionOptionsFromModelProto(/*out*/ SessionOptions& session_options);

  Status ParseRunOptionsFromModelProto(/*out*/ RunOptions& run_options);

 private:
  // Logger instance that will be used to log events along the parsing steps
  const logging::Logger& logger_;

  // Flag indicating if the model has been checked for ort config json existence
  bool is_model_checked_for_ort_config_json_ = false;

  // Parsed json available for other utility methods to use (if the model did have a valid json)
  nlohmann::json parsed_json_;

  // Flag indicating if an ort config json is available to be used
  bool is_ort_config_json_available_ = false;
};

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace inference_session_utils
}  // namespace onnxruntime
