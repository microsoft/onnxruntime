// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session_utils.h"
#include "single_include/nlohmann/json.hpp"

using json = nlohmann::json;

namespace onnxruntime {

namespace inference_session_utils {

//---------------------
//--- local helpers ---
//---------------------

//--------------------------------------------
//--- general JSON parsing related helpers ---
//--------------------------------------------

static Status parse_json(const std::string& config_string, /*out*/ json& json_obj) {
  try {
    json_obj = json::parse(config_string);
  } catch (const std::exception&) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Json stored in the `ort_config` key is invalid. "
                           "Cannot parse necessary config options to run the model with.");
  }

  return Status::OK();
}

static Status parse_json_and_search_for_key(const std::string& config_string,
                                            const std::string& key,
                                            /*out*/ bool& found,
                                            /*out*/ json& json_obj) {
  // Try to parse the given json
  auto status = parse_json(config_string, json_obj);

  // Json parsing failed
  if (!status.IsOK()) {
    found = false;
    return status;
  }

  // The given json doesn't contain requested key
  if (!json_obj.contains(key)) {
    found = false;
  } else {
    found = true;
  }

  return Status::OK();
}

//---------------------------------------------------
//--- end of general JSON parsing related helpers ---
//---------------------------------------------------

//--------------------------------------------
//--- session options related helpers ---
//--------------------------------------------
// Below are some helpers that will be used to set corresponding session option values

static Status
set_intra_op_num_threads(SessionOptions& session_options,
                         int value,
                         const logging::Logger& logger) {
  if (value < 0) {
    LOGS(logger, ERROR) << "Unsupported value for intra_op_num_threads: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for intra_op_num_threads: ", value);
  }

  LOGS(logger, INFO) << "Setting intra_op_num_threads to " << value;
  session_options.intra_op_num_threads = value;
  return Status::OK();
}

static Status set_inter_op_num_threads(SessionOptions& session_options,
                                       int value,
                                       const logging::Logger& logger) {
  if (value < 0) {
    LOGS(logger, ERROR) << "Unsupported value for inter_op_num_threads: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for inter_op_num_threads: ", value);
  }

  LOGS(logger, INFO) << "Setting inter_op_num_threads to " << value;
  session_options.inter_op_num_threads = value;
  return Status::OK();
}

static Status set_execution_mode(SessionOptions& session_options,
                                 int value,
                                 const logging::Logger& logger) {
  switch (value) {
    case ExecutionMode::ORT_SEQUENTIAL:
      LOGS(logger, INFO) << "Setting execution_mode to Sequential mode";
      session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
      return Status::OK();

    case ExecutionMode::ORT_PARALLEL:
      LOGS(logger, INFO) << "Setting execution_mode to Parallel mode";
      session_options.execution_mode = ExecutionMode::ORT_PARALLEL;
      return Status::OK();

    default:
      LOGS(logger, ERROR) << "Unsupported execution_mode value in ORT config: " << value;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported execution_mode value in ORT config: ", value);
  }
}

static Status set_graph_optimization_level(SessionOptions& session_options,
                                           int value,
                                           const logging::Logger& logger) {
  switch (value) {
    case 0:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_DISABLE_ALL";
      session_options.graph_optimization_level = TransformerLevel::Default;
      return Status::OK();

    case 1:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_BASIC";
      session_options.graph_optimization_level = TransformerLevel::Level1;
      return Status::OK();

    case 2:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_EXTENDED";
      session_options.graph_optimization_level = TransformerLevel::Level2;
      return Status::OK();

    case 3:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_ALL";
      session_options.graph_optimization_level = TransformerLevel::Level3;
      return Status::OK();

    default:
      LOGS(logger, ERROR) << "Unsupported graph_optimization_level value in ORT config: " << value;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported graph_optimization_level value in ORT config: ", value);
  }
}

static Status set_enable_profiling(SessionOptions& session_options,
                                   int value,
                                   const logging::Logger& logger) {
  switch (value) {
    case 0:
      LOGS(logger, INFO) << "Setting enable_profiling to false";
      session_options.enable_profiling = false;
      return Status::OK();

    case 1:
      LOGS(logger, INFO) << "Setting enable_profiling to true";
      session_options.enable_profiling = true;
      return Status::OK();

    default:
      LOGS(logger, ERROR) << "Unsupported value for enable_profiling option: " << value;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for enable_profiling option: ", value);
  }
}

//---------------------------------------------------
//--- end of session options related helpers ---
//---------------------------------------------------

//---------------------
//--- end of local helpers ---
//---------------------

static const std::string session_options_key = "session_options";

Status parse_session_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto,
                                              SessionOptions& session_options,
                                              const logging::Logger& logger) {
  json json_obj;
  bool session_options_found = false;

  for (auto metadata_field : model_proto.metadata_props()) {
    if (metadata_field.has_key() && metadata_field.key() == "ort_config") {
      LOGS(logger, INFO)
          << "Found session/run/environment configuration in the model file to be used while running the model";

      auto status =
          parse_json_and_search_for_key(metadata_field.value(), session_options_key, session_options_found, json_obj);

      if (!status.IsOK()) {
        LOGS(logger, ERROR) << "Could not parse session/run/environment configuration json in the model file";
        return status;
      }
    }
  }

  if (!session_options_found) {
    LOGS(logger, INFO) << "Did not find session options in the model file to be used while running the model";
    return Status::OK();
  }

  auto& session_options_from_model = json_obj.at("session_options");

  // TODO: Support all valid session options
  // Only the following config options from the json will be supported in this version
  // Any other option if part of the json (even if valid session option) will be ignored

  for (auto it = session_options_from_model.cbegin(); it != session_options_from_model.cend(); ++it) {
    auto& key = it.key();
    auto& value = it.value();

    if (key == "intra_op_num_threads ") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "intra_op_num_threads option in the model file must be an integer");
      }

      auto status = set_intra_op_num_threads(session_options, it.value().get<int>(), logger);

      if (!status.IsOK()) {
        return status;
      }

    } else if (key == "inter_op_num_threads") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "inter_op_num_threads option in the model file must be an integer");
      }

      auto status = set_inter_op_num_threads(session_options, it.value().get<int>(), logger);

      if (!status.IsOK()) {
        return status;
      }
    } else if (key == "execution_mode") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "execution_mode option in the model file must be an integer");
      }

      auto status = set_execution_mode(session_options, it.value().get<int>(), logger);

      if (!status.IsOK()) {
        return status;
      }
    } else if (key == "graph_optimization_level") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "graph_optimization_level option in the model file must be an integer");
      }

      auto status = set_graph_optimization_level(session_options, it.value().get<int>(), logger);

      if (!status.IsOK()) {
        return status;
      }
    } else if (key == "enable_profiling") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "enable_profiling option in the model file must be an integer");
      }

      auto status = set_enable_profiling(session_options, it.value().get<int>(), logger);

      if (!status.IsOK()) {
        return status;
      }
    } else {
      LOGS(logger, INFO) << "Ignoring unsupported session option in ORT config: " << key;
    }
  }

  return Status::OK();
}

Status parse_run_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& /*model_proto*/,
                                          RunOptions& /*run_options*/,
                                          const logging::Logger& /*logger*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Parsing RunOptions from ModelProto is not supported yet");
}

}  // namespace inference_session_utils
}  // namespace onnxruntime
