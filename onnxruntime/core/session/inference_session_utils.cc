// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session_utils.h"

namespace onnxruntime {

//---------------------
//--- local helpers ---
//---------------------

//--------------------------------------------
//--- session options related helpers ---
//--------------------------------------------
// Below are some helpers that will be used to set corresponding session option values

static Status SetIntraOpNumThreads(SessionOptions& session_options,
                                   int value,
                                   const logging::Logger& logger) {
  if (value < 0) {
    LOGS(logger, ERROR) << "Unsupported value for intra_op_num_threads: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for intra_op_num_threads: ", value);
  }

  LOGS(logger, INFO) << "Setting intra_op_num_threads to " << value;
  session_options.intra_op_param.thread_pool_size = value;
  return Status::OK();
}

static Status SetInterOpNumThreads(SessionOptions& session_options,
                                   int value,
                                   const logging::Logger& logger) {
  if (value < 0) {
    LOGS(logger, ERROR) << "Unsupported value for inter_op_num_threads: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for inter_op_num_threads: ", value);
  }

  LOGS(logger, INFO) << "Setting inter_op_num_threads to " << value;
  session_options.inter_op_param.thread_pool_size= value;
  return Status::OK();
}

static Status SetExecutionMode(SessionOptions& session_options,
                               int value,
                               const logging::Logger& logger) {
  if (value != 0 && value != 1) {
    LOGS(logger, ERROR) << "Unsupported execution_mode value in ORT config: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported execution_mode value in ORT config: ", value);
  }

  LOGS(logger, INFO) << "Setting execution_mode to " << (value == 0 ? "Sequential mode" : "Parallel mode");
  session_options.execution_mode = (value == 0 ? ExecutionMode::ORT_SEQUENTIAL : ExecutionMode::ORT_PARALLEL);
  return Status::OK();
}

static Status SetGraphOptimizationLevel(SessionOptions& session_options,
                                        int value,
                                        const logging::Logger& logger) {
  switch (value) {
    case ORT_DISABLE_ALL:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_DISABLE_ALL";
      session_options.graph_optimization_level = TransformerLevel::Default;
      return Status::OK();

    case ORT_ENABLE_BASIC:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_BASIC";
      session_options.graph_optimization_level = TransformerLevel::Level1;
      return Status::OK();

    case ORT_ENABLE_EXTENDED:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_EXTENDED";
      session_options.graph_optimization_level = TransformerLevel::Level2;
      return Status::OK();

    case ORT_ENABLE_ALL:
      LOGS(logger, INFO) << "Setting graph_optimization_level to ORT_ENABLE_ALL";
      session_options.graph_optimization_level = TransformerLevel::MaxLevel;
      return Status::OK();

    default:
      std::ostringstream message_stream;
      message_stream << "Unsupported graph_optimization_level value in ORT config: " << value;

      std::string message = message_stream.str();

      LOGS(logger, ERROR) << message;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, message);
  }
}

static Status SetEnableProfiling(SessionOptions& session_options,
                                 int value,
                                 const logging::Logger& logger) {
  if (value != 0 && value != 1) {
    LOGS(logger, ERROR) << "Unsupported value for enable_profiling option: " << value;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported value for enable_profiling option: ", value);
  }

  LOGS(logger, INFO) << "Setting enable_profiling to " << (value == 0 ? "false" : "true");
  session_options.enable_profiling = (value == 0 ? false : true);
  return Status::OK();
}

//---------------------------------------------------
//--- end of session options related helpers ---
//---------------------------------------------------

//---------------------
//--- end of local helpers ---
//---------------------

Status InferenceSessionUtils::ParseOrtConfigJsonInModelProto(const ONNX_NAMESPACE::ModelProto& model_proto) {
  if (is_model_checked_for_ort_config_json_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The Model Proto has already been checked for the ORT config json.");
  }

  for (const auto& metadata_field : model_proto.metadata_props()) {
    if (metadata_field.has_key() && metadata_field.key() == inference_session_utils::kOrtConfigKey) {
      LOGS(logger_, INFO)
          << "Found session/run/environment configuration in the model file to be used while running the model";

      try {
        const auto& val = metadata_field.value();
        LOGS(logger_, INFO) << "ORT config json from the model: " << val;

        parsed_json_ = json::parse(val);
        // set the flag indicating that the model has the ORT config json.
        is_ort_config_json_available_ = true;
      } catch (const std::exception& e) {
        std::ostringstream message_stream;
        message_stream << "Json stored in the `ort_config` key cannot be parsed. Error message: " << e.what();

        std::string message = message_stream.str();

        LOGS(logger_, ERROR) << message;
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, message);
      }

      break;
    }
  }

  // all steps complete, set the flag indicating that the model has been checked for the ORT config json.
  is_model_checked_for_ort_config_json_ = true;
  return Status::OK();
}

Status InferenceSessionUtils::ParseSessionOptionsFromModelProto(SessionOptions& session_options) {
  if (!is_model_checked_for_ort_config_json_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The Model Proto hasn't been checked for the ORT config json.");
  }

  if (!is_ort_config_json_available_ || !parsed_json_.contains(inference_session_utils::kSessionOptionsKey)) {
    LOGS(logger_, INFO) << "Did not find session options in the model file to be used while running the model";
    return Status::OK();
  }

  const auto& session_options_from_model =
      parsed_json_.at(inference_session_utils::kSessionOptionsKey);

  // TODO: Support all valid session options
  // Only the following config options from the json will be supported in this version
  // Any other option if part of the json (even if valid session option) will be ignored

  for (const auto& it : session_options_from_model.items()) {
    const auto& key = it.key();
    const auto& value = it.value();

    if (key == "intra_op_num_threads") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "intra_op_num_threads option in the model file must be an integer");
      }

      ORT_RETURN_IF_ERROR(SetIntraOpNumThreads(session_options, it.value().get<int>(), logger_));

    } else if (key == "inter_op_num_threads") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "inter_op_num_threads option in the model file must be an integer");
      }

      ORT_RETURN_IF_ERROR(SetInterOpNumThreads(session_options, it.value().get<int>(), logger_));

    } else if (key == "execution_mode") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "execution_mode option in the model file must be an integer");
      }

      ORT_RETURN_IF_ERROR(SetExecutionMode(session_options, it.value().get<int>(), logger_));

    } else if (key == "graph_optimization_level") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "graph_optimization_level option in the model file must be an integer");
      }

      ORT_RETURN_IF_ERROR(SetGraphOptimizationLevel(session_options, it.value().get<int>(), logger_));

    } else if (key == "enable_profiling") {
      if (!value.is_number_integer()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "enable_profiling option in the model file must be an integer");
      }

      ORT_RETURN_IF_ERROR(SetEnableProfiling(session_options, it.value().get<int>(), logger_));

    } else {
      LOGS(logger_, INFO) << "Ignoring unsupported session option in ORT config: " << key;
    }
  }

  return Status::OK();
}

Status InferenceSessionUtils::ParseRunOptionsFromModelProto(RunOptions& /*run_options*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Parsing RunOptions from ModelProto is not supported yet");
}

}  // namespace onnxruntime
