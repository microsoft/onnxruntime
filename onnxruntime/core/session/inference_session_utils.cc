// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session_utils.h"
#include "single_include/nlohmann/json.hpp"

namespace onnxruntime {

namespace inference_session_utils {

//---------------------
//--- local helpers ---
//---------------------

//----------------------------
//--- end of local helpers ---
//----------------------------

Status parse_session_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto, SessionOptions& session_options) {
  for (auto metadata_field : model_proto.metadata_props()) {
    if (metadata_field.has_key() && metadata_field.key() == "ort_config") {
      // TODO: Add parsing logic here
    }
  }
  return Status::OK();
}

Status parse_run_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto, RunOptions& run_options) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Parsing RunOptions from ModelProto is not supported yet");
}

}  // namespace inference_session_utils
}  // namespace onnxruntime
