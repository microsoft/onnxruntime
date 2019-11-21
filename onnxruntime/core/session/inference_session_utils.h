// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

namespace onnxruntime {

namespace inference_session_utils {

Status parse_session_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto,
                                              /*out*/ SessionOptions& session_options,
                                              const logging::Logger& logger);

Status parse_run_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto,
                                          /*out*/ RunOptions& run_options,
                                          const logging::Logger& logger);

}  // namespace inference_session_utils
}  // namespace onnxruntime
