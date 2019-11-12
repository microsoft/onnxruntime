// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

namespace onnxruntime {

namespace inference_session_utils {

Status parse_session_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto,
                                              SessionOptions& session_options, bool& use_session_options);

Status parse_run_options_from_model_proto(const ONNX_NAMESPACE::ModelProto& model_proto,
                                          RunOptions& session_options, bool& use_run_options);

}  // namespace inference_session_utils
}  // namespace onnxruntime
