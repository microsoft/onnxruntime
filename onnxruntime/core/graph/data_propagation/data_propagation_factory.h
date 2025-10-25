// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/graph/graph.h"
#include "data_propagation.h"

namespace onnxruntime {

std::unique_ptr<OrtDataPropagation> CreateOrtDataPropagation(const Node& node,
                                                             NodeArg& output_def,
                                                             std::function<Status(const std::string&, TensorShapeVector&)> funcs,
                                                             const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation);

}  // namespace onnxruntime
