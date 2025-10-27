// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/graph/graph.h"
#include "data_propagation.h"

namespace onnxruntime {
/**
 * @brief Create custom data propagation for the operator.
 *
 * For certain operators (e.g., Size, Squeeze, Unsqueeze), ONNX's
 * PartialDataPropagationFunction() does not always produce complete or accurate
 * inferred shape values.
 *
 * In particular:
 *  - Scalar inputs and outputs are not handled correctly.
 *  - Some operators require additional logic that is not covered by the default function.
 *
 * Therefore, for these cases, we perform custom data propagation to ensure
 * correct and complete inference.
 *
 * @param node The ORT's node
 * @param output_def The node's output NodeArg to save the inferred shape values if needed
 * @param func Helper function to get the input value if it's a initializer
 * @param output_from_onnx_op_data_propagation The result from executing ONNX operator's data propagation
 * @param logger The reference to a logger
 * @return std::unique_ptr<CustomDataPropagation> Returns a CustomDataPropagation object if available
 */
std::unique_ptr<CustomDataPropagation> CreateCustomDataPropagation(const Node& node,
                                                                   NodeArg& output_def,
                                                                   std::function<Status(const std::string&, TensorShapeVector&)> funcs,
                                                                   const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                                                                   const logging::Logger& logger);

}  // namespace onnxruntime
