// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "custom_data_propagation.h"
#include "core/graph/graph.h"

namespace onnxruntime {

/**
 * @brief Class to infer the output scalar for 'Div' operator given the input is a scalar related to shape.
 *
 *
 * For example:
 *
 *  (input with the shape as float32[1, 3, 64, 64])
 *     |
 *     v
 *   Shape            (It saves [1, 3, 64, 64] in inferred_shape_values_ in output's node_arg
 *     |               during graph::SaveShapeValuesFromDataPropagation())
 *     |
 *     | ______
 *     |       |
 *     v       v
 *   Gather  Gather   (First 'Gather' saves 64 in inferred_scalar_value_ in output node_arg, and
 *     |       |       second 'Gather' saves 1 in inferred_scalar_value_ in output node_arg
 *     |       |       during GatherOpDataPropagation(), if the 'index' attributes
 *     |       |       are 2 and 0 respectively)
 *      \     /
 *       \   /
 *        | |
 *        v v
 *        Div        (It gets 64 from inferred_scalar_value_ in input A's node_arg and 1 from inferred_scalar_value_
 *         |          in input B's node_arg, then performs div operation to get 64 and saves in inferred_scalar_value_
 *         |          in output's node_arg)
 *         v
 *         ...
 */
class DivOpDataPropagation : public CustomDataPropagationBase {
 public:
  DivOpDataPropagation(const Node& node,
                       NodeArg& output_def,
                       std::function<Status(const std::string&, TensorShapeVector&)> func,
                       const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                       const logging::Logger& logger) noexcept
      : CustomDataPropagationBase(node, output_def, func, output_from_onnx_op_data_propagation, logger) {}

  Status infer() override;
};

}  // namespace onnxruntime
