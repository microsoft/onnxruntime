// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "custom_data_propagation.h"
#include "core/graph/graph.h"

namespace onnxruntime {

/**
 * @brief Class to infer the output values/scalar for 'Squeeze' operator given the input is shape values.
 *
 */
class SqueezeOpDataPropagation : public CustomDataPropagationBase {
 public:
  SqueezeOpDataPropagation(const Node& node,
                           NodeArg& output_def,
                           std::function<Status(const std::string&, TensorShapeVector&)> func,
                           const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                           const logging::Logger& logger) noexcept
      : CustomDataPropagationBase(node, output_def, func, output_from_onnx_op_data_propagation, logger) {}

  Status infer() override;
};

}  // namespace onnxruntime
