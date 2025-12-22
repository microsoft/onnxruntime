// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"
#include <onnx/onnx-ml.pb.h>

namespace onnxruntime {

/**
 * @class CustomDataPropagation
 * Custom data propagation for the operator to help enhance shape inference.
 *
 * Calling infer() can infer the output values for the specific operator given the input is shape values
 * and saves the output values in output node_arg for other operators to use later.
 * The purpose of this class is to make shape values being correctly inferred and propogated through the graph.
 */
class CustomDataPropagationBase {
 public:
  ORT_DISALLOW_COPY(CustomDataPropagationBase);
  virtual ~CustomDataPropagationBase() = default;
  virtual Status infer() = 0;

 protected:
  CustomDataPropagationBase(const Node& node,
                            NodeArg& output_def,
                            std::function<Status(const std::string&, TensorShapeVector&)> func,
                            const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                            const logging::Logger& logger) noexcept
      : node_(node),
        output_def_(output_def),
        get_initialized_input_values_func_(std::move(func)),
        output_from_onnx_op_data_propagation_(output_from_onnx_op_data_propagation),
        logger_(logger) {}

  const Node& node_;
  NodeArg& output_def_;
  std::function<Status(const std::string&, TensorShapeVector&)> get_initialized_input_values_func_;
  const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation_;
  const logging::Logger& logger_;
};

/**
 * @brief Create custom data propagation for the operator.
 *
 * For certain operators (e.g., Size, Squeeze, Unsqueeze), ONNX's
 * PartialDataPropagationFunction() does not always produce complete or accurate
 * inferred shape values.
 *
 * In particular:
 *  - Scalar inputs and outputs are not handled correctly.
 *  - Some operators require additional logic that is not covered by the default function,
      e.g. PartialDataPropagationFunction.
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
std::unique_ptr<CustomDataPropagationBase> CreateCustomDataPropagation(
    const Node& node,
    NodeArg& output_def,
    std::function<Status(const std::string&, TensorShapeVector&)> func,
    const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
    const logging::Logger& logger);

}  // namespace onnxruntime
