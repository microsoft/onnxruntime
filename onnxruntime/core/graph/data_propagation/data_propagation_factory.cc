// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "data_propagation_factory.h"
#include "core/common/logging/logging.h"
#include "size_op_data_propagation.h"
#include "squeeze_op_data_propagation.h"
#include "unsqueeze_op_data_propagation.h"
#include "gather_op_data_propagation.h"
#include "add_op_data_propagation.h"
#include "sub_op_data_propagation.h"
#include "mul_op_data_propagation.h"
#include "div_op_data_propagation.h"
#include <onnx/onnx-ml.pb.h>

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
                                                                   std::function<Status(const std::string&, TensorShapeVector&)> func,
                                                                   const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                                                                   const logging::Logger& logger) {
  auto dim_size = output_from_onnx_op_data_propagation.tensor_type().shape().dim_size();

  if (node.OpType() == "Size") {
    return std::make_unique<SizeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
  } else if (node.OpType() == "Squeeze") {
    return std::make_unique<SqueezeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
  } else if (node.OpType() == "Unsqueeze") {
    return std::make_unique<UnsqueezeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
  } else if (dim_size == 0) {
    if (node.OpType() == "Gather") {
      return std::make_unique<GatherOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Add") {
      return std::make_unique<AddOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Sub") {
      return std::make_unique<SubOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Mul") {
      return std::make_unique<MulOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Div") {
      return std::make_unique<DivOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation, logger);
    }
  }
  return nullptr;
}

}  // namespace onnxruntime
