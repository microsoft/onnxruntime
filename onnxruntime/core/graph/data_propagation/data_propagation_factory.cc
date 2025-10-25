// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "data_propagation_factory.h"
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

// For certain operators (e.g., Size, Squeeze, Unsqueeze), invoking ONNX operator's PartialDataPropagationFunction()
// alone does not yield fully accurate inferred shape values.
// Moreover, ONNX operator's PartialDataPropagationFunction() does not handle scalar inputs or outputs.
// Therefore, for those cases, we run our own data propagation.
std::unique_ptr<OrtDataPropagation> CreateOrtDataPropagation(const Node& node,
                                                       NodeArg& output_def,
                                                       std::function<Status(const std::string&, TensorShapeVector&)> func,
                                                       const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation) {
  auto dim_size = output_from_onnx_op_data_propagation.tensor_type().shape().dim_size();

  if (node.OpType() == "Size") {
    return std::make_unique<SizeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
  } else if (node.OpType() == "Squeeze") {
    return std::make_unique<SqueezeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
  } else if (node.OpType() == "Unsqueeze") {
    return std::make_unique<UnsqueezeOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
  } else if (dim_size == 0) {
    if (node.OpType() == "Gather") {
      return std::make_unique<GatherOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
    } else if (node.OpType() == "Add") {
      return std::make_unique<AddOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
    } else if (node.OpType() == "Sub") {
      return std::make_unique<SubOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
    } else if (node.OpType() == "Mul") {
      return std::make_unique<MulOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
    } else if (node.OpType() == "Div") {
      return std::make_unique<DivOpDataPropagation>(node, output_def, func, output_from_onnx_op_data_propagation);
    }
  } 
  return nullptr;
}

} // namespace onnxruntime
