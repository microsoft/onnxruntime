// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/graph.h"
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

std::unique_ptr<CustomDataPropagationBase> CreateCustomDataPropagation(const Node& node,
                                                                       NodeArg& output_def,
                                                                       std::function<Status(const std::string&, TensorShapeVector&)> func,
                                                                       const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                                                                       const logging::Logger& logger) {
  int dim_size = 0;
  if (output_from_onnx_op_data_propagation.has_tensor_type() &&
      output_from_onnx_op_data_propagation.tensor_type().has_shape()) {
    dim_size = output_from_onnx_op_data_propagation.tensor_type().shape().dim_size();
  }

  if (node.OpType() == "Size") {
    return std::make_unique<SizeOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
  } else if (node.OpType() == "Squeeze") {
    return std::make_unique<SqueezeOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
  } else if (node.OpType() == "Unsqueeze") {
    return std::make_unique<UnsqueezeOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
  } else if (dim_size == 0) {
    if (node.OpType() == "Gather") {
      return std::make_unique<GatherOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Add") {
      return std::make_unique<AddOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Sub") {
      return std::make_unique<SubOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Mul") {
      return std::make_unique<MulOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
    } else if (node.OpType() == "Div") {
      return std::make_unique<DivOpDataPropagation>(node, output_def, std::move(func), output_from_onnx_op_data_propagation, logger);
    }
  }
  return nullptr;
}

}  // namespace onnxruntime