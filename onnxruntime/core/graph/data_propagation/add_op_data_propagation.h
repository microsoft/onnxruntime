// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "data_propagation.h"
#include "core/graph/graph.h"
namespace onnxruntime {

class AddOpDataPropagation : public CustomDataPropagation {
 public:
  AddOpDataPropagation(const Node& node,
                       NodeArg& output_def,
                       std::function<Status(const std::string&, TensorShapeVector&)> func,
                       const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation) noexcept
      : CustomDataPropagation(node, output_def, func, output_from_onnx_op_data_propagation) {}

  Status infer() override;
};

}  // namespace onnxruntime
