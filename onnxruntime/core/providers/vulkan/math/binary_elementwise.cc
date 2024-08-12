
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/binary_elementwise.h"

#include "include/ncnn/layer/binaryop.h"

#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace vulkan {

Status BinaryElementwiseKernel::SetupNcnnParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) {
  params.set(Params::kOperationType, op_type_);

  const auto& node = Node();
  const auto& input_defs = node.InputDefs();

  // Use the NCNN scalar input optimization if `b` is a scalar.
  // TODO: If we wanted to support `a` being a scalar it requires a lot more work as we'd be flipping the inputs around
  // and other things like adding shape hints for inputs would need to be aware of this. Probably easier to change
  // the model instead of adding complexity here for what may be a rare use case.
  const auto& b_arg = *input_defs[1];
  const auto* b = graph_viewer.GetConstantInitializer(b_arg.Name(), true);

  const auto is_scalar_input = [](const NodeArg& input) {
    const auto* shape = input.Shape();
    return (shape && shape->dim_size() == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == 1);
  };

  if (b && is_scalar_input(b_arg)) {
    params.set(Params::kWithScalar, 1);
    Initializer data(*b);

    // we only support float currently. need to handle other data types
    assert(data.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    params.set(Params::kScalarValue, *data.data<float>());
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
