// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SoftmaxOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_size = input_shape.size();

  emscripten::val options = emscripten::val::object();

  NodeAttrHelper helper(node);
  const auto since_version = node.SinceVersion();
  const int32_t default_axis = since_version < 13 ? 1 : -1;
  int32_t axis = helper.Get("axis", default_axis);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, input_size));

  // Prior to opset 13, Softmax operates with different semantics compared to opset 13 and later.
  // Specifically, it normalizes over the flattened range of dimensions starting from the specified
  // axis to the last dimension.
  // In contrast, WebNN's softmax aligns with the behavior introduced in opset 13 and later.
  // To handle the differences for earlier opsets, a reshape operation can be applied if necessary.
  const bool do_reshape = since_version < 13 && axis != SafeInt<int32_t>(input_size - 1);
  std::vector<uint32_t> input_shape_uint32;
  if (do_reshape) {
    input_shape_uint32 = GetNarrowedIntFromInt64<uint32_t>(input_shape);
    // Need to reshape the input to 2D tensor with new shape [M, N].
    // M = d0*d1*...*d(axis-1), N = d(axis)*...*d(n-1)
    const auto M = Product(std::vector<uint32_t>(input_shape_uint32.begin(), input_shape_uint32.begin() + axis));
    const auto N = Product(std::vector<uint32_t>(input_shape_uint32.begin() + axis, input_shape_uint32.end()));
    emscripten::val new_shape = emscripten::val::array();
    new_shape.set(0, M);
    new_shape.set(1, N);

    options.set("label", node.Name() + "_reshape_input");
    input = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape, options);
    // Apply softmax along the last dimension (N).
    axis = 1;
  }

  options.set("label", node.Name());
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("softmax", input, axis, options);

  if (do_reshape) {
    // Softmax has the same output shape as input shape.
    // Reshape the output back to the original input shape.
    options.set("label", node.Name() + "_reshape_output");
    output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", output, emscripten::val::array(input_shape_uint32), options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
