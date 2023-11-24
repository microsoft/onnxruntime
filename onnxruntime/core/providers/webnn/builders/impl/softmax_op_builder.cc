// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
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

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val output = emscripten::val::object();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_size = input_shape.size();
  NodeAttrHelper helper(node);
  if (node.SinceVersion() < 13) {
    int32_t axis = helper.Get("axis", 1);
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_size));
    //  Coerce the input into a 2-dimensional tensor with dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}].
    if (input_size != 2) {
      int32_t first_dim = static_cast<int32_t>(std::reduce(input_shape.begin(), input_shape.begin() + axis,
                                                           1, std::multiplies<int64_t>()));
      int32_t second_dim = static_cast<int32_t>(std::reduce(input_shape.begin() + axis, input_shape.end(),
                                                            1, std::multiplies<int64_t>()));
      emscripten::val new_shape = emscripten::val::array(std::vector<int32_t>{first_dim, second_dim});
      input = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape);
    }

    output = model_builder.GetBuilder().call<emscripten::val>("softmax", input);

    // Reshape output to the same shape of input.
    if (input_size != 2) {
      emscripten::val new_shape = emscripten::val::array();
      for (size_t i = 0; i < input_size; i++) {
        new_shape.call<void>("push", static_cast<int32_t>(input_shape[i]));
      }
      output = model_builder.GetBuilder().call<emscripten::val>("reshape", output, new_shape);
    }
  } else {
    int32_t axis = helper.Get("axis", -1);
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_size));
    // Wraparound for transpose the target axis to the last.
    // WebNN compute the softmax values of the 2-D input tensor along axis 1.
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax-method
    if (axis != static_cast<int>(input_shape.size() - 1)) {
      emscripten::val options = emscripten::val::object();
      std::vector<uint32_t> permutation(input_shape.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::rotate(permutation.begin() + axis, permutation.begin() + axis + 1, permutation.end());
      options.set("permutation", emscripten::val::array(permutation));
      input = model_builder.GetBuilder().call<emscripten::val>("transpose", input, options);
    }
    // Wraparound for reshape input tensor to 2-D.
    if (input_shape.size() != 2) {
      uint32_t first_dim = static_cast<uint32_t>(std::reduce(input_shape.begin(), input_shape.begin() + axis,
                                                             1, std::multiplies<int64_t>()));
      first_dim *= static_cast<uint32_t>(std::reduce(input_shape.begin() + axis + 1, input_shape.end(),
                                                     1, std::multiplies<int64_t>()));
      uint32_t second_dim = static_cast<uint32_t>(input_shape[axis]);
      emscripten::val new_shape = emscripten::val::array(std::vector<uint32_t>{first_dim, second_dim});
      input = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape);
    }

    output = model_builder.GetBuilder().call<emscripten::val>("softmax", input);

    // Restore from 2-D to the original shape.
    if (input_shape.size() != 2) {
      std::vector<uint32_t> new_shape;
      std::transform(input_shape.begin(), input_shape.begin() + axis, std::back_inserter(new_shape),
                     [](int64_t dim) -> uint32_t { return static_cast<uint32_t>(dim); });
      std::transform(input_shape.begin() + axis + 1, input_shape.end(), std::back_inserter(new_shape),
                     [](int64_t dim) -> uint32_t { return static_cast<uint32_t>(dim); });
      new_shape.push_back(static_cast<int32_t>(input_shape[axis]));
      output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                output, emscripten::val::array(new_shape));
    }
    // Restore the corresponding axis back to the initial position from the last position.
    if (axis != static_cast<int>(input_shape.size() - 1)) {
      emscripten::val options = emscripten::val::object();
      std::vector<uint32_t> permutation(input_shape.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::rotate(permutation.rbegin(), permutation.rbegin() + 1, permutation.rend() - axis);
      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", output, options);
    }
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool SoftmaxOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size < 2) {
    LOGS(logger, VERBOSE) << "SoftMax only support input size >= 2d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
