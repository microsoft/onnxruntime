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

class LRNOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t input_data_type;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_data_type, logger), "Cannot get input type");
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  const auto node_name = node.Name();
  emscripten::val wnn_builder = model_builder.GetBuilder();

  NodeAttrHelper helper(node);
  const float alpha = helper.Get("alpha", 0.0001f);
  const float beta = helper.Get("beta", 0.75f);
  const float bias = helper.Get("bias", 1.0f);
  const uint32_t size = helper.Get("size", 1);

  // Prepare WebNN constants for alpha, beta, bias attributes.
  // Assume T is float, because input_data_type has been limited to float32 and float16 in 'hasSupportedInitsImpl'.
  emscripten::val alpha_constant = model_builder.CreateOrGetConstant<float>(input_data_type, alpha);
  emscripten::val beta_constant = model_builder.CreateOrGetConstant<float>(input_data_type, beta);
  emscripten::val bias_constant = model_builder.CreateOrGetConstant<float>(input_data_type, bias);
  emscripten::val pow1_constant = model_builder.CreateOrGetConstant<float>(input_data_type, 2);

  /**
      WebNN doesn't support LRN. So decompose it into a series of ops:
      X --> Pow --> (Transpose)--> Pad --> AveragePool--> (Transpose) --> Mul --> Add --> Pow --> Div
             ^           ^                      ^               ^          ^       ^       ^       ^
             |           |                      |               |          |       |       |       |
            Y:2      (0,2,3,1)           Kernel:(1,size)     (0,3,1,2)   B:alpha  B:bias B:beta  A:input
      */
  //
  // pow(input, 2)
  emscripten::val label_options = emscripten::val::object();
  label_options.set("label", node_name + "_pow1");
  emscripten::val pow1_output = wnn_builder.call<emscripten::val>("pow", input, pow1_constant, label_options);

  // transpose(pow1_output, permutation=[0, 2, 3, 1])
  // LRN is one of NHWC layout sensitive ops. When preferred layout is NCHW, move dimension 1 to dimension 3 (rightmost).
  if (model_builder.GetPreferredLayout() == DataLayout::NCHW) {
    std::vector<uint32_t> perm{0, 2, 3, 1};
    emscripten::val transpose_options = emscripten::val::object();
    transpose_options.set("label", node_name + "_transpose_rightmost");
    transpose_options.set("permutation", emscripten::val::array(perm));
    pow1_output =
        wnn_builder.call<emscripten::val>("transpose", pow1_output, transpose_options);
  }

  // pad(pow1_output, beginning_padding = {0, 0, 0, leading_padding}, ending_padding = {0, 0, 0, trailing_padding})
  // Adding a Pad before averagePool2d and calling AveragePool with pads as 0's.
  const uint32_t leading_padding = floor((size - 1) / 2);
  const uint32_t trailing_padding = ceil((size - 1) / 2);
  std::vector<uint32_t> beginning_padding{0, 0, 0, leading_padding};
  std::vector<uint32_t> ending_padding{0, 0, 0, trailing_padding};
  emscripten::val pad_options = emscripten::val::object();
  pad_options.set("label", node_name + "_pad");
  emscripten::val pad_output =
      wnn_builder.call<emscripten::val>("pad", pow1_output, emscripten::val::array(beginning_padding),
                                        emscripten::val::array(ending_padding), pad_options);

  // averagePool2d(pad_output, pool_options)
  const std::vector<uint32_t> kernel_shape = {1, size};
  emscripten::val pool_options = emscripten::val::object();
  pool_options.set("label", node_name + "_averagePool2d");
  pool_options.set("windowDimensions", emscripten::val::array(kernel_shape));
  emscripten::val pool_output = wnn_builder.call<emscripten::val>("averagePool2d", pad_output, pool_options);

  // transpose(pool_output, permutation=[0, 3, 1, 2])
  // Move dimension 3 back to dimension 1.
  if (model_builder.GetPreferredLayout() == DataLayout::NCHW) {
    std::vector<uint32_t> perm{0, 3, 1, 2};
    emscripten::val transpose_options = emscripten::val::object();
    transpose_options.set("label", node_name + "_transpose_inverse");
    transpose_options.set("permutation", emscripten::val::array(perm));
    pool_output =
        wnn_builder.call<emscripten::val>("transpose", pool_output, transpose_options);
  }

  // mul(pool_output, alpha_constant)
  label_options.set("label", node_name + "_mul");
  emscripten::val mul_output =
      wnn_builder.call<emscripten::val>("mul", pool_output, alpha_constant, label_options);

  // add(mul_output, bias_constant)
  label_options.set("label", node_name + "_add");
  emscripten::val add_output = wnn_builder.call<emscripten::val>("add", mul_output, bias_constant, label_options);

  // pow(add_output, beta_constant)
  label_options.set("label", node_name + "_pow2");
  emscripten::val pow2_output = wnn_builder.call<emscripten::val>("pow", add_output, beta_constant, label_options);

  // div(input, pow2_output)
  label_options.set("label", node_name + "_div");
  emscripten::val div_output = wnn_builder.call<emscripten::val>("div", input, pow2_output, label_options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(div_output));
  return Status::OK();
}

// Operator support related.
bool LRNOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                     const Node& node,
                                     const WebnnDeviceType /* device_type */,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "LRN only supports 4D input shape, input is "
                          << input_size << "D shape";
    return false;
  }

  return true;
}

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LRNOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
