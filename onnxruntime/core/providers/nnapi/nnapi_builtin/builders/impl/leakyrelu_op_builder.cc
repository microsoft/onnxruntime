// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class LeakyReluOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // LeakyRelu opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }
};

// Add operator related

Status LeakyReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input_shape = shaper[input];

  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 0.01f);

  // TODO: We will use the NNAPI Select to simulate the behavior here
  // input x = [-1, 0, 1]
  // iterate and multiply by the alpha value from attribute: z = alpha * x = [-0.1, 0, 0.1]
  // then construct the mask  c = [false, true, true] , true means select the element from the first input, and false vice versa.
  // output y = [-0.1, 0, 1]
  // the inputs for Select operation will be prepared.

  InlinedVector<uint32_t> input_indices;

  // Construct the boolean type mask array for ANEURALNETWORKS_SELECT - input0
  std::vector<bool> mask(input_shape.size());
  const auto& input_tensor = *initializers.at(input);

  // Note: by default NNAPI only supports float type input, so uses a float type gsl::span here
  // See BaseOpBuilder::HasSupportedInputOutputsImpl.
  Initializer unpacked_tensor(input_tensor);
  auto raw_input_data = unpacked_tensor.DataAsSpan<float>();
  std::transform(raw_input_data.begin(), raw_input_data.end(), mask.begin(),
                 [](auto value) { return value >= 0; });

  // Iterate and multiply by alpha to construct ANEURALNETWORKS_SELECT - input2
  std::vector<float> input2;
  input2.reserve(raw_input_data.size());
  std::transform(raw_input_data.begin(), raw_input_data.end(), std::back_inserter(input2),
                 [alpha](float value) { return alpha * value; });

  const auto mask_tensor_name = model_builder.GetUniqueName(node_unit.Name() + input + "_select_c");
  const auto input2_tensor_name = model_builder.GetUniqueName(node_unit.Name() + input + "_select_y");

  shaper.AddShape(mask_tensor_name, input_shape);
  shaper.AddShape(input2_tensor_name, input_shape);

  Shape input_dimen = {static_cast<uint32_t>(input_shape.size())};

  const OperandType mask_operand_type(operand_types.at(input).type, input_dimen);
  std::vector<char> mask_char(mask.begin(), mask.end());  // Convert to vector of char
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(mask_tensor_name, mask_char.data(), mask_operand_type));
  const OperandType input2_operand_type(operand_types.at(input).type, input_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(input2_tensor_name, input2.data(), input2_operand_type));

  input_indices.push_back(operand_indices.at(mask_tensor_name));    // input0
  input_indices.push_back(operand_indices.at(input));               // input1
  input_indices.push_back(operand_indices.at(input2_tensor_name));  // input2

  const OperandType output_operand_type(operand_types.at(input).type, input_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SELECT, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

// Operator support related

bool LeakyReluOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                           const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  // Note: We use NNAPI ANEURALNETWORKS_SELECT operation to simulate LeakyRelu op, ANEURALNETWORKS_SELECT has supported
  // tensor rank from 1:
  // https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a49b2dc37ea9219789a6d82f281499dbb
  if (input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "NNAPI LeakyRelu supports tensor rank starting from 1. Empty shape is not supported.";
    return false;
  }

  return true;
}

void CreateLeakyReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LeakyReluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
