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

class PadOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;  // for ANEURALNETWORKS_PAD_V2
  }

  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override {
    // before Pad-11, inputs `pads` and `constant_value` were attributes
    // only support inputs now
    // Note: Could add support for attributes later.
    return 11;
  }

  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // pads
  if (inputs.size() > 2) {
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // constant_value
  }
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  InlinedVector<uint32_t> input_indices{};

  // `data` input
  const auto& data = inputs[0].node_arg.Name();
  input_indices.push_back(operand_indices.at(data));

  // `pads` input
  // convert from [begin_1, begin_2, ..., end_1, end_2, ...] to [begin_1, end_1, begin_2, end_2, ...]
  // convert from int64_t to int32_t
  const auto data_shape = shaper[data];
  const uint32_t data_rank = SafeInt<uint32_t>(data_shape.size());

  const auto& pads = inputs[1].node_arg.Name();
  const auto* pads_initializer = model_builder.GetConstantInitializer(pads);
  ORT_RETURN_IF_NOT(pads_initializer, "pads must be a constant");

  Initializer pads_initializer_raw_data(*pads_initializer);
  // assume pads_initializer has int64 data, per ONNX spec
  std::vector<int32_t> converted_pads_data{};
  converted_pads_data.reserve(2 * data_rank);
  auto pads_span = pads_initializer_raw_data.DataAsSpan<int64_t>();
  for (size_t i = 0; i < data_rank; ++i) {
    converted_pads_data.push_back(SafeInt<int32_t>(pads_span[i]));
    converted_pads_data.push_back(SafeInt<int32_t>(pads_span[i + data_rank]));
  }

  const Shape converted_pads_shape{data_rank, 2};
  const OperandType converted_pads_operand_type{Type::TENSOR_INT32, converted_pads_shape};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(pads, converted_pads_data.data(),
                                                                      converted_pads_operand_type));
  input_indices.push_back(operand_indices.at(pads));

  // `constant_value` input
  float pad_value = 0.0f;
  if (inputs.size() > 2 && inputs[2].node_arg.Exists()) {
    const auto& constant_value = inputs[2].node_arg.Name();
    const auto* constant_value_initializer = model_builder.GetConstantInitializer(constant_value);
    ORT_RETURN_IF_NOT(constant_value_initializer, "constant_value must be a constant");
    Initializer pad_value_raw_data_init(*constant_value_initializer);
    pad_value = pad_value_raw_data_init.DataAsSpan<float>()[0];
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, pad_value);

  const auto& output = outputs[0].node_arg.Name();
  const OperandType output_operand_type{operand_types.at(data).type, shaper[output]};
  const auto op_code = ANEURALNETWORKS_PAD_V2;

  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

bool PadOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                     const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();

  // only support 1-4d input shape
  // only support input with more than 0 elements
  {
    Shape input_shape;
    if (!GetShape(inputs[0].node_arg, input_shape)) {
      return false;
    }

    if (input_shape.size() > 4 || input_shape.empty()) {
      LOGS_DEFAULT(VERBOSE) << "Pad only supports up to 1-4d shape, input is "
                            << input_shape.size() << "d shape";
      return false;
    }

    if (std::find(input_shape.begin(), input_shape.end(), uint32_t{0}) != input_shape.end()) {
      LOGS_DEFAULT(VERBOSE) << "Pad input with zero elements is not supported";
      return false;
    }
  }

  // only support "constant" mode
  // Note: Could possibly add support for "reflect" later using ANEURALNETWORKS_MIRROR_PAD.
  {
    NodeAttrHelper helper{node_unit};
    const auto mode = helper.Get("mode", "constant");
    if (mode != "constant") {
      LOGS_DEFAULT(VERBOSE) << "Mode is not supported: " << mode;
      return false;
    }
  }

  // only support if `pads` input is known and does not contain negative values
  {
    const auto* pads_initializer = graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name());
    if (!pads_initializer) {
      LOGS_DEFAULT(VERBOSE) << "pads must be a constant initializer";
      return false;
    }

    Initializer unpacked_tensor(*pads_initializer);
    auto tensor_data = unpacked_tensor.DataAsSpan<int64_t>();
    for (size_t i = 0; i < unpacked_tensor.size(); i++) {
      if (tensor_data[i] < 0) {
        LOGS_DEFAULT(VERBOSE) << "Negative pad value is not supported: pads["
                              << i << "] = " << tensor_data[i];
        return false;
      }
    }
  }

  // only support if `constant_value` input is known
  // Note: Could add support for non-constant initializer later. Then we need to ensure it is a scalar (with shape []).
  if (inputs.size() > 2) {
    if (!graph_viewer.GetConstantInitializer(inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "constant_value must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
