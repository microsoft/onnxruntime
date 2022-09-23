// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
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
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // pads
  if (inputs.size() > 2) {
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // constant_value
  }
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper = model_builder.GetShaper();
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  std::vector<uint32_t> input_indices{};

  // `data` input
  const auto& data = inputs[0].node_arg.Name();
  input_indices.push_back(operand_indices.at(data));

  // `pads` input
  // convert from [begin_1, begin_2, ..., end_1, end_2, ...] to [begin_1, end_1, begin_2, end_2, ...]
  // convert from int64_t to int32_t
  const auto& data_shape = shaper[data];
  const uint32_t data_rank = SafeInt<uint32_t>(data_shape.size());

  const auto& pads = inputs[1].node_arg.Name();
  const auto* pads_initializer = model_builder.GetConstantInitializer(pads);
  ORT_RETURN_IF_NOT(pads_initializer, "pads must be a constant");

  std::vector<uint8_t> pads_initializer_raw_data{};
  ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*pads_initializer, pads_initializer_raw_data));
  // assume pads_initializer has int64 data, per ONNX spec
  ORT_RETURN_IF_NOT(pads_initializer_raw_data.size() == 2 * data_rank * sizeof(int64_t),
                    "Expected pads initializer size in bytes: ", 2 * data_rank * sizeof(int64_t),
                    ", actual: ", pads_initializer_raw_data.size());

  std::vector<int32_t> converted_pads_data{};
  converted_pads_data.reserve(2 * data_rank);

  auto copy_and_convert = [](const void* raw_i64_src,
                             std::back_insert_iterator<decltype(converted_pads_data)> i32_dst) {
    int64_t i64;
    memcpy(&i64, raw_i64_src, sizeof(i64));
    *i32_dst = SafeInt<int32_t>(i64);
  };

  for (size_t i = 0; i < data_rank; ++i) {
    copy_and_convert(&pads_initializer_raw_data[i * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));

    copy_and_convert(&pads_initializer_raw_data[(i + data_rank) * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));
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

    std::vector<uint8_t> pad_value_raw_data{};
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*constant_value_initializer, pad_value_raw_data));
    // assume constant_value_initializer has float data
    // ONNX spec says it matches `data` input type, and op support checker limits that to float
    ORT_RETURN_IF_NOT(pad_value_raw_data.size() == sizeof(float),
                      "Expected constant_value initializer size in bytes: ", sizeof(float),
                      ", actual size: ", pad_value_raw_data.size());
    memcpy(&pad_value, pad_value_raw_data.data(), sizeof(float));
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, pad_value);

  const auto& output = outputs[0].node_arg.Name();

  ORT_RETURN_IF_ERROR(shaper.Pad(data, converted_pads_data, output));

  const OperandType output_operand_type{operand_types.at(data).type, shaper[output]};
  const auto op_code = ANEURALNETWORKS_PAD_V2;

  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

}  // namespace nnapi
}  // namespace onnxruntime
